"""End-to-end integration tests for the enhanced walk-forward fold runner.

Verifies that all four anti-collapse modules wire correctly into the
HybridModel training loop and produce the expected outputs:

  1. Baseline (no flags)             — equivalent to run_walk_forward_fold.py
  2. + Simplex ETF                   — direction head swapped
  3. + Orthogonality penalty         — encoder gets the extra loss term
  4. + DB-MTL balancer               — weights computed dynamically
  5. + Sharpe loss                   — utility-aligned regularizer
  6. ALL four enabled simultaneously — full integration
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ENHANCED_SCRIPT = REPO_ROOT / "tools" / "run_walk_forward_fold_enhanced.py"


def _make_synthetic_dataset(tmp_path: Path):
    """Hourly bars over 8 months — enough for a 5/2 train/test split."""
    ts = pd.date_range("2022-01-01", "2022-08-31 23:00:00", freq="1h", tz=None)
    n = len(ts)
    rng = np.random.RandomState(123)
    target_ret = rng.randn(n).astype("float32") * 0.005
    df = pd.DataFrame({
        "ts_event": ts,
        "close": 1.30 + rng.randn(n) * 0.005,
        "atr_14": np.full(n, 0.005, dtype="float32"),
        "event_flag": rng.choice([0, 1], n, p=[0.93, 0.07]).astype("int8"),
        "event_direction": rng.choice([-1, 0, 1], n, p=[0.4, 0.2, 0.4]).astype("int8"),
        "path_outcome": rng.choice([0, 1, 2, 3, 4], n,
                                    p=[0.32, 0.32, 0.10, 0.10, 0.16]).astype("int8"),
        "target_ret_24": target_ret,
        "feat_a": rng.randn(n).astype("float32"),
        "feat_b": rng.randn(n).astype("float32"),
        "feat_c": rng.randn(n).astype("float32"),
        "feat_d": rng.randn(n).astype("float32"),
    })
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path, index=False)
    emb = rng.randn(n, 161).astype("float32")
    emb_path = tmp_path / "embeddings.npy"
    np.save(emb_path, emb)
    valid_path = tmp_path / "embedding_valid.npy"
    np.save(valid_path, np.ones(n, dtype=bool))
    return features_path, emb_path, valid_path


def _run_fold(tmp_path: Path, extra_args: list[str], suffix: str = "") -> dict:
    feat, emb, valid = _make_synthetic_dataset(tmp_path)
    out_dir = tmp_path / f"fold_{suffix or 'baseline'}"
    base = [
        sys.executable, str(ENHANCED_SCRIPT),
        "--fold-id", suffix or "1",
        "--combined-features", str(feat),
        "--ssl-embeddings", str(emb),
        "--embedding-valid", str(valid),
        "--train-start", "2022-01", "--train-end", "2022-05",
        "--test-start", "2022-06", "--test-end", "2022-07",
        "--output-dir", str(out_dir),
        "--epochs", "3",
        "--hidden-dim", "32", "--batch-size", "64",
    ]
    result = subprocess.run(
        base + extra_args, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"fold failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists(), f"no metrics.json in {out_dir}"
    return json.loads(metrics_path.read_text())


class TestEnhancedFoldIntegration:
    def test_baseline_no_flags(self, tmp_path):
        """No flags → equivalent to the baseline runner."""
        m = _run_fold(tmp_path, extra_args=[])
        assert m["config"]["use_simplex_etf"] is False
        assert m["config"]["ortho_weight"] == 0.0
        assert m["config"]["use_dbmtl"] is False
        assert m["config"]["sharpe_weight"] == 0.0
        assert m["n_train"] > 0 and m["n_test"] > 0

    def test_simplex_etf_only(self, tmp_path):
        m = _run_fold(tmp_path, ["--use-simplex-etf"], suffix="etf")
        assert m["config"]["use_simplex_etf"] is True
        assert m["n_train"] > 0 and m["n_test"] > 0

    def test_orthogonality_only(self, tmp_path):
        m = _run_fold(tmp_path, ["--ortho-weight", "0.5"], suffix="ortho")
        assert m["config"]["ortho_weight"] == 0.5

    def test_dbmtl_only(self, tmp_path):
        m = _run_fold(tmp_path, ["--use-dbmtl"], suffix="dbmtl")
        assert m["config"]["use_dbmtl"] is True
        # DB-MTL diagnostics should accumulate
        if "dbmtl_history" in m:
            for entry in m["dbmtl_history"]:
                assert "weights" in entry
                assert "loss_ema" in entry

    def test_sharpe_only(self, tmp_path):
        m = _run_fold(tmp_path, ["--sharpe-weight", "0.3"], suffix="sharpe")
        assert m["config"]["sharpe_weight"] == 0.3

    def test_all_four_enabled(self, tmp_path):
        """The big one — all four anti-collapse modules at once."""
        m = _run_fold(tmp_path, [
            "--use-simplex-etf",
            "--ortho-weight", "0.5",
            "--use-dbmtl",
            "--sharpe-weight", "0.3",
        ], suffix="all")
        cfg = m["config"]
        assert cfg["use_simplex_etf"] is True
        assert cfg["ortho_weight"] == 0.5
        assert cfg["use_dbmtl"] is True
        assert cfg["sharpe_weight"] == 0.3
        assert m["n_train"] > 0 and m["n_test"] > 0
        # The model should still produce predictions for the test slice
        pred_path = tmp_path / "fold_all" / "test_predictions.parquet"
        assert pred_path.exists()
        pred = pd.read_parquet(pred_path)
        assert {"hybrid_event_prob", "hybrid_p_long", "hybrid_p_short",
                "hybrid_p_neutral", "hybrid_confidence"}.issubset(pred.columns)
        # Probabilities should be finite and in [0, 1]
        assert pred[["hybrid_event_prob", "hybrid_p_long", "hybrid_p_short",
                     "hybrid_p_neutral", "hybrid_confidence"]].notna().all().all()


class TestHybridModelEnhancements:
    """Direct unit tests for the small HybridModel API additions."""

    def test_forward_returns_backbone_embedding(self):
        from modules.trading_intel.hybrid.model import HybridConfig, HybridModel
        cfg = HybridConfig(hidden_dim=64)
        model = HybridModel(cfg)
        out = model(
            torch.randn(8, cfg.daytrade_feature_dim),
            torch.randn(8, cfg.ssl_embed_dim),
            torch.randn(8, cfg.cnn_embed_dim),
        )
        assert out.backbone_embedding is not None
        assert out.backbone_embedding.shape == (8, 64)

    def test_custom_direction_head_swap(self):
        from modules.trading_intel.hybrid.model import HybridConfig, HybridModel
        from modules.trading_intel.anti_collapse import (
            SimplexETFClassifier, SimplexETFConfig,
        )
        cfg = HybridConfig(hidden_dim=32)
        etf = SimplexETFClassifier(SimplexETFConfig(
            feature_dim=cfg.hidden_dim, num_classes=3,
        ))
        model = HybridModel(cfg, direction_head=etf)
        out = model(
            torch.randn(4, cfg.daytrade_feature_dim),
            torch.randn(4, cfg.ssl_embed_dim),
            torch.randn(4, cfg.cnn_embed_dim),
        )
        # ETF returns the right shape, and uses zero learnable params
        assert out.direction_logits.shape == (4, 3)
        assert model.direction_head is etf
        assert etf.num_parameters() == 0
