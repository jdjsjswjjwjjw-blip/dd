"""Tests for walk-forward fold runner + aggregator."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ════════════════════════════════════════════════════════════════════
# Aggregator
# ════════════════════════════════════════════════════════════════════
class TestAggregator:
    def _write_fold(self, base: Path, fold_id: int, rule_hit: float,
                    hyb_hit: float, rule_n: int = 30, hyb_n: int = 18):
        d = base / f"fold_{fold_id}"
        d.mkdir(parents=True, exist_ok=True)
        m = {
            "fold_id": str(fold_id),
            "train_period": f"2022-01..2022-{fold_id:02d}",
            "test_period": f"2022-{fold_id+1:02d}..2022-{fold_id+1:02d}",
            "n_train": 1000 * fold_id,
            "n_test": 200,
            "rule_baseline": {
                "n_events": rule_n,
                "wins": int(round(rule_n * rule_hit)),
                "losses": rule_n - int(round(rule_n * rule_hit)),
                "hit_rate": rule_hit,
            },
            "hybrid_filtered": {
                "n_events": hyb_n,
                "wins": int(round(hyb_n * hyb_hit)),
                "losses": hyb_n - int(round(hyb_n * hyb_hit)),
                "hit_rate": hyb_hit,
            },
            "lift_hit_rate": hyb_hit - rule_hit,
        }
        (d / "metrics.json").write_text(json.dumps(m))

    def test_aggregate_basic(self, tmp_path):
        from tools.aggregate_walk_forward import aggregate
        # 3 folds with varying hit rates
        self._write_fold(tmp_path, 1, 0.75, 0.82)
        self._write_fold(tmp_path, 2, 0.78, 0.85)
        self._write_fold(tmp_path, 3, 0.80, 0.79)
        s = aggregate(tmp_path)
        assert s["n_folds"] == 3
        assert abs(s["rule_baseline_hit_rate"]["mean"] - 0.7766666) < 0.001
        assert abs(s["hybrid_filtered_hit_rate"]["mean"] - 0.8200000) < 0.001
        # 2 folds with positive lift (fold 1 + 2), 1 with negative (fold 3)
        assert s["stability_pct_positive_lift"] == pytest.approx(2 / 3)

    def test_empty_dir_returns_error(self, tmp_path):
        from tools.aggregate_walk_forward import aggregate
        s = aggregate(tmp_path)
        assert "error" in s

    def test_cli_subprocess(self, tmp_path):
        self._write_fold(tmp_path, 1, 0.75, 0.80)
        self._write_fold(tmp_path, 2, 0.78, 0.82)
        out_path = tmp_path / "summary.json"
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "aggregate_walk_forward.py"),
             "--folds-dir", str(tmp_path), "--output", str(out_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out_path.exists()
        s = json.loads(out_path.read_text())
        assert s["n_folds"] == 2


# ════════════════════════════════════════════════════════════════════
# Fold runner — synthetic end-to-end smoke
# ════════════════════════════════════════════════════════════════════
class TestFoldRunnerSmoke:
    def _make_synthetic_dataset(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        """Build a tiny synthetic combined parquet + ssl embeddings + valid mask
        covering 2022-01..2022-08 with hourly bars (fast for the test)."""
        ts = pd.date_range("2022-01-01", "2022-08-31 23:00:00", freq="1h", tz=None)
        n = len(ts)
        rng = np.random.RandomState(42)

        # Build a minimal day_trading_features parquet
        df = pd.DataFrame({
            "ts_event": ts,
            "close": 1.30 + rng.randn(n) * 0.005,
            "atr_14": np.full(n, 0.005),
            "event_flag": rng.choice([0, 1], n, p=[0.93, 0.07]).astype("int8"),
            "event_direction": rng.choice([-1, 0, 1], n, p=[0.4, 0.2, 0.4]).astype("int8"),
            "path_outcome": rng.choice([0, 1, 2, 3, 4], n,
                                        p=[0.32, 0.32, 0.10, 0.10, 0.16]).astype("int8"),
            # A few generic features
            "feat_a": rng.randn(n).astype("float32"),
            "feat_b": rng.randn(n).astype("float32"),
            "feat_c": rng.randn(n).astype("float32"),
        })
        features_path = tmp_path / "features.parquet"
        df.to_parquet(features_path, index=False)

        # SSL embeddings: 161-dim random
        emb = rng.randn(n, 161).astype("float32")
        emb_path = tmp_path / "embeddings.npy"
        np.save(emb_path, emb)
        valid = np.ones(n, dtype=bool)
        valid_path = tmp_path / "embedding_valid.npy"
        np.save(valid_path, valid)

        return features_path, emb_path, valid_path

    def test_fold_runs_end_to_end(self, tmp_path):
        feat, emb, valid = self._make_synthetic_dataset(tmp_path)
        out_dir = tmp_path / "fold_1"
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "run_walk_forward_fold.py"),
             "--fold-id", "1",
             "--combined-features", str(feat),
             "--ssl-embeddings", str(emb),
             "--embedding-valid", str(valid),
             "--train-start", "2022-01",
             "--train-end", "2022-05",
             "--test-start", "2022-06",
             "--test-end", "2022-07",
             "--output-dir", str(out_dir),
             "--epochs", "3",          # fast for test
             "--hidden-dim", "32",
             "--batch-size", "64"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"fold runner failed:\n{result.stderr}"
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "test_predictions.parquet").exists()
        m = json.loads((out_dir / "metrics.json").read_text())
        assert m["fold_id"] == "1"
        assert m["train_period"] == "2022-01..2022-05"
        assert m["test_period"] == "2022-06..2022-07"
        assert m["n_train"] > 0
        assert m["n_test"] > 0
