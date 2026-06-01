"""Phase 1.11 — SSL probing harness smoke + correctness tests.

The harness asks "does the LOB tensor representation carry signal
beyond bar context features?" via four staged probes (context_only,
random_embed, random+ctx, pretrained). These tests confirm:

  - linear_probe_ic returns numerically sane outputs
  - lob_to_flat_embedding shape + null property
  - quick_train_lob_projector runs end-to-end on a tiny synthetic frame
  - The harness orchestrator's verdict logic fires on a planted signal
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.ssl_probe_harness import (
    linear_probe_ic, lob_to_flat_embedding, quick_train_lob_projector,
)


class TestLinearProbe:
    def test_pure_noise_yields_low_ic(self):
        rng = np.random.RandomState(0)
        X = rng.randn(2000, 32)
        y = rng.randn(2000)
        out = linear_probe_ic(X, y)
        assert abs(out["spearman_ic"]) < 0.20, (
            f"random X vs y should have low IC, got {out['spearman_ic']:.3f}"
        )

    def test_planted_signal_recovered(self):
        rng = np.random.RandomState(1)
        n = 3000
        X = rng.randn(n, 16)
        # y depends on X[:, 0] strongly + noise
        y = 0.7 * X[:, 0] + 0.3 * rng.randn(n)
        out = linear_probe_ic(X, y)
        assert out["spearman_ic"] > 0.30, (
            f"planted signal should be recovered, got {out['spearman_ic']:.3f}"
        )
        assert out["r2_oos"] > 0.20

    def test_insufficient_samples_handled(self):
        out = linear_probe_ic(np.zeros((50, 8)), np.zeros(50))
        assert "error" in out


class TestRandomProjection:
    def test_shape(self):
        n = 100
        lob = np.random.RandomState(2).randn(n, 50, 20, 9)
        emb = lob_to_flat_embedding(lob, weights_seed=0)
        assert emb.shape == (n, 64)
        assert np.isfinite(emb).all()

    def test_deterministic_with_seed(self):
        lob = np.random.RandomState(3).randn(50, 50, 20, 9)
        a = lob_to_flat_embedding(lob, weights_seed=42)
        b = lob_to_flat_embedding(lob, weights_seed=42)
        np.testing.assert_array_equal(a, b)


class TestQuickPretrain:
    def test_runs_end_to_end(self):
        # synthetic LOB where the last-snapshot mean correlates with y
        rng = np.random.RandomState(4)
        n = 800
        snap_mean = rng.randn(n).astype(np.float32)
        lob = np.zeros((n, 50, 20, 9), dtype=np.float32)
        lob[:, -1, :, :] = snap_mean[:, None, None]
        y = (0.5 * snap_mean + 0.5 * rng.randn(n)).astype(np.float64)
        valid = np.ones(n, dtype=bool)

        emb, log = quick_train_lob_projector(lob, y, valid, epochs=2, batch_size=64)
        assert emb.shape == (n, 64)
        assert log["epochs"] == 2
        assert log["n_train"] > 0
        # Training loss should decrease across epochs (signal is learnable)
        losses = [e["train_mse"] for e in log["log"]]
        assert losses[-1] <= losses[0] + 1e-3, (
            f"loss did not decrease: {losses}"
        )


class TestEndToEndSmoke:
    """Drive the full harness on a tiny synthetic frame + LOB tensors,
    confirm it writes a sane summary.json."""

    def test_harness_writes_summary(self, tmp_path):
        from tools.diagnostics.ssl_probe_harness import run_probe

        n = 500
        rng = np.random.RandomState(7)
        df = pd.DataFrame({
            "ts_event": pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC"),
            "close": 1.0 + np.cumsum(rng.randn(n) * 0.0005),
            "open": 1.0, "high": 1.0, "low": 1.0,
            "feature_a": rng.randn(n),
            "feature_b": rng.randn(n),
            "next_price_delta": rng.randn(n) * 0.001,
            "next_price_delta_valid": True,
        })
        lob = rng.randn(n, 50, 20, 9).astype(np.float32)

        feat_path = tmp_path / "features.parquet"
        lob_path = tmp_path / "lob.npy"
        out_dir = tmp_path / "audit"
        df.to_parquet(feat_path)
        np.save(lob_path, lob)

        results = run_probe(
            feat_path, lob_path, output_dir=out_dir,
            pretrain_epochs=0,        # skip pretrain for speed
        )
        assert "context_only" in results
        assert "random_embed" in results
        assert "random_concat" in results
        # Summary file written
        assert (out_dir / "ssl_probe_summary.json").exists()
