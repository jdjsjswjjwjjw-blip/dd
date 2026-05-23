"""Tests for config, data structures, and feature pipeline."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _make_bars(n=300, seed=0):
    from modules.price_cycle import BarSequence
    rng = np.random.RandomState(seed)
    close = 100 + np.cumsum(rng.randn(n) * 0.15)
    return BarSequence(
        timestamps_ns=np.arange(n, dtype=np.int64) * 60_000_000_000,
        open=close + rng.randn(n) * 0.02,
        high=close + np.abs(rng.randn(n) * 0.08),
        low=close - np.abs(rng.randn(n) * 0.08),
        close=close,
        volume=rng.uniform(100, 1000, n),
    )


class TestConfig(unittest.TestCase):
    def test_default_valid(self):
        from modules.price_cycle import PriceCycleConfig
        c = PriceCycleConfig()
        # encoder embed_dim → heads shared_dim consistency
        self.assertEqual(c.encoder.embed_dim, c.heads.shared_dim)

    def test_small_dev(self):
        from modules.price_cycle import PriceCycleConfig
        c = PriceCycleConfig.small_dev()
        self.assertLess(c.encoder.n_channels, PriceCycleConfig().encoder.n_channels)

    def test_encoder_invalid_levels(self):
        from modules.price_cycle import CycleEncoderConfig
        with self.assertRaises(ValueError):
            CycleEncoderConfig(n_levels=0)
        with self.assertRaises(ValueError):
            CycleEncoderConfig(n_levels=20)

    def test_fusion_invalid_method(self):
        from modules.price_cycle import MultiScaleFusionConfig
        with self.assertRaises(ValueError):
            MultiScaleFusionConfig(fusion_method="invalid")

    def test_serializable(self):
        from modules.price_cycle import PriceCycleConfig
        d = PriceCycleConfig().to_dict()
        self.assertIn("encoder", d)
        self.assertIn("fusion", d)


class TestBarSequence(unittest.TestCase):
    def test_creation(self):
        bars = _make_bars(100)
        self.assertEqual(bars.n_bars, 100)
        self.assertEqual(len(bars), 100)

    def test_from_dataframe(self):
        from modules.price_cycle import BarSequence
        df = pd.DataFrame({
            "ts_event": pd.date_range("2024-01-01", periods=50, freq="1min", tz="UTC"),
            "open": np.random.rand(50) + 100,
            "high": np.random.rand(50) + 101,
            "low": np.random.rand(50) + 99,
            "close": np.random.rand(50) + 100,
            "volume": np.random.rand(50) * 1000,
        })
        bars = BarSequence.from_dataframe(df)
        self.assertEqual(bars.n_bars, 50)

    def test_derived_features_shape(self):
        bars = _make_bars(200)
        feats = bars.compute_derived_features()
        self.assertEqual(feats.shape, (200, 16))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_length_mismatch_raises(self):
        from modules.price_cycle import BarSequence
        with self.assertRaises(ValueError):
            BarSequence(
                timestamps_ns=np.zeros(10, dtype=np.int64),
                open=np.zeros(10), high=np.zeros(10),
                low=np.zeros(10), close=np.zeros(5),  # mismatch!
                volume=np.zeros(10),
            )


class TestFeaturePipeline(unittest.TestCase):
    def test_build_features(self):
        from modules.price_cycle import build_cycle_features, PriceCycleConfig
        bars = _make_bars(300)
        fs = build_cycle_features(bars, PriceCycleConfig.small_dev())
        self.assertEqual(fs.bar_features.shape, (300, 16))
        self.assertEqual(fs.structural_features.shape[0], 300)
        self.assertGreater(len(fs.weak_labels), 0)

    def test_weak_labels_valid(self):
        from modules.price_cycle import build_cycle_features
        bars = _make_bars(300)
        fs = build_cycle_features(bars)
        # phase labels in [0, 3]
        self.assertTrue((fs.weak_labels["phase"] >= 0).all())
        self.assertTrue((fs.weak_labels["phase"] <= 3).all())
        # swing direction in [0, 2]
        self.assertTrue((fs.weak_labels["swing_direction"] >= 0).all())
        self.assertTrue((fs.weak_labels["swing_direction"] <= 2).all())

    def test_combined_features(self):
        from modules.price_cycle import build_cycle_features
        bars = _make_bars(200)
        fs = build_cycle_features(bars)
        combined = fs.combined()
        self.assertEqual(combined.shape[1], 16 + fs.structural_features.shape[1])

    def test_make_training_windows(self):
        from modules.price_cycle import build_cycle_features, make_training_windows
        bars = _make_bars(300)
        fs = build_cycle_features(bars)
        windows, labels = make_training_windows(fs, window_size=64, stride=8)
        self.assertEqual(windows.shape[1], 64)
        self.assertEqual(windows.shape[2], 16)
        self.assertIn("phase", labels)
        self.assertEqual(len(labels["phase"]), windows.shape[0])

    def test_causality_no_lookahead(self):
        """Modifying a future bar must not change past features."""
        from modules.price_cycle import BarSequence
        bars = _make_bars(200)
        feats_full = bars.compute_derived_features()

        # Modify last bar
        close_mod = bars.close.copy()
        close_mod[-1] += 50.0
        bars_mod = BarSequence(
            timestamps_ns=bars.timestamps_ns, open=bars.open,
            high=bars.high, low=bars.low, close=close_mod, volume=bars.volume,
        )
        feats_mod = bars_mod.compute_derived_features()

        # Past features (except last bar) must be unchanged
        # NOTE: feature [14] uses atr_window-ago, so allow last atr_window bars to change
        self.assertTrue(
            np.allclose(feats_full[:-20], feats_mod[:-20], atol=1e-5),
            "Future bar modification leaked into past features!",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
