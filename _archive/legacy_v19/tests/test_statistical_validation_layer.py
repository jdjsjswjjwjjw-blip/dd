"""tests/test_statistical_validation_layer.py — Phase 4 tests."""
from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd


def _make_synthetic_for_discovery(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """يحاكي مخرج enrich_features (مع zones, levels, sim_*)."""
    rng = np.random.RandomState(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 100.0 + np.cumsum(rng.randn(n) * 0.05)
    df = pd.DataFrame({
        "ts_event": ts,
        "close": close,
        "sim_depth_pressure":  rng.uniform(-1, 1, n),
        "sim_informed_prob":   rng.uniform(0, 1, n),
        "sim_absorb_intensity": rng.uniform(0, 1, n),
        "sim_flow_direction":  rng.uniform(-1, 1, n),
        "sim_depth_imbalance": rng.uniform(-0.5, 0.5, n),
        "sim_wall_growth":     rng.uniform(-0.5, 0.5, n),
        "sim_wall_consumed":   rng.uniform(0, 1, n),
        "sim_wall_persist":    rng.uniform(0, 1, n),
        "sim_wall_shift":      rng.uniform(-0.5, 0.5, n),
        "sim_wall_bid_level":  rng.uniform(0, 10, n),
        "sim_wall_ask_level":  rng.uniform(0, 10, n),
        "sim_wall_bid_size":   rng.uniform(0, 1, n),
        "sim_wall_ask_size":   rng.uniform(0, 1, n),
        "sim_iceberg_prob":    rng.uniform(0, 1, n),
        "sim_iceberg_side":    rng.uniform(-1, 1, n),
        "sim_iceberg_replenish": rng.uniform(0, 1, n),
        "sim_iceberg_stealth": rng.uniform(0, 1, n),
        "sim_iceberg_strength": rng.uniform(0, 1, n),
    })
    zones = ["asia_q1", "asia_q2", "london_q1", "ny_q1"]
    df["zone_full"] = rng.choice(zones, n)
    from edge_scanner import LEVEL_NAMES, EVENT_TYPES
    for level in LEVEL_NAMES:
        df[f"{level}_event"] = rng.choice(EVENT_TYPES + ["none"], n)
    return df


class TestAlphaSetDataclass(unittest.TestCase):
    def test_from_empty_result(self):
        from modules.statistical_validation_layer import AlphaSet
        result = {"candidates": [], "criteria": {}, "diagnostics": {}, "split_info": {}}
        a = AlphaSet.from_scan_result(result)
        self.assertEqual(len(a.candidates), 0)
        self.assertFalse(a.is_valid())

    def test_summary(self):
        from modules.statistical_validation_layer import AlphaSet
        a = AlphaSet(
            candidates=[
                {"zone": "asia", "direction": 1, "combo": "x"},
                {"zone": "ny", "direction": -1, "combo": "y"},
            ],
        )
        s = a.summary()
        self.assertEqual(s["n_alphas"], 2)
        self.assertEqual(s["directions"]["long"], 1)
        self.assertEqual(s["directions"]["short"], 1)


class TestDiscoverAlphas(unittest.TestCase):
    """integration test: تشغيل discovery على synthetic data."""

    def test_runs_without_error_on_random(self):
        """على random data، عادي 0 alphas تنجح (لا يوجد signal)."""
        from modules.statistical_validation_layer import discover_alphas
        df = _make_synthetic_for_discovery(n=1500)
        alpha_set = discover_alphas(df, run_permutation=False, verbose=False)
        # ما يرمي exception
        self.assertIsNotNone(alpha_set)
        # diagnostics بـ stage counts
        self.assertIn("cells_scanned", alpha_set.diagnostics)

    def test_summary_structure(self):
        from modules.statistical_validation_layer import discover_alphas
        df = _make_synthetic_for_discovery(n=1500)
        alpha_set = discover_alphas(df, run_permutation=False, verbose=False)
        s = alpha_set.summary()
        for key in ("n_alphas", "diagnostics", "split", "directions"):
            self.assertIn(key, s)


class TestApplyAlphaFilter(unittest.TestCase):
    def test_empty_alpha_set_returns_empty_df(self):
        from modules.statistical_validation_layer import AlphaSet, apply_alpha_filter
        df = _make_synthetic_for_discovery(n=500)
        empty_set = AlphaSet(candidates=[])
        out = apply_alpha_filter(df, empty_set)
        self.assertEqual(len(out), 0)
        # columns تبقى نفسها
        self.assertEqual(list(out.columns), list(df.columns))

    def test_returns_mask_when_requested(self):
        from modules.statistical_validation_layer import AlphaSet, apply_alpha_filter
        df = _make_synthetic_for_discovery(n=500)
        empty_set = AlphaSet(candidates=[])
        out, mask = apply_alpha_filter(df, empty_set, return_mask=True)
        self.assertEqual(len(mask), len(df))
        self.assertEqual(mask.dtype, np.bool_)

    def test_manual_alpha_filter(self):
        """ضع alpha يدوياً وتأكد إن الـ filter يلتقط الـ rows الصحيحة."""
        from modules.statistical_validation_layer import AlphaSet, apply_alpha_filter
        df = _make_synthetic_for_discovery(n=500)
        # alpha يدوي: combo دائماً يلتقط حالة depth_pressure_pos
        alpha = {
            "zone": "asia_q1", "level": "PDH", "event": "touch",
            "combo": "LONG_dp_pos", "direction": 1, "horizon": 6,
            "filters": ["depth_pressure_pos"],
        }
        alpha_set = AlphaSet(candidates=[alpha])
        filtered, mask = apply_alpha_filter(df, alpha_set, return_mask=True)
        # mask يجب أن يحدد فقط:
        # zone == asia_q1 AND PDH_event == touch AND sim_depth_pressure > 0.20
        expected = (
            (df["zone_full"] == "asia_q1").to_numpy()
            & (df["PDH_event"] == "touch").to_numpy()
            & (df["sim_depth_pressure"] > 0.20).to_numpy()
        )
        np.testing.assert_array_equal(mask, expected)


class TestSignalToNoise(unittest.TestCase):
    def test_snr_estimate_with_fwd_ret(self):
        from modules.statistical_validation_layer import (
            AlphaSet, signal_to_noise_estimate
        )
        df = _make_synthetic_for_discovery(n=500)
        # أضف fwd_ret_6
        close = df["close"].to_numpy()
        h = 6
        ret = np.full(len(close), np.nan)
        ret[:-h] = (close[h:] - close[:-h]) / close[:-h]
        df["fwd_ret_6"] = ret

        empty_set = AlphaSet(candidates=[])
        snr = signal_to_noise_estimate(df, empty_set, horizon=6)
        self.assertIn("snr_all", snr)
        self.assertGreaterEqual(snr["snr_all"], 0.0)
        # filtered = 0 لـ empty set
        self.assertEqual(snr["n_filtered"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
