"""tests/test_edge_scanner_v2.py — Phase A parity + speedup tests.

يتأكد:
  ① النتائج مطابقة 1:1 بين edge_scanner.scan_edges و scan_edges_v2
  ② الـ pre-computed masks مطابقة لنتائج _apply_combo
  ③ vectorized version أسرع بشكل ملموس

شغّل من v19_2/:
    cd v19_2 && python tests/test_edge_scanner_v2.py
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from edge_scanner import scan_edges, _apply_combo, FILTER_COMBOS, LEVEL_NAMES, EVENT_TYPES
from edge_scanner_v2 import (
    scan_edges_v2,
    _precompute_combo_masks,
    _precompute_event_masks,
    _precompute_zone_masks,
)


def _make_synthetic_df(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """يولّد DataFrame شبيه بمخرج session_mapper مع features ثابتة."""
    rng = np.random.RandomState(seed)
    ts_start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    ts = pd.date_range(ts_start, periods=n, freq="1min")
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
    zones = ["asia_q1", "asia_q2", "london_q1", "london_q2", "ny_q1", "ny_q2"]
    df["zone_full"] = rng.choice(zones, n)
    for level in LEVEL_NAMES:
        df[f"{level}_event"] = rng.choice(EVENT_TYPES + ["none"], n)
    return df


class TestComboMaskParity(unittest.TestCase):
    """Phase A: pre-computed combo masks مطابقة لنتائج _apply_combo."""

    def test_combo_mask_equals_apply_combo(self):
        df = _make_synthetic_df(n=500)
        combo_masks = _precompute_combo_masks(df)
        for combo_name, direction, filts in FILTER_COMBOS:
            pre_mask = combo_masks[combo_name]
            apply_mask = _apply_combo(df, filts).to_numpy()
            np.testing.assert_array_equal(
                pre_mask, apply_mask,
                err_msg=f"combo {combo_name} mismatch",
            )

    def test_combo_mask_on_subset_matches(self):
        """ضرب global mask مع cell mask = نتيجة _apply_combo على cell."""
        df = _make_synthetic_df(n=500)
        combo_masks = _precompute_combo_masks(df)
        rng = np.random.RandomState(0)
        cell_mask = rng.rand(len(df)) < 0.3
        cell_df = df[cell_mask].copy()
        for combo_name, direction, filts in FILTER_COMBOS[:5]:
            orig_mask = _apply_combo(cell_df, filts).to_numpy()
            vec_in_cell = combo_masks[combo_name][cell_mask]
            np.testing.assert_array_equal(
                vec_in_cell, orig_mask,
                err_msg=f"combo {combo_name} subset mismatch",
            )


class TestScanEdgesParity(unittest.TestCase):
    """v1 و v2 يُعطون نفس النتائج النهائية."""

    def test_scan_parity_small(self):
        df = _make_synthetic_df(n=1500, seed=7)
        r1 = scan_edges(df, horizons=[3, 6], run_permutation=False, verbose=False)
        r2 = scan_edges_v2(df, horizons=[3, 6], run_permutation=False, verbose=False)
        self.assertEqual(
            r1["diagnostics"]["cells_scanned"],
            r2["diagnostics"]["cells_scanned"],
            msg=f"cells_scanned mismatch",
        )
        self.assertEqual(
            r1["diagnostics"]["stage1_candidates"],
            r2["diagnostics"]["stage1_candidates"],
        )
        self.assertEqual(
            r1["diagnostics"]["stage2_passed_fdr"],
            r2["diagnostics"]["stage2_passed_fdr"],
        )

    def test_scan_parity_candidate_keys(self):
        df = _make_synthetic_df(n=1500, seed=11)
        r1 = scan_edges(df, horizons=[3, 6], run_permutation=False, verbose=False)
        r2 = scan_edges_v2(df, horizons=[3, 6], run_permutation=False, verbose=False)
        keys1 = sorted([(c["zone"], c["level"], c["event"], c["combo"], c["horizon"])
                        for c in r1["candidates"]])
        keys2 = sorted([(c["zone"], c["level"], c["event"], c["combo"], c["horizon"])
                        for c in r2["candidates"]])
        self.assertEqual(keys1, keys2, msg="final candidate keys differ")


class TestScanEdgesSpeed(unittest.TestCase):
    """Phase A: v2 أسرع بشكل ملموس."""

    def test_v2_faster_than_v1(self):
        df = _make_synthetic_df(n=3000, seed=42)

        t0 = time.perf_counter()
        r1 = scan_edges(df, horizons=[3, 6, 12], run_permutation=False, verbose=False)
        t_v1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        r2 = scan_edges_v2(df, horizons=[3, 6, 12], run_permutation=False, verbose=False)
        t_v2 = time.perf_counter() - t0

        speedup = t_v1 / max(t_v2, 1e-9)
        print(f"\n   ⏱️  n=3000: v1={t_v1*1000:.0f}ms  v2={t_v2*1000:.0f}ms  speedup={speedup:.1f}×")
        self.assertGreater(
            speedup, 1.5,
            msg=f"v2 ليس أسرع بما يكفي: speedup={speedup:.2f}×",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
