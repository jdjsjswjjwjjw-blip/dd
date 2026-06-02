"""D4 — LOB tensor ↔ MBP-10 fidelity harness tests.

The from_mbp tensor READS MBP-10 directly (no MBO→book reconstruction exists —
see validate_lob_vs_mbp.py docstring). The RULE 5 risk on the current code is
that the 9-channel ENCODING silently corrupts the real book (swap bid↔ask,
reverse levels, mangle sizes). These tests prove the validator:
  • PASSES on a faithful build,
  • TEETH: FAILS on a planted bid↔ask swap and on a planted level reversal,
  • (ب) asserts the mbo_only fallback emits zero depth channels (no fake book).

Exercised on synthetic MBP/MBO/bars with ONE snapshot per bar (so the
imbalance-blend selection is exact). The operator runs the same harness on real
MBO+MBP before any depth training (data lives on their machine).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as P
from tools.diagnostics.validate_lob_vs_mbp import (
    check_lob_tensor_vs_mbp, assert_mbo_only_has_zero_depth,
)


def _synth(n_bars=12, levels=10, seed=0):
    """One MBP snapshot + a couple of trades per bar → exact-match validation."""
    rng = np.random.RandomState(seed)
    bar_ts = pd.date_range("2025-04-07 08:00", periods=n_bars, freq="5min", tz="UTC")
    bars = pd.DataFrame({"ts_event": bar_ts, "close": 100.0})
    mbp_rows, mbo_rows = [], []
    for bi, t0 in enumerate(bar_ts):
        ts = t0 + pd.Timedelta(seconds=30)
        mid = 100.0 + bi * 0.1
        row = {"ts_event": ts}
        for L in range(levels):
            row[f"bid_px_{L:02d}"] = mid - 0.01 * (L + 1)
            row[f"ask_px_{L:02d}"] = mid + 0.01 * (L + 1)
            row[f"bid_sz_{L:02d}"] = float(rng.randint(1, 90))   # distinct per level
            row[f"ask_sz_{L:02d}"] = float(rng.randint(1, 90))
        mbp_rows.append(row)
        mbo_rows.append({"ts_event": ts, "action": "T", "side": "A",
                         "price": mid, "size": 5.0})
    return bars, pd.DataFrame(mbp_rows), pd.DataFrame(mbo_rows)


def _raw_tensor(bars, mbp, mbo, levels=10):
    return P.build_rolling_lob_tensors_from_mbp(
        mbo, mbp, bars, freq="5min", lookback_bars=5, levels=levels, normalize=False,
    )


class TestFaithfulPasses:
    def test_correct_encoding_passes(self):
        bars, mbp, mbo = _synth()
        tensors, ts, _ = _raw_tensor(bars, mbp, mbo)
        rep = check_lob_tensor_vs_mbp(tensors, ts, mbp, freq="5min", levels=10)
        assert rep["verdict"] == "PASS", rep
        assert rep["n_bars_checked"] > 0
        assert not rep["swap_detected"]


class TestTeeth:
    def test_bid_ask_swap_is_caught(self):
        bars, mbp, mbo = _synth(seed=1)
        tensors, ts, _ = _raw_tensor(bars, mbp, mbo)
        corrupt = tensors.copy()
        corrupt[..., [3, 4]] = corrupt[..., [4, 3]]      # swap bid/ask depth channels
        rep = check_lob_tensor_vs_mbp(corrupt, ts, mbp, freq="5min", levels=10)
        assert rep["verdict"] == "FAIL"
        assert rep["swap_detected"], rep            # zero-region leak exposes the swap

    def test_level_reversal_is_caught(self):
        bars, mbp, mbo = _synth(seed=2)
        tensors, ts, _ = _raw_tensor(bars, mbp, mbo)
        corrupt = tensors.copy()
        # reverse the bid-depth ordering within the bid half (levels 0..9 of ch3)
        corrupt[:, :, :10, 3] = corrupt[:, :, 9::-1, 3]
        rep = check_lob_tensor_vs_mbp(corrupt, ts, mbp, freq="5min", levels=10)
        assert rep["verdict"] == "FAIL", rep
        assert rep["max_range_violation"] > 1e-3

    def test_size_mangle_is_caught(self):
        bars, mbp, mbo = _synth(seed=3)
        tensors, ts, _ = _raw_tensor(bars, mbp, mbo)
        corrupt = tensors.copy()
        corrupt[:, :, :10, 3] += 2.0                # inflate decoded bid sizes
        rep = check_lob_tensor_vs_mbp(corrupt, ts, mbp, freq="5min", levels=10)
        assert rep["verdict"] == "FAIL"


class TestMboOnlyZeroDepth:
    def test_mbo_only_depth_channels_are_zero(self):
        bars, _, mbo = _synth(seed=4)
        tensors, _, _ = P.build_rolling_lob_tensors_mbo_only(
            mbo, bars, freq="5min", lookback_bars=5, n_levels=20,
        )
        assert_mbo_only_has_zero_depth(tensors)      # must not raise

    def test_fabricated_depth_would_be_caught(self):
        bars, _, mbo = _synth(seed=5)
        tensors, _, _ = P.build_rolling_lob_tensors_mbo_only(
            mbo, bars, freq="5min", lookback_bars=5, n_levels=20,
        )
        tensors[0, 0, 0, 3] = 1.0                     # plant fake bid depth
        try:
            assert_mbo_only_has_zero_depth(tensors)
            raise AssertionError("should have flagged fabricated depth")
        except AssertionError as e:
            assert "non-zero" in str(e)
