"""Tests for Phase 1 / Commit B1 — strict execution labels.

`compute_strict_execution_labels` adds the execution head of the dual-
target pipeline: an MT5-faithful triple-barrier label (no MFE/MAE rescue
on timeout) plus a per-bar `exec_valid` mask the head filters loss on.
It delegates to label_engine_v2's symmetric engine with
rescue_on_timeout=False, and invalidates rows whose forward window crosses
a session break.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as pdt


def _bars(closes, *, atr=0.001, session_break_at=None):
    n = len(closes)
    c = np.asarray(closes, dtype=np.float64)
    sb = np.zeros(n, dtype=np.int8)
    if session_break_at is not None:
        sb[session_break_at] = 1
    return pd.DataFrame({
        "close": c,
        "high": c,
        "low": c,
        "atr_14": np.full(n, atr),
        "is_session_break": sb,
    })


class TestExecLabelBranches:
    def test_columns_added(self):
        out = pdt.compute_strict_execution_labels(_bars([1.0] * 10))
        for col in ("exec_label", "exec_path", "exec_valid"):
            assert col in out.columns

    def test_upward_hit_is_long_and_valid(self):
        # barrier_mult 1.5, atr 0.001 → upper = 1.0015; close[3]=1.0030 crosses
        df = _bars([1.0, 1.0010, 1.0020, 1.0030, 1.0030, 1.0030, 1.0030])
        out = pdt.compute_strict_execution_labels(df, horizon_bars=6, barrier_atr_mult=1.5)
        assert int(out["exec_label"].iloc[0]) == 0      # LONG
        assert bool(out["exec_valid"].iloc[0]) is True

    def test_downward_hit_is_short_and_valid(self):
        # Finding #7 at the execution layer: downward → SHORT, never long_sl
        df = _bars([1.0, 0.9990, 0.9980, 0.9970, 0.9970, 0.9970, 0.9970])
        out = pdt.compute_strict_execution_labels(df, horizon_bars=6, barrier_atr_mult=1.5)
        assert int(out["exec_label"].iloc[0]) == 1      # SHORT
        assert bool(out["exec_valid"].iloc[0]) is True

    def test_flat_path_is_neutral_and_masked(self):
        out = pdt.compute_strict_execution_labels(_bars([1.0] * 12), horizon_bars=6)
        assert int(out["exec_label"].iloc[0]) == 2      # NEUTRAL
        assert bool(out["exec_valid"].iloc[0]) is False  # masked out (no rescue)


class TestExecNoRescue:
    def test_no_rescue_on_timeout(self):
        # Path drifts up to MFE ~1.0 ATR but never touches 1.5-ATR barrier:
        # default mode would rescue to LONG; strict must keep it NEUTRAL+masked.
        df = _bars([1.0, 1.0005, 1.0010, 1.0008, 1.0006, 1.0004, 1.0004],
                   atr=0.001)
        out = pdt.compute_strict_execution_labels(df, horizon_bars=6, barrier_atr_mult=1.5)
        assert int(out["exec_label"].iloc[0]) == 2
        assert bool(out["exec_valid"].iloc[0]) is False

    def test_exec_path_never_uses_rescue_codes(self):
        from modules.label_engine_v2 import (
            PATH_TIMEOUT_MFE_LONG, PATH_TIMEOUT_MFE_SHORT,
        )
        np.random.seed(5)
        n = 400
        closes = 1.0 + np.cumsum(np.random.randn(n) * 0.0008)
        out = pdt.compute_strict_execution_labels(_bars(closes.tolist()), horizon_bars=6)
        paths = set(int(x) for x in out["exec_path"].unique())
        assert PATH_TIMEOUT_MFE_LONG not in paths
        assert PATH_TIMEOUT_MFE_SHORT not in paths


class TestExecSessionBreakMasking:
    def test_window_crossing_break_is_invalidated(self):
        # An upward hit that WOULD be valid, but a session break falls inside
        # row 0's window (bars 1..6) → exec_valid forced False.
        df = _bars([1.0, 1.0010, 1.0020, 1.0030, 1.0030, 1.0030, 1.0030],
                   session_break_at=2)
        out = pdt.compute_strict_execution_labels(df, horizon_bars=6, barrier_atr_mult=1.5)
        assert bool(out["exec_valid"].iloc[0]) is False

    def test_break_outside_window_does_not_invalidate(self):
        # Same hit, but break is past the horizon → stays valid.
        n = 12
        closes = [1.0, 1.0010, 1.0020, 1.0030] + [1.0030] * (n - 4)
        df = _bars(closes, session_break_at=10)
        out = pdt.compute_strict_execution_labels(df, horizon_bars=4, barrier_atr_mult=1.5)
        assert bool(out["exec_valid"].iloc[0]) is True


class TestExecAdditiveNonDestructive:
    def test_does_not_modify_bias_label(self):
        df = _bars([1.0, 1.0010, 1.0020, 1.0030, 1.0030, 1.0030])
        df["bias_label"] = np.array([0, 1, 2, 0, 1, 2], dtype=np.int8)
        out = pdt.compute_strict_execution_labels(df)
        np.testing.assert_array_equal(
            out["bias_label"].to_numpy(), df["bias_label"].to_numpy()
        )

    def test_empty_frame_safe(self):
        out = pdt.compute_strict_execution_labels(pd.DataFrame({"close": []}))
        for col in ("exec_label", "exec_path", "exec_valid"):
            assert col in out.columns
            assert len(out[col]) == 0
