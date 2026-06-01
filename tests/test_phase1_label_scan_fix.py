"""Tests for Phase 1 / Commit A1 — symmetric single-pair barrier scan
inside `label_by_outcome` (Finding #7: LONG-biased four-barrier was making
`short_tp` mathematically impossible).

The tests drive `label_by_outcome` on small synthetic frames that hit
each branch deterministically:
  - A clean upward path → `path_outcome=0` (long_tp), `bias_label=0`
  - A clean downward path → `path_outcome=1` (short_tp), `bias_label=1`
  - A flat path → `path_outcome=4` (timeout), `bias_label=2` (NEUTRAL)

Source-level guards lock in:
  - The new scan markers ("Finding #7", "symmetric single-pair") stay
  - The dead `long_sl` / `short_sl` branches do NOT come back
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as pdt


# ── helpers ─────────────────────────────────────────────────────────────────
def _make_bars(closes: list[float], *, atr: float = 0.001) -> pd.DataFrame:
    """Build a minimal OHLC frame matching label_by_outcome's expectations.

    We use high=low=close (no intra-bar wicks) so barrier hits are driven
    purely by the close path — easiest to reason about in tests.
    """
    n = len(closes)
    closes_arr = np.asarray(closes, dtype=np.float64)
    ts = pd.date_range("2025-04-01 08:00", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "ts_event":      ts,
        "open":          closes_arr,
        "high":          closes_arr,
        "low":           closes_arr,
        "close":         closes_arr,
        "atr_14":        np.full(n, atr, dtype=np.float64),
        # required-by-signature columns; values picked to be neutral
        "is_event":      np.ones(n, dtype=np.int8),
        "event_score":   np.zeros(n, dtype=np.float64),
        "kalman_direction": np.zeros(n, dtype=np.int8),
        "event_direction": np.zeros(n, dtype=np.int8),
        "is_london":     np.ones(n, dtype=bool),
        "is_overlap":    np.zeros(n, dtype=bool),
        "regime_label":  np.array(["ranging"] * n),
        "session":       np.array(["london"] * n),
    })


def _label(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run label_by_outcome with the veto disabled — A1 tests scan logic,
    not veto behaviour (that's A2's territory)."""
    return pdt.label_by_outcome(
        df,
        default_tp_mult=1.5,
        default_sl_mult=1.0,
        ignore_event_direction_veto=True,
        **kwargs,
    )


class TestSymmetricScanBranches:
    """Each path_outcome branch the new scan can produce."""

    def test_upward_path_yields_long_tp(self):
        # entry=1.0, ATR=0.001, tp_mult=1.5 → upper = 1.0015
        # path crosses 1.0015 at bar 2
        df = _make_bars([1.0, 1.0010, 1.0020, 1.0025, 1.0030, 1.0030])
        out = _label(df)
        assert int(out["bias_label"].iloc[0]) == 0       # LONG
        assert int(out["path_outcome"].iloc[0]) == 0     # long_tp

    def test_downward_path_yields_short_tp(self):
        # Finding #7 regression: a clean downward move was being recorded
        # as `long_sl` (path=2) under the old four-barrier scan, since
        # sl_long at -1.0 ATR was hit before tp_short at -1.5 ATR and
        # LONG was evaluated first. Symmetric single-pair must now record
        # this as short_tp (path=1).
        df = _make_bars([1.0, 0.9990, 0.9980, 0.9975, 0.9970, 0.9970])
        out = _label(df)
        assert int(out["bias_label"].iloc[0]) == 1       # SHORT
        assert int(out["path_outcome"].iloc[0]) == 1     # short_tp

    def test_flat_path_yields_timeout_neutral(self):
        df = _make_bars([1.0] * 8)
        out = _label(df)
        assert int(out["bias_label"].iloc[0]) == 2       # NEUTRAL
        assert int(out["path_outcome"].iloc[0]) == 4     # timeout
        # Reason should be TIMEOUT (no veto → not KALMAN)
        assert int(out["neutral_reason"].iloc[0]) == pdt.NEUTRAL_REASON_TIMEOUT


class TestNoMoreLongSlOrShortSlInOutput:
    """The dead path_outcome codes (2, 3) must never appear on real paths."""

    def test_random_paths_emit_only_long_tp_short_tp_or_timeout(self):
        np.random.seed(42)
        n = 500
        # random walk centered at 1.0, enough volatility to hit barriers
        steps = np.random.randn(n) * 0.0008
        closes = 1.0 + np.cumsum(steps)
        df = _make_bars(closes.tolist(), atr=0.001)
        out = _label(df)

        # Only codes {0, 1, 4} should appear on rows past the warmup edge
        seen = set(int(x) for x in out["path_outcome"].unique())
        forbidden = {2, 3, 5, 6}
        intersect = seen & forbidden
        assert not intersect, (
            f"Dead path_outcome codes appeared: {intersect}. "
            f"Symmetric scan must only emit {{0,1,4}}; seen={seen}."
        )


class TestPhase1SourceGuards:
    """Lock in the A1 fix at source level — guards against silent regression."""

    def _src(self) -> str:
        return (REPO_ROOT / "prepare_day_trading.py").read_text()

    def test_a1_markers_present(self):
        src = self._src()
        # Whichever paraphrasing future edits use, these tokens must stay
        for marker in ("Finding #7", "symmetric single-pair", "barrier_mult"):
            assert marker in src, f"A1 marker missing: {marker!r}"

    def test_dead_four_barrier_scan_did_not_come_back(self):
        src = self._src()
        # The exact dead expressions from the old scan
        forbidden_exprs = [
            "tp_long  = entry + tp_dist",
            "sl_long  = entry - sl_dist",
            "tp_short = entry - tp_dist",
            "sl_short = entry + sl_dist",
        ]
        for expr in forbidden_exprs:
            assert expr not in src, (
                f"A1 regressed — the old four-barrier scan is back: {expr!r}"
            )

    def test_dead_long_sl_short_sl_branches_did_not_come_back(self):
        src = self._src()
        for marker in ("elif ev_type == 'long_sl':", "elif ev_type == 'short_sl':"):
            assert marker not in src, (
                f"A1 regressed — dead branch reappeared: {marker!r}"
            )
