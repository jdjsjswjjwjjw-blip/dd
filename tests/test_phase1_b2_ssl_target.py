"""Tests for Phase 1 / Commit B2 — continuous SSL directional target.

`compute_ssl_directional_target` adds next_price_delta (log return over a
fixed H-bar horizon) + next_price_delta_valid. Gate-free (every bar gets
a value) but right-edge / session-break aware for the validity mask.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as pdt


def _bars(closes, *, session_break_at=None):
    n = len(closes)
    sb = np.zeros(n, dtype=np.int8)
    if session_break_at is not None:
        sb[session_break_at] = 1
    return pd.DataFrame({
        "close": np.asarray(closes, dtype=np.float64),
        "is_session_break": sb,
    })


class TestDirectionalTargetValues:
    def test_columns_added(self):
        out = pdt.compute_ssl_directional_target(_bars([1.0] * 10), horizon_bars=3)
        assert "next_price_delta" in out.columns
        assert "next_price_delta_valid" in out.columns

    def test_log_return_value_correct(self):
        closes = [1.0, 1.0, 1.0, 1.10]  # H=3 → delta[0] = log(1.10/1.0)
        out = pdt.compute_ssl_directional_target(_bars(closes), horizon_bars=3)
        expected = float(np.log(1.10 / 1.0))
        assert abs(float(out["next_price_delta"].iloc[0]) - expected) < 1e-6
        assert bool(out["next_price_delta_valid"].iloc[0]) is True

    def test_pct_mode_value_correct(self):
        # clip raised to 1.0 so the 20% jump isn't truncated
        closes = [1.0, 1.0, 1.20]
        out = pdt.compute_ssl_directional_target(
            _bars(closes), horizon_bars=2, use_log=False, clip=1.0,
        )
        assert abs(float(out["next_price_delta"].iloc[0]) - 0.20) < 1e-6

    def test_gate_free_every_bar_has_value(self):
        # Dead chop: every bar gets a value (0.0), not NaN
        out = pdt.compute_ssl_directional_target(_bars([1.0] * 12), horizon_bars=4)
        assert out["next_price_delta"].notna().all()
        assert (out["next_price_delta"] == 0.0).all()

    def test_sign_direction(self):
        up = pdt.compute_ssl_directional_target(_bars([1.0, 1.0, 1.05]), horizon_bars=2)
        dn = pdt.compute_ssl_directional_target(_bars([1.0, 1.0, 0.95]), horizon_bars=2)
        assert float(up["next_price_delta"].iloc[0]) > 0
        assert float(dn["next_price_delta"].iloc[0]) < 0


class TestValidityMask:
    def test_right_edge_invalid(self):
        # Last H bars have no full forward window → invalid
        out = pdt.compute_ssl_directional_target(_bars([1.0] * 10), horizon_bars=3)
        v = out["next_price_delta_valid"].to_numpy()
        assert not v[-1] and not v[-2] and not v[-3]
        assert v[0]  # first bar has a full window

    def test_session_break_in_window_invalidates(self):
        # break at bar 2 falls inside row 0's window (bars 1..4) → invalid
        out = pdt.compute_ssl_directional_target(
            _bars([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], session_break_at=2),
            horizon_bars=4,
        )
        assert bool(out["next_price_delta_valid"].iloc[0]) is False

    def test_break_outside_window_stays_valid(self):
        out = pdt.compute_ssl_directional_target(
            _bars([1.0, 1.0, 1.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                  session_break_at=8),
            horizon_bars=2,
        )
        assert bool(out["next_price_delta_valid"].iloc[0]) is True

    def test_clip_bounds(self):
        # Huge jump must be clipped to ±clip (float32 tolerance)
        out = pdt.compute_ssl_directional_target(
            _bars([1.0, 1.0, 100.0]), horizon_bars=2, clip=0.1,
        )
        assert abs(float(out["next_price_delta"].iloc[0])) <= 0.1 + 1e-4


class TestNonDestructive:
    def test_does_not_touch_existing_columns(self):
        df = _bars([1.0, 1.01, 1.02, 1.03])
        df["bias_label"] = np.array([0, 1, 2, 0], dtype=np.int8)
        df["exec_label"] = np.array([0, 0, 0, 0], dtype=np.int8)
        out = pdt.compute_ssl_directional_target(df, horizon_bars=2)
        np.testing.assert_array_equal(out["bias_label"], df["bias_label"])
        np.testing.assert_array_equal(out["exec_label"], df["exec_label"])

    def test_empty_frame_safe(self):
        out = pdt.compute_ssl_directional_target(pd.DataFrame({"close": []}))
        assert "next_price_delta" in out.columns
        assert len(out["next_price_delta"]) == 0
