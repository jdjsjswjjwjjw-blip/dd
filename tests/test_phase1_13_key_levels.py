"""Phase 1.13 — weekly key-level wiring + leak-safety tests.

THE GAP THIS CLOSES
═══════════════════
compute_daily_weekly_levels() always computed weekly levels (pwh/pwl) and
daily-low distance causally, but the refinery harvested only 4 of 9 outputs
— so the model never saw a weekly support/resistance level. These tests
pin the now-wired weekly key levels AND prove they carry no look-ahead.

THE CENTREPIECE — TRUNCATE-INVARIANCE
═════════════════════════════════════
The strongest possible leak test: the level (and its distance) at bar t must
be byte-identical whether or not any bars after t exist. We compute on the
full frame, then on the frame truncated at t, and assert the row at t matches.
If a future bar ever leaked into the level at t, this test fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.context_features import compute_daily_weekly_levels


def _multi_week_frame(n_days: int = 21, seed: int = 0) -> pd.DataFrame:
    """Hourly bars spanning `n_days` (≈3 ISO weeks) starting on a Monday."""
    rng = np.random.RandomState(seed)
    start = pd.Timestamp("2025-04-07 00:00:00")  # a Monday
    n = n_days * 24
    ts = pd.date_range(start, periods=n, freq="1h")
    price = 1.2500 + np.cumsum(rng.randn(n) * 0.0008)
    return pd.DataFrame({"ts_event": ts, "price": price})


class TestColumnsPresent:
    def test_all_weekly_outputs_emitted(self):
        out = compute_daily_weekly_levels(_multi_week_frame())
        for c in ("pdh", "pdl", "pwh", "pwl",
                  "dist_to_pdh", "dist_to_pdl", "dist_to_pwh", "dist_to_pwl",
                  "price_position", "weekly_price_position"):
            assert c in out.columns, f"missing output: {c}"

    def test_weekly_price_position_bounded(self):
        out = compute_daily_weekly_levels(_multi_week_frame())
        wpp = out["weekly_price_position"].to_numpy()
        assert np.all((wpp >= 0.0) & (wpp <= 1.0)), "weekly_price_position must be in [0,1]"


class TestWeeklyUsesPreviousWeek:
    def test_pwh_equals_prior_week_max(self):
        """A bar in week 2 must carry week 1's max as its pwh — NOT week 2's
        own (running or full) high. This is the previous-session contract."""
        df = _multi_week_frame()
        out = compute_daily_weekly_levels(df)
        ts = pd.to_datetime(df["ts_event"]).dt.tz_localize(None)
        wk = ts.dt.to_period("W").dt.start_time
        weeks = sorted(wk.unique())
        assert len(weeks) >= 2, "need ≥2 weeks for the test"

        w1_max = float(df.loc[wk == weeks[0], "price"].max())
        w1_min = float(df.loc[wk == weeks[0], "price"].min())
        w2_mask = (wk == weeks[1]).to_numpy()

        pwh_w2 = out.loc[w2_mask, "pwh"].to_numpy()
        pwl_w2 = out.loc[w2_mask, "pwl"].to_numpy()
        assert np.allclose(pwh_w2, w1_max), "week-2 pwh must equal week-1 max"
        assert np.allclose(pwl_w2, w1_min), "week-2 pwl must equal week-1 min"

    def test_pwh_not_contaminated_by_current_week(self):
        """pwh in week 2 must not equal week 2's own max (would be leakage)."""
        df = _multi_week_frame(seed=3)
        out = compute_daily_weekly_levels(df)
        ts = pd.to_datetime(df["ts_event"]).dt.tz_localize(None)
        wk = ts.dt.to_period("W").dt.start_time
        weeks = sorted(wk.unique())
        w2_mask = (wk == weeks[1]).to_numpy()
        w2_own_max = float(df.loc[wk == weeks[1], "price"].max())
        pwh_w2 = out.loc[w2_mask, "pwh"].to_numpy()
        # They should differ (the prior-week high is not the current-week high)
        assert not np.allclose(pwh_w2, w2_own_max), (
            "week-2 pwh equals week-2's own max — future leak!"
        )


class TestTruncateInvariance:
    """The decisive leak test — value at bar t is independent of bars > t."""

    def test_levels_unchanged_when_future_truncated(self):
        df = _multi_week_frame(seed=7)
        full = compute_daily_weekly_levels(df)
        n = len(df)
        cut_points = np.linspace(48, n - 1, 15).astype(int)
        cols = ["pdh", "pdl", "pwh", "pwl",
                "dist_to_pwh", "dist_to_pwl", "weekly_price_position"]
        for c in cut_points:
            trunc = compute_daily_weekly_levels(df.iloc[: c + 1].copy())
            for col in cols:
                full_val = float(full[col].iloc[c])
                trunc_val = float(trunc[col].iloc[c])
                assert np.isclose(full_val, trunc_val, equal_nan=True), (
                    f"LEAK: {col} at bar {c} changed when future was removed "
                    f"({full_val} → {trunc_val})"
                )


class TestATRDistanceWiring:
    """The refinery's engineering-fixes stage builds the ATR-normalised
    weekly distances. Drive it directly on a minimal frame."""

    def test_atr_distances_built(self):
        import prepare_day_trading as P
        rng = np.random.RandomState(1)
        n = 120
        df = pd.DataFrame({
            "ts_event": pd.date_range("2025-04-07", periods=n, freq="5min", tz="UTC"),
            "close": 1.25 + np.cumsum(rng.randn(n) * 0.0005),
            "atr_14": np.abs(rng.randn(n)) * 0.001 + 0.0005,
            "pdh": 1.26, "pdl": 1.24, "pwh": 1.27, "pwl": 1.23,
        })
        out = P._apply_phase1_engineering_fixes(df, warmup_drop_bars=0)
        for c in ("dist_to_pdh_atr", "dist_to_pdl_atr", "dist_to_pwh_atr", "dist_to_pwl_atr"):
            assert c in out.columns, f"missing ATR distance: {c}"
            assert np.isfinite(pd.to_numeric(out[c], errors="coerce")).all(), f"{c} has non-finite"

    def test_atr_distance_sign_convention(self):
        """dist_to_pwh_atr = (close - pwh)/atr → negative when price is below
        the resistance (which it should be, given pwh above close)."""
        import prepare_day_trading as P
        n = 60
        df = pd.DataFrame({
            "ts_event": pd.date_range("2025-04-07", periods=n, freq="5min", tz="UTC"),
            "close": np.full(n, 1.25),
            "atr_14": np.full(n, 0.001),
            "pdh": 1.255, "pdl": 1.245, "pwh": 1.28, "pwl": 1.22,
        })
        out = P._apply_phase1_engineering_fixes(df, warmup_drop_bars=0)
        assert (out["dist_to_pwh_atr"] < 0).all(), "below resistance → negative"
        assert (out["dist_to_pwl_atr"] > 0).all(), "above support → positive"


class TestSchemaRegistration:
    def test_new_levels_in_all_feature_specs(self):
        from modules.dataset_schema import ALL_FEATURE_SPECS
        names = {s.name for s in ALL_FEATURE_SPECS}
        for c in ("pwh", "pwl", "weekly_price_position",
                  "dist_to_pdl_atr", "dist_to_pwh_atr", "dist_to_pwl_atr"):
            assert c in names, f"{c} not registered in ALL_FEATURE_SPECS (keep_top would drop it)"
