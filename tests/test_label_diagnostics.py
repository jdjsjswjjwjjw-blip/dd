"""Teeth tests for tools/diagnostics/label_diagnostics.py.

The diagnostic itself must be trustworthy before we believe its verdict on real
data. Tests verify each check on KNOWN synthetic scenarios:

  C1: a planted predictive baseline → NO_BUG; pure-noise baselines → POSSIBLE_BUG.
  C2: a feature that predicts raw forward_return but whose signal a (synthetic)
      triple-barrier-style label destroys → LABEL_DESTROYS_SIGNAL; a consistent
      case → "consistent".
  C3: a planted daily trend → DAILY_TREND_PRESENT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.label_diagnostics import (
    check_1_sanity, check_2_label_destruction, check_3_daily,
)


def _base(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    ts = pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC")
    close = 1.25 + np.cumsum(rng.randn(n) * 1e-4)
    return pd.DataFrame({"ts_event": ts, "close": close,
                         "is_session_break": np.zeros(n, dtype=bool)})


# ── C1 ─────────────────────────────────────────────────────────────────────
class TestCheck1Sanity:
    def test_planted_baseline_yields_no_bug(self):
        df = _base(6000, seed=1)
        # return_6b that genuinely predicts the next 6-bar return (persistence)
        fwd6 = pd.Series(df["close"]).pct_change(6).shift(-6).fillna(0).to_numpy()
        rng = np.random.RandomState(2)
        df["return_6b"] = (fwd6 * 1.5 + rng.randn(len(df)) * fwd6.std()).astype(np.float32)
        out = check_1_sanity(df, horizons=(1, 6, 24))
        assert out["any_baseline_reaches_moderate"] is True
        assert "NO_BUG" in out["verdict"]

    def test_pure_noise_baselines_flag_possible_bug(self):
        df = _base(6000, seed=3)
        rng = np.random.RandomState(4)
        for f in ("return_6b", "return_1h", "rsi_14"):
            df[f] = rng.randn(len(df)).astype(np.float32)
        out = check_1_sanity(df, horizons=(1, 6, 24))
        assert out["any_baseline_reaches_moderate"] is False
        assert "POSSIBLE_BUG" in out["verdict"]


# ── C2 ─────────────────────────────────────────────────────────────────────
class TestCheck2LabelDestruction:
    def test_label_destroys_signal_detected(self):
        """Feature predicts raw forward_return strongly, but bias_label is set
        to pure noise → the diagnostic must flag LABEL_DESTROYS_SIGNAL."""
        df = _base(8000, seed=5)
        fwd6 = pd.Series(df["close"]).pct_change(6).shift(-6).fillna(0).to_numpy()
        rng = np.random.RandomState(6)
        df["cvd_bar_5m"] = (fwd6 * 2.0 + rng.randn(len(df)) * fwd6.std() * 0.8).astype(np.float32)
        # bias_label uncorrelated with the feature (label "destroys" the signal)
        df["bias_label"] = rng.choice([0, 1, 2], size=len(df)).astype(np.int8)
        out = check_2_label_destruction(df, ("cvd_bar_5m",), horizons=(1, 6))
        assert out["best_abs_ic_raw_return"] >= 0.05, out
        assert out["best_abs_ic_bias_label"] < 0.05, out
        assert out["label_destroys_signal"] is True
        assert "LABEL_DESTROYS_SIGNAL" in out["verdict"]

    def test_consistent_when_label_preserves_signal(self):
        """Feature predicts raw return AND bias_label aligns → 'consistent'."""
        df = _base(8000, seed=7)
        fwd6 = pd.Series(df["close"]).pct_change(6).shift(-6).fillna(0).to_numpy()
        rng = np.random.RandomState(8)
        df["cvd_bar_5m"] = (fwd6 * 2.0 + rng.randn(len(df)) * fwd6.std() * 0.8).astype(np.float32)
        # bias_label aligned with sign of the same forward return
        df["bias_label"] = np.where(fwd6 > 0, 0, np.where(fwd6 < 0, 1, 2)).astype(np.int8)
        out = check_2_label_destruction(df, ("cvd_bar_5m",), horizons=(1, 6))
        assert out["best_abs_ic_raw_return"] >= 0.05
        assert out["best_abs_ic_bias_label"] >= 0.05
        assert out["label_destroys_signal"] is False
        assert "consistent" in out["verdict"]


# ── C3 ─────────────────────────────────────────────────────────────────────
class TestCheck3Daily:
    def test_planted_daily_trend_detected(self):
        """Build a close series with strong day-to-day momentum (trending
        daily) → daily momentum predicts next-day direction."""
        rng = np.random.RandomState(9)
        n_days = 90
        # daily log returns with positive autocorrelation (trend)
        daily_ret = np.zeros(n_days)
        for i in range(1, n_days):
            daily_ret[i] = 0.6 * daily_ret[i - 1] + rng.randn() * 0.01
        daily_close = 100.0 * np.exp(np.cumsum(daily_ret))
        # expand each day into 288 5-min bars (flat within day for simplicity)
        bars = []
        ts0 = pd.Timestamp("2025-04-01", tz="UTC")
        for d in range(n_days):
            for m in range(288):
                bars.append({"ts_event": ts0 + pd.Timedelta(days=d, minutes=5 * m),
                             "close": daily_close[d], "is_session_break": False})
        df = pd.DataFrame(bars)
        out = check_3_daily(df)
        assert not out.get("skipped"), out
        assert out["n_daily_bars"] >= 60
        # a genuine AR(0.6) daily trend must exceed BOTH the floor AND the null
        assert out["best_abs_daily_directional_ic"] >= 0.15, out
        assert out["above_null"] is True, out
        assert "DAILY_TREND_PRESENT" in out["verdict"]

    def test_random_daily_does_not_beat_null(self):
        """A random-walk close can produce a large |IC| by chance on ~88 daily
        samples — but it must NOT exceed the shuffle null. This is the exact
        small-sample trap the null-test in C3 was added to defend against."""
        # use many seeds; the null-test must reject ALL of them as no-trend
        for seed in (10, 11, 12, 13, 20, 42, 100):
            df = _base(90 * 288, seed=seed)
            out = check_3_daily(df)
            assert not out.get("skipped")
            assert out["above_null"] is False, (seed, out)
            assert "no clear daily directional IC" in out["verdict"], (seed, out)
