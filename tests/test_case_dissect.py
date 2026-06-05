"""Teeth tests for tools/diagnostics/case_dissect.py.

This is a DESCRIPTIVE microscope (one case), so the tests verify it locates the
right window and computes the descriptive numbers correctly — NOT a verdict.

  T1. find_sweep_window locates the planted bottom near the target low, with the
      correct before/after window.
  T2. absorption_proxy is HIGH when signed flow is heavy but price is flat, and
      LOW when price moves with the flow (the absorption signature).
  T3. CVD turn: a selling-then-buying cumulative-CVD path is flagged turned_up.
  T4. date filtering restricts the search window.
  T5. end-to-end writes a utf-8 report + a per-bar CSV, and the report carries
      the R7 'one case, not predictive' disclaimer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.case_dissect import (
    find_sweep_window, dissect, summarize_at_sweep, run_case_dissect,
    DEFAULT_TICK,
)


def _frame(n, *, close, low=None, bar_cvd_delta=None, start="2025-05-01 00:00"):
    ts = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    close = np.asarray(close, float)
    low = close - 0.0003 if low is None else np.asarray(low, float)
    df = pd.DataFrame({
        "ts_event": ts, "open": close, "high": close + 0.0003, "low": low,
        "close": close, "volume": np.full(n, 100.0),
        "tick_count": np.full(n, 200.0),
        "bar_cvd_delta": (np.zeros(n) if bar_cvd_delta is None else np.asarray(bar_cvd_delta, float)),
        "cvd": np.cumsum(np.zeros(n) if bar_cvd_delta is None else np.asarray(bar_cvd_delta, float)),
        "order_flow_imbalance": np.zeros(n),
        "pwl": np.full(n, close.min()), "pdl": np.full(n, close.min()),
    })
    return df


# ── T1: window location ────────────────────────────────────────────────────
class TestFindWindow:
    def test_locates_planted_bottom(self):
        n = 300
        close = 1.32 + np.zeros(n)
        low = close.copy()
        low[150] = 1.3124                      # the planted sweep bottom
        df = _frame(n, close=close, low=low)
        window, sweep_idx = find_sweep_window(df, target_low=1.3124, before=20, after=40)
        # the sweep bar within the window must be the planted bottom
        assert abs(float(window["low"].iloc[sweep_idx]) - 1.3124) < 1e-9
        assert len(window) == 20 + 40 + 1

    def test_falls_back_to_global_min_if_target_not_reached(self):
        n = 100
        close = 1.32 + np.zeros(n)
        low = close.copy()
        low[60] = 1.315                        # never reaches 1.3124
        df = _frame(n, close=close, low=low)
        window, sweep_idx = find_sweep_window(df, target_low=1.3124, before=10, after=10)
        assert abs(float(window["low"].iloc[sweep_idx]) - 1.315) < 1e-9


# ── T2: absorption proxy ───────────────────────────────────────────────────
class TestAbsorptionProxy:
    def test_high_when_flow_heavy_price_flat(self):
        n = 50
        close = np.full(n, 1.3124)             # price PINNED
        dcvd = np.zeros(n); dcvd[25] = 500.0   # heavy signed flow at i=25
        df = _frame(n, close=close, bar_cvd_delta=dcvd)
        w = dissect(df)
        # at i=25: huge flow, ~zero price move → absorption proxy must be large
        assert w["absorption_proxy"].iloc[25] > 1000, w["absorption_proxy"].iloc[25]

    def test_low_when_price_moves_with_flow(self):
        n = 50
        # price moves WITH the flow (big Δprice per bar) → low absorption ratio.
        # bar Δclose = 0.0020 (20 ticks), flow = 50 → proxy = 50/0.0020 = 25,000.
        # The PINNED case (price flat) gives 500/floor = ~10,000,000 — so the
        # discriminator is the RATIO between the two regimes, not an absolute.
        close = 1.3124 + np.arange(n) * 0.0020          # +20 ticks every bar
        dcvd = np.full(n, 50.0)
        df = _frame(n, close=close, bar_cvd_delta=dcvd)
        w_moving = dissect(df)

        # contrast: same flow, price pinned
        df_pinned = _frame(n, close=np.full(n, 1.3124), bar_cvd_delta=dcvd)
        w_pinned = dissect(df_pinned)

        # absorption proxy must be MUCH larger when price is pinned than moving:
        # pinned 50/floor(0.5 tick)=1e6 vs moving 50/(20 ticks)=25e3 → ~40× gap.
        assert w_pinned["absorption_proxy"].iloc[25] > 20 * w_moving["absorption_proxy"].iloc[25], (
            w_pinned["absorption_proxy"].iloc[25], w_moving["absorption_proxy"].iloc[25])


# ── T3: CVD turn ───────────────────────────────────────────────────────────
class TestCvdTurn:
    def test_selling_then_buying_flagged(self):
        n = 60
        close = np.full(n, 1.3124)
        # cumulative CVD: falling (selling) for first half, rising (buying) after
        dcvd = np.concatenate([np.full(30, -10.0), np.full(30, +10.0)])
        df = _frame(n, close=close, bar_cvd_delta=dcvd)
        w = dissect(df)
        summ = summarize_at_sweep(w, sweep_idx=30, span=12)
        assert summ["cvd_slope_pre"] < 0
        assert summ["cvd_slope_post"] > 0
        assert summ["cvd_turned_up"] is True

    def test_continued_selling_not_flagged(self):
        n = 60
        close = np.full(n, 1.3124)
        dcvd = np.full(n, -10.0)               # pure selling throughout
        df = _frame(n, close=close, bar_cvd_delta=dcvd)
        w = dissect(df)
        summ = summarize_at_sweep(w, sweep_idx=30, span=12)
        assert summ["cvd_turned_up"] is False


# ── T4: date filter ────────────────────────────────────────────────────────
class TestDateFilter:
    def test_date_window_restricts(self):
        n = 600                                 # ~2 days of 5-min bars
        close = 1.32 + np.zeros(n)
        low = close.copy()
        low[100] = 1.3124                       # in-range bottom (day 1)
        low[500] = 1.3100                       # lower bottom but OUT of date range
        df = _frame(n, close=close, low=low, start="2025-05-01 00:00")
        # restrict to first day only
        window, sweep_idx = find_sweep_window(
            df, target_low=1.3124, date_start="2025-05-01", date_end="2025-05-01 23:59",
            before=10, after=10)
        # must pick the in-range 1.3124, NOT the out-of-range 1.3100
        assert abs(float(window["low"].iloc[sweep_idx]) - 1.3124) < 1e-9


# ── T5: end-to-end ─────────────────────────────────────────────────────────
class TestEndToEnd:
    def test_writes_report_and_csv_with_disclaimer(self, tmp_path):
        n = 300
        close = 1.32 + np.cumsum(np.full(n, -1e-5))
        low = close.copy(); low[150] = 1.3124
        dcvd = np.concatenate([np.full(150, -8.0), np.full(150, +8.0)])
        df = _frame(n, close=close, low=low, bar_cvd_delta=dcvd)
        path = tmp_path / "f.parquet"
        df.to_parquet(path, index=False)
        s = run_case_dissect(path, tmp_path / "out", target_low=1.3124,
                             date_start=None, date_end=None, before=30, after=60)
        # outputs exist
        assert (tmp_path / "out" / "case_dissect_series.csv").exists()
        report = (tmp_path / "out" / "case_dissect_report.txt").read_text(encoding="utf-8")
        # R7 disclaimer must be present (the cardinal honesty guard)
        assert "DESCRIPTIVE, NOT PREDICTIVE" in report
        assert "ONE CASE" in s["DISCLAIMER"]
        # series CSV is readable + has the reconstructed columns
        series = pd.read_csv(tmp_path / "out" / "case_dissect_series.csv")
        assert "absorption_proxy" in series.columns
        assert "cvd_cum_window" in series.columns
