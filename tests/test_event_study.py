"""Teeth tests for tools/diagnostics/event_study.py.

The sharpest risk in an event study is look-ahead in the EVENT DEFINITION
(selecting bars by outcome = guaranteed fake edge). The first two tests target
exactly that, then we cover the null test and the verdict logic:

  T1. LOOK-AHEAD GUARD: detect_sweep_events at bar i uses ONLY level[i], low[i],
      close[i], atr[i] — mutating any bar AFTER i must NOT change which bars are
      flagged (truncate-invariance). This proves the pattern is pre-move defined.
  T2. DEFINITION-NOT-OUTCOME: events are detected on bars whose FORWARD return is
      both positive and negative — i.e. detection does not secretly select
      winners.
  T3. PLANTED NO-EDGE: random-walk close with sweeps planted at random → null
      test must reject (no edge), verdict NO_EDGE / VOLATILITY / UNDERPOWERED.
  T4. PLANTED BOUNCE: sweeps deterministically followed by an up-move → verdict
      EDGE_DIRECTIONAL, bounce rate above null.
  T5. R1 forward-metrics window: outcome uses only (i, i+h]; mutating bars after
      i+h does not change the measured forward metrics.
  T6. classify thresholds locked (a priori).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.event_study import (
    detect_sweep_events, _forward_metrics, _null_metrics, _classify,
    run_event_study,
    MIN_EVENTS, BOUNCE_RATE_EDGE, DIRECTIONAL_VS_VOL,
)


def _frame(n, close, level, *, low=None, high=None, atr=1e-3, level_col="london_sess_low"):
    ts = pd.date_range("2025-04-01 08:00", periods=n, freq="5min", tz="UTC")
    close = np.asarray(close, dtype=float)
    low = close - 0.0005 if low is None else np.asarray(low, float)
    high = close + 0.0005 if high is None else np.asarray(high, float)
    df = pd.DataFrame({
        "ts_event": ts, "close": close, "high": high, "low": low,
        "atr_14": np.full(n, atr),
        level_col: np.asarray(level, float),
    })
    return df


# ── T1: LOOK-AHEAD GUARD on event detection (the cardinal sin) ─────────────
class TestLookAheadGuard:
    def test_detection_invariant_to_future(self):
        rng = np.random.RandomState(0)
        n = 500
        close = 1.25 + np.cumsum(rng.randn(n) * 1e-4)
        level = pd.Series(close).rolling(20, min_periods=1).min().shift(1).bfill().to_numpy()
        low = close - np.abs(rng.randn(n)) * 5e-4
        df = _frame(n, close, level, low=low)
        ev_a = detect_sweep_events(df, "london_sess_low", 0.15, require_reclaim=False)

        # Mutate EVERYTHING after bar 300 — close/low/high/level
        df_mut = df.copy()
        for col in ("close", "low", "high", "london_sess_low"):
            df_mut.loc[df_mut.index[300:], col] = 999.0
        ev_b = detect_sweep_events(df_mut, "london_sess_low", 0.15, require_reclaim=False)
        # bars [0..300] flagged-or-not must be IDENTICAL
        np.testing.assert_array_equal(ev_a[:300], ev_b[:300])

    def test_reclaim_uses_close_at_i_not_future(self):
        n = 200
        rng = np.random.RandomState(1)
        close = 1.25 + np.cumsum(rng.randn(n) * 1e-4)
        level = pd.Series(close).rolling(10, min_periods=1).min().shift(1).bfill().to_numpy()
        low = close - np.abs(rng.randn(n)) * 6e-4
        df = _frame(n, close, level, low=low)
        ev_a = detect_sweep_events(df, "london_sess_low", 0.15, require_reclaim=True)
        df_mut = df.copy()
        df_mut.loc[df_mut.index[150:], ["close", "low", "high"]] = 999.0
        ev_b = detect_sweep_events(df_mut, "london_sess_low", 0.15, require_reclaim=True)
        np.testing.assert_array_equal(ev_a[:150], ev_b[:150])


# ── T2: detection selects by definition, not by outcome ───────────────────
class TestDefinitionNotOutcome:
    def test_events_include_both_winners_and_losers(self):
        """Plant sweeps; half are followed by up-moves, half by down-moves.
        Detection (pre-move) must flag BOTH — it cannot be selecting winners."""
        n = 600
        rng = np.random.RandomState(2)
        close = np.full(n, 1.25)
        level = np.full(n, 1.25)
        low = close.copy()
        # plant a sweep every 20 bars: low pierces level by 0.5 ATR
        atr = 1e-3
        ev_positions = list(range(40, n - 60, 20))
        for k, i in enumerate(ev_positions):
            low[i] = level[i] - 0.6 * atr            # pierce (sweep)
            # outcome: alternate up / down move over next bars
            direction = +1 if k % 2 == 0 else -1
            close[i + 1: i + 30] = 1.25 + direction * 0.5 * atr
        df = _frame(n, close, level, low=low, atr=atr)
        events = detect_sweep_events(df, "london_sess_low", 0.5, require_reclaim=False)
        idx = np.where(events)[0]
        assert len(idx) >= 20, f"expected ~{len(ev_positions)} sweeps, got {len(idx)}"
        # forward returns of detected events must contain BOTH signs
        fwd = np.array([(close[i + 6] - close[i]) / close[i] for i in idx if i + 6 < n])
        assert (fwd > 0).any() and (fwd < 0).any(), "detection appears to select only winners"


# ── T3: planted NO-edge → null rejects ─────────────────────────────────────
class TestPlantedNoEdge:
    def test_random_sweeps_no_edge(self):
        n = 8000
        rng = np.random.RandomState(3)
        close = 1.25 + np.cumsum(rng.randn(n) * 1e-4)
        level = pd.Series(close).rolling(60, min_periods=1).min().shift(1).bfill().to_numpy()
        low = close - np.abs(rng.randn(n)) * 4e-4
        high = close + np.abs(rng.randn(n)) * 4e-4
        df = _frame(n, close, level, low=low, high=high)
        events = detect_sweep_events(df, "london_sess_low", 0.15, require_reclaim=False)
        idx = np.where(events)[0]
        if len(idx) < MIN_EVENTS:
            return  # underpowered is an acceptable outcome for this synthetic
        m = _forward_metrics(df, idx, h=12, tp_sl_atr=1.0)
        nu = _null_metrics(df, idx, h=12, tp_sl_atr=1.0, n_null=200, seed=0)
        verdict = _classify(m, nu)
        assert verdict in {"NO_EDGE", "VOLATILITY_NOT_DIRECTION", "UNDERPOWERED"}, (verdict, m, nu)


# ── T4: planted BOUNCE → EDGE_DIRECTIONAL ──────────────────────────────────
class TestPlantedBounce:
    def test_planted_bounce_is_detected(self):
        n = 9000
        rng = np.random.RandomState(4)
        atr = 1e-3
        close = np.full(n, 1.25) + np.cumsum(rng.randn(n) * 1e-5)
        level = pd.Series(close).rolling(60, min_periods=1).min().shift(1).bfill().to_numpy()
        low = close - np.abs(rng.randn(n)) * 2e-4
        high = close + np.abs(rng.randn(n)) * 2e-4
        # plant sweeps that RELIABLY bounce up over the next ~12 bars
        for i in range(80, n - 60, 40):
            low[i] = level[i] - 0.6 * atr                 # pierce
            bounce = 1.2 * atr                            # strong up-move
            close[i + 1: i + 13] = close[i] + bounce
            high[i + 1: i + 13] = close[i] + bounce + 1e-4
        df = _frame(n, close, level, low=low, high=high, atr=atr)
        events = detect_sweep_events(df, "london_sess_low", 0.5, require_reclaim=False)
        idx = np.where(events)[0]
        assert len(idx) >= MIN_EVENTS, f"need >= {MIN_EVENTS} events, got {len(idx)}"
        m = _forward_metrics(df, idx, h=12, tp_sl_atr=1.0)
        nu = _null_metrics(df, idx, h=12, tp_sl_atr=1.0, n_null=200, seed=0)
        assert m["bounce_rate"] > BOUNCE_RATE_EDGE, m
        assert _classify(m, nu) == "EDGE_DIRECTIONAL", (m, nu)


# ── T5: forward metrics use only (i, i+h] ──────────────────────────────────
class TestForwardWindowCausal:
    def test_metrics_invariant_to_beyond_horizon(self):
        n = 400
        rng = np.random.RandomState(5)
        close = 1.25 + np.cumsum(rng.randn(n) * 1e-4)
        level = np.full(n, close.min() - 1.0)            # never triggers; pick manual idx
        df = _frame(n, close, level)
        idx = np.array([50, 100, 150])
        m_a = _forward_metrics(df, idx, h=12, tp_sl_atr=1.0)
        # mutate everything beyond i+h for the LAST event (150+12=162)
        df_mut = df.copy()
        df_mut.loc[df_mut.index[170:], ["close", "high", "low"]] = 999.0
        m_b = _forward_metrics(df_mut, idx[:2], h=12, tp_sl_atr=1.0)   # events 50,100 unaffected
        # events 50 & 100 forward windows end at 62 & 112 — far before 170
        assert m_a["mean_fwd_return"] is not None
        m_a2 = _forward_metrics(df, idx[:2], h=12, tp_sl_atr=1.0)
        assert abs(m_a2["mean_fwd_return"] - m_b["mean_fwd_return"]) < 1e-12


# ── T6: classify thresholds locked ────────────────────────────────────────
class TestClassifyThresholds:
    def test_underpowered_below_min_events(self):
        assert _classify({"n": MIN_EVENTS - 1}, {}) == "UNDERPOWERED"

    def test_no_edge_when_bounce_low(self):
        m = {"n": 200, "bounce_rate": 0.50, "mean_fwd_return": 0.0,
             "mean_abs_fwd_return": 0.001, "win_rate_tp_before_sl": 0.50}
        nu = {"bounce_p95": 0.56, "mean_ret_std": 0.001, "mean_ret_mean": 0.0}
        assert _classify(m, nu) in {"NO_EDGE", "VOLATILITY_NOT_DIRECTION"}

    def test_run_end_to_end_smoke(self, tmp_path):
        n = 6000
        rng = np.random.RandomState(7)
        close = 1.25 + np.cumsum(rng.randn(n) * 1e-4)
        level = pd.Series(close).rolling(60, min_periods=1).min().shift(1).bfill().to_numpy()
        low = close - np.abs(rng.randn(n)) * 4e-4
        high = close + np.abs(rng.randn(n)) * 4e-4
        ts = pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC")
        df = pd.DataFrame({"ts_event": ts, "close": close, "high": high, "low": low,
                           "atr_14": np.full(n, 1e-3),
                           "london_sess_low": level, "pdl": level})
        path = tmp_path / "f.parquet"
        df.to_parquet(path, index=False)
        s = run_event_study(path, tmp_path / "out",
                            depths_atr=(0.15, 0.3), horizons=(6, 12), n_null=50)
        assert "cells" in s and len(s["cells"]) > 0
        report = (tmp_path / "out" / "event_study_report.txt").read_text(encoding="utf-8")
        assert "Event study" in report
