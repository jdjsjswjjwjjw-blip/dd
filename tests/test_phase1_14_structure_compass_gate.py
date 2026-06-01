"""Phase 1.14 — Structure + Liquidity-Compass event gate tests.

The new gate defines an event as a STRUCTURALLY significant moment confirmed
by institutional pressure:

    is_event = (active session ∧ near key level) ∧ (OFI | Δ-divergence | VWAP-z)

with NO manufactured min_event_rate floor. These tests pin each conjunct
independently so a regression in any one (session gating, level proximity, or
the compass confirmation) is caught, and prove the weekly levels (pwh/pwl)
now feed both the gate and the divergence feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as P


# ── the shared divergence helper ────────────────────────────────────────────
class TestDeltaDivergenceHelper:
    def test_bullish_at_ceiling_with_short_delta(self):
        close = np.array([100.0, 100.0])
        atr = np.array([1.0, 1.0])
        cvd = np.array([-5.0, +5.0])          # short, long
        levels = [(np.array([100.0, 100.0]), -1)]   # ceiling
        div = P._compute_delta_divergence_at_level(close, atr, cvd, levels)
        assert div[0] == +1, "near ceiling + short delta → bullish divergence"
        assert div[1] == 0, "near ceiling + long delta → no bullish divergence"

    def test_bearish_at_floor_with_long_delta(self):
        close = np.array([100.0, 100.0])
        atr = np.array([1.0, 1.0])
        cvd = np.array([+5.0, -5.0])
        levels = [(np.array([100.0, 100.0]), +1)]   # floor
        div = P._compute_delta_divergence_at_level(close, atr, cvd, levels)
        assert div[0] == -1, "near floor + long delta → bearish divergence"
        assert div[1] == 0

    def test_far_from_level_no_divergence(self):
        close = np.array([100.0])
        atr = np.array([1.0])
        cvd = np.array([-5.0])
        levels = [(np.array([120.0]), -1)]    # 20 ATR away
        div = P._compute_delta_divergence_at_level(close, atr, cvd, levels)
        assert div[0] == 0


def _gate_frame() -> pd.DataFrame:
    """5 bars, each isolating one gate condition (see assertions below)."""
    return pd.DataFrame({
        "ts_event": pd.date_range("2025-04-07 08:00", periods=5, freq="5min", tz="UTC"),
        "close": [100.0, 100.0, 100.0, 100.0, 100.0],
        "atr_14": [1.0, 1.0, 1.0, 1.0, 1.0],
        "is_london": [1, 0, 1, 1, 1],
        "is_overlap": [0, 0, 0, 0, 0],
        "is_ny": [0, 0, 0, 0, 0],
        "order_flow_imbalance": [0.5, 0.5, 0.9, 0.0, 0.0],
        "vwap_z_score": [0.0, 0.0, 0.0, 0.0, 0.0],
        "bar_cvd_delta": [0.0, 0.0, 0.0, 0.0, +5.0],
        "pdh": [100.0, 100.0, 120.0, 100.0, 120.0],
        "pdl": [80.0, 80.0, 80.0, 80.0, 80.0],
        "pwh": [120.0, 120.0, 120.0, 120.0, 120.0],
        "pwl": [80.0, 80.0, 80.0, 80.0, 100.0],   # row4: near weekly floor
        "london_sess_high": [120.0, 120.0, 120.0, 120.0, 120.0],
        "london_sess_low": [80.0, 80.0, 80.0, 80.0, 80.0],
    })


class TestGateConjuncts:
    def test_output_contract(self):
        out = P.detect_structure_compass_events(_gate_frame())
        for c in ("is_event", "event_score", "event_score_continuous", "event_score_binary"):
            assert c in out.columns
        assert set(np.unique(out["is_event"])) <= {0, 1}

    def test_each_condition_isolated(self):
        """row0: active+near+OFI → EVENT
           row1: near+OFI but INACTIVE session → no
           row2: active+OFI but FAR from levels → no
           row3: active+near but NO compass → no
           row4: active+near weekly-floor + Δ-divergence → EVENT (weekly wired)"""
        out = P.detect_structure_compass_events(_gate_frame())
        got = out["is_event"].to_numpy().tolist()
        assert got == [1, 0, 0, 0, 1], f"gate conjuncts wrong: {got}"

    def test_weekly_level_drives_divergence_event(self):
        """row4 fires ONLY because pwl (weekly) is near + delta is long — if
        weekly levels weren't wired into the gate, row4 would be a non-event."""
        out = P.detect_structure_compass_events(_gate_frame())
        assert out["is_event"].iloc[4] == 1


class TestNoManufacturedFloor:
    def test_zero_events_when_nothing_qualifies(self):
        """A quiet frame (active sessions but no level proximity, no pressure)
        must yield ZERO events — the old gate would manufacture up to 12%."""
        n = 200
        rng = np.random.RandomState(0)
        df = pd.DataFrame({
            "ts_event": pd.date_range("2025-04-07 08:00", periods=n, freq="5min", tz="UTC"),
            "close": np.full(n, 100.0),
            "atr_14": np.full(n, 1.0),
            "is_london": np.ones(n, dtype=int),
            "is_overlap": np.zeros(n, dtype=int),
            "is_ny": np.zeros(n, dtype=int),
            "order_flow_imbalance": np.zeros(n),          # no pressure
            "vwap_z_score": np.zeros(n),
            "bar_cvd_delta": np.zeros(n),
            "pdh": np.full(n, 130.0), "pdl": np.full(n, 70.0),   # all far
            "pwh": np.full(n, 130.0), "pwl": np.full(n, 70.0),
            "london_sess_high": np.full(n, 130.0), "london_sess_low": np.full(n, 70.0),
        })
        out = P.detect_structure_compass_events(df)
        assert int(out["is_event"].sum()) == 0, "no floor → no manufactured events"


class TestCausality:
    def test_per_bar_truncate_invariance(self):
        """The gate is pure per-bar elementwise — bar t's verdict can't depend
        on bars after t. Truncating must leave earlier verdicts unchanged."""
        rng = np.random.RandomState(2)
        n = 120
        df = pd.DataFrame({
            "ts_event": pd.date_range("2025-04-07 08:00", periods=n, freq="5min", tz="UTC"),
            "close": 100 + np.cumsum(rng.randn(n) * 0.5),
            "atr_14": np.full(n, 1.0),
            "is_london": np.ones(n, dtype=int),
            "is_overlap": np.zeros(n, dtype=int),
            "is_ny": np.zeros(n, dtype=int),
            "order_flow_imbalance": rng.uniform(-1, 1, n),
            "vwap_z_score": rng.randn(n) * 2,
            "bar_cvd_delta": rng.randn(n) * 3,
            "pdh": 100 + rng.randn(n), "pdl": 99 + rng.randn(n),
            "pwh": 101 + rng.randn(n), "pwl": 98 + rng.randn(n),
            "london_sess_high": 100.5 + rng.randn(n), "london_sess_low": 99.5 + rng.randn(n),
        })
        full = P.detect_structure_compass_events(df)["is_event"].to_numpy()
        for cut in (40, 80, 119):
            trunc = P.detect_structure_compass_events(df.iloc[: cut + 1].copy())["is_event"].to_numpy()
            assert np.array_equal(full[: cut + 1], trunc), f"verdict changed at cut={cut}"
