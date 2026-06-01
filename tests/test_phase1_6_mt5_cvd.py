"""Phase 1.4 core — MT5-style CVD features (5 cols).

The audit confirmed bar-level flow carries signal (volume STRONG/MODERATE,
tick_count STRONG, volume_burst MODERATE). The existing `cvd` is a
cumulative line and the `cvd_slope_*` variants were dropped as WEAK.
Phase 1.4 adds five explicit per-bar features derived from bar_cvd_delta:

  1. cvd_bar_5m              — signed volume delta per bar
  2. cvd_direction_ratio_5m  — |delta|/total ∈ [0,1]
  3. cvd_intensity_vs_atr    — |delta|/ATR (normalised intensity)
  4. cvd_divergence_at_level — ±1 at structural ceilings/floors
  5. cvd_consecutive_imbalance — streak counter of same-sign strong bars

Each test seeds the inputs deterministically and verifies output
correctness + NaN robustness + the schema spec exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as pdt
from modules.dataset_schema import (
    PHASE_1_4_NEW_FEATURES, ALL_FEATURE_SPECS,
    FAMILY_CVD_MT5, IC_VERDICT_NEW,
)


def _cvd_input_frame(n: int = 20, *, bar_cvd_delta=None) -> pd.DataFrame:
    if bar_cvd_delta is None:
        bar_cvd_delta = np.linspace(-100, 100, n, dtype=np.float64)
    return pd.DataFrame({
        'ts_event': pd.date_range("2025-04-01 08:00", periods=n, freq="5min", tz="UTC"),
        'open': 1.0, 'high': 1.001, 'low': 0.999, 'close': 1.0,
        'volume': np.full(n, 1000.0),
        'regime_label': "ranging", 'atr_14': 0.001,
        'bar_cvd_delta': bar_cvd_delta,
        'pdh': 1.001, 'pdl': 0.999,
        'london_sess_high': 1.001, 'london_sess_low': 0.999,
        'train_event_flag': np.ones(n, dtype=np.int8),
        'exec_valid': np.ones(n, dtype=bool),
        'next_price_delta_valid': np.ones(n, dtype=bool),
    })


# ── Schema integration ────────────────────────────────────────────────────
class TestSchemaIntegration:
    def test_five_specs_present(self):
        names = {s.name for s in PHASE_1_4_NEW_FEATURES}
        assert names == {
            'cvd_bar_5m',
            'cvd_direction_ratio_5m',
            'cvd_intensity_vs_atr',
            'cvd_divergence_at_level',
            'cvd_consecutive_imbalance',
        }

    def test_specs_tagged_correctly(self):
        for s in PHASE_1_4_NEW_FEATURES:
            assert s.family == FAMILY_CVD_MT5
            assert s.ic_verdict == IC_VERDICT_NEW


# ── (1) cvd_bar_5m ─────────────────────────────────────────────────────────
class TestCvdBar5m:
    def test_column_added(self):
        out = pdt._apply_phase1_engineering_fixes(_cvd_input_frame())
        assert 'cvd_bar_5m' in out.columns

    def test_equals_bar_cvd_delta(self):
        deltas = np.array([10.0, -20.0, 30.0, 0.0, -5.0])
        df = _cvd_input_frame(n=5, bar_cvd_delta=deltas)
        out = pdt._apply_phase1_engineering_fixes(df)
        np.testing.assert_allclose(
            out['cvd_bar_5m'].to_numpy(), deltas.astype(np.float32),
        )

    def test_dtype_float32(self):
        out = pdt._apply_phase1_engineering_fixes(_cvd_input_frame())
        assert out['cvd_bar_5m'].dtype == np.float32


# ── (2) cvd_direction_ratio_5m ────────────────────────────────────────────
class TestCvdDirectionRatio5m:
    def test_column_added(self):
        out = pdt._apply_phase1_engineering_fixes(_cvd_input_frame())
        assert 'cvd_direction_ratio_5m' in out.columns

    def test_ratio_formula(self):
        deltas = np.array([300.0, -500.0, 100.0])
        df = _cvd_input_frame(n=3, bar_cvd_delta=deltas)
        df['volume'] = np.array([1000.0, 1000.0, 1000.0])
        out = pdt._apply_phase1_engineering_fixes(df)
        # 300/1000=0.3, 500/1000=0.5, 100/1000=0.1
        expected = np.array([0.3, 0.5, 0.1], dtype=np.float32)
        np.testing.assert_allclose(
            out['cvd_direction_ratio_5m'].to_numpy(), expected, rtol=1e-5,
        )

    def test_clipped_to_unit_range(self):
        # |delta| > volume → ratio should clip to 1.0
        df = _cvd_input_frame(n=2, bar_cvd_delta=np.array([2000.0, -3000.0]))
        df['volume'] = np.array([1000.0, 1000.0])
        out = pdt._apply_phase1_engineering_fixes(df)
        assert (out['cvd_direction_ratio_5m'] == 1.0).all()

    def test_zero_volume_safe(self):
        df = _cvd_input_frame(n=3, bar_cvd_delta=np.array([100.0, 0.0, -50.0]))
        df['volume'] = np.array([1000.0, 0.0, 500.0])
        out = pdt._apply_phase1_engineering_fixes(df)
        # zero-volume row gets ratio=0 (not NaN)
        assert out['cvd_direction_ratio_5m'].iloc[1] == 0.0
        assert np.isfinite(out['cvd_direction_ratio_5m']).all()


# ── (3) cvd_intensity_vs_atr ──────────────────────────────────────────────
class TestCvdIntensityVsAtr:
    def test_column_added(self):
        out = pdt._apply_phase1_engineering_fixes(_cvd_input_frame())
        assert 'cvd_intensity_vs_atr' in out.columns

    def test_intensity_formula(self):
        deltas = np.array([100.0, -200.0])
        df = _cvd_input_frame(n=2, bar_cvd_delta=deltas)
        df['atr_14'] = np.array([0.001, 0.002])
        out = pdt._apply_phase1_engineering_fixes(df)
        expected = np.array([100.0 / 0.001, 200.0 / 0.002], dtype=np.float32)
        np.testing.assert_allclose(
            out['cvd_intensity_vs_atr'].to_numpy(), expected, rtol=1e-5,
        )

    def test_zero_atr_yields_zero(self):
        df = _cvd_input_frame(n=2, bar_cvd_delta=np.array([100.0, 50.0]))
        df['atr_14'] = np.array([0.0, 0.001])
        out = pdt._apply_phase1_engineering_fixes(df)
        assert out['cvd_intensity_vs_atr'].iloc[0] == 0.0
        assert np.isfinite(out['cvd_intensity_vs_atr']).all()


# ── (4) cvd_divergence_at_level ───────────────────────────────────────────
class TestCvdDivergence:
    def test_column_added(self):
        out = pdt._apply_phase1_engineering_fixes(_cvd_input_frame())
        assert 'cvd_divergence_at_level' in out.columns

    def test_bullish_divergence_at_ceiling(self):
        # Close at pdh ceiling AND cvd_bar SHORT → bullish divergence (+1)
        n = 1
        df = _cvd_input_frame(n=n, bar_cvd_delta=np.array([-500.0]))
        df['close'] = np.array([1.001])    # exactly at pdh=1.001
        df['pdh']  = np.array([1.001])
        df['atr_14'] = np.array([0.001])
        out = pdt._apply_phase1_engineering_fixes(df)
        assert int(out['cvd_divergence_at_level'].iloc[0]) == +1

    def test_bearish_divergence_at_floor(self):
        # Close at pdl floor AND cvd_bar LONG → bearish divergence (-1)
        df = _cvd_input_frame(n=1, bar_cvd_delta=np.array([+500.0]))
        df['close'] = np.array([0.999])
        df['pdl']  = np.array([0.999])
        df['atr_14'] = np.array([0.001])
        out = pdt._apply_phase1_engineering_fixes(df)
        assert int(out['cvd_divergence_at_level'].iloc[0]) == -1

    def test_no_divergence_when_far_from_levels(self):
        # Close mid-range, levels far → no divergence
        df = _cvd_input_frame(n=1, bar_cvd_delta=np.array([+500.0]))
        df['close'] = np.array([1.000])
        df['pdh'] = np.array([1.100])      # 100 ATR away
        df['pdl'] = np.array([0.900])
        df['london_sess_high'] = np.array([1.100])
        df['london_sess_low']  = np.array([0.900])
        df['atr_14'] = np.array([0.001])
        out = pdt._apply_phase1_engineering_fixes(df)
        assert int(out['cvd_divergence_at_level'].iloc[0]) == 0


# ── (5) cvd_consecutive_imbalance ─────────────────────────────────────────
class TestCvdStreak:
    def test_column_added(self):
        out = pdt._apply_phase1_engineering_fixes(_cvd_input_frame())
        assert 'cvd_consecutive_imbalance' in out.columns

    def test_streak_increments_then_resets_on_sign_flip(self):
        # 5 strong-positive bars, then 1 strong-negative
        deltas = np.array([100.0, 200.0, 300.0, 400.0, 500.0, -100.0],
                          dtype=np.float64)
        df = _cvd_input_frame(n=6, bar_cvd_delta=deltas)
        out = pdt._apply_phase1_engineering_fixes(df)
        streak = out['cvd_consecutive_imbalance'].to_numpy()
        # median |bar_cvd_delta| = median([100,200,300,400,500,100]) = 250
        # bars [100,200] → weak (< 250); [300,400,500] → strong positive;
        # [-100] → weak (< 250). Streak peaks at 3 then resets.
        # We just assert the strong-streak monotonicity:
        assert streak[2] == 1     # first strong positive
        assert streak[3] == 2
        assert streak[4] == 3
        assert streak[5] == 0     # weak bar resets


# ── Smoke: end-to-end with all 5 ──────────────────────────────────────────
class TestSmokeAllFive:
    def test_all_five_present_after_call(self):
        out = pdt._apply_phase1_engineering_fixes(_cvd_input_frame(n=30))
        for c in (
            'cvd_bar_5m', 'cvd_direction_ratio_5m', 'cvd_intensity_vs_atr',
            'cvd_divergence_at_level', 'cvd_consecutive_imbalance',
        ):
            assert c in out.columns, f"missing CVD feature: {c}"

    def test_no_nan_or_inf_in_any(self):
        out = pdt._apply_phase1_engineering_fixes(_cvd_input_frame(n=50))
        for c in (
            'cvd_bar_5m', 'cvd_direction_ratio_5m', 'cvd_intensity_vs_atr',
            'cvd_consecutive_imbalance',
        ):
            v = pd.to_numeric(out[c]).to_numpy()
            assert np.isfinite(v).all(), f"{c} contains NaN/inf"

    def test_keep_top_preserves_all_five(self):
        # Build a minimal frame with the 5 CVD cols already populated
        n = 5
        df = pd.DataFrame({
            'ts_event': pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC"),
            'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0,
            'regime_label': "r", 'atr_14': 0.001,
            'is_session_break': np.zeros(n, dtype=np.int8),
            'obi': 0.0, 'obi_net': 0.0, 'cvd': 0.0, 'cvd_cumulative': 0.0,
            'cvd_bar_5m': 0.0, 'cvd_direction_ratio_5m': 0.0,
            'cvd_intensity_vs_atr': 0.0, 'cvd_divergence_at_level': 0,
            'cvd_consecutive_imbalance': 0,
        })
        out = pdt._apply_phase1_feature_selection(df, mode='keep_top')
        for c in (
            'cvd_bar_5m', 'cvd_direction_ratio_5m', 'cvd_intensity_vs_atr',
            'cvd_divergence_at_level', 'cvd_consecutive_imbalance',
        ):
            assert c in out.columns
