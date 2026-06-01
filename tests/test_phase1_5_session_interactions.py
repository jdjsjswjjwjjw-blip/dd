"""Phase 1.5 — II.A session × feature interactions.

The audit's per-session IC split on the two STRONG features
`hawkes_intrabar_sum` and `tick_count` showed catastrophic sign-flip:

    Session 0 (Asia)      IC = +0.060
    Session 1 (London)    IC = +0.037
    Session 2 (overlap)   IC = -0.039
    Session 3 (NY close)  IC = -0.257    ← 5× stronger AND opposite sign
    Session 5 (post NY)   IC = -0.129

The aggregate IC = -0.106 hides this 5× reversal — a model without an
interaction term sees an average that's wrong everywhere. Fix: explicit
hawkes_x_session_phase = hawkes_intrabar_sum × session_phase (and same
for tick_count). The model can now learn per-session signature.

This is implemented as an additive step inside the same
_apply_phase1_engineering_fixes function that R4 added.
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
    PHASE_1_5_INTERACTION_FEATURES, ALL_FEATURE_SPECS,
    FAMILY_INTERACTION, IC_VERDICT_NEW,
)


def _interaction_input_frame(n: int = 20) -> pd.DataFrame:
    """Frame with session_phase + the two sign-flip features."""
    return pd.DataFrame({
        'ts_event': pd.date_range("2025-04-01 08:00", periods=n, freq="5min", tz="UTC"),
        'close': 1.0, 'open': 1.0, 'high': 1.0, 'low': 1.0, 'volume': 1.0,
        'regime_label': "ranging", 'atr_14': 0.001,
        'london_sess_high': 1.005, 'current_vwap': 1.001, 'pdh': 1.002,
        'session_phase': np.arange(n, dtype=np.int8) % 6,    # cycle 0..5
        'hawkes_intrabar_sum': np.linspace(0, 1, n, dtype=np.float32),
        'tick_count': np.full(n, 100, dtype=np.int32),
        'train_event_flag': np.ones(n, dtype=np.int8),
        'exec_valid': np.ones(n, dtype=bool),
        'next_price_delta_valid': np.ones(n, dtype=bool),
    })


# ── Schema integration ────────────────────────────────────────────────────
class TestSchemaIntegration:
    def test_two_interaction_specs_present(self):
        names = {s.name for s in PHASE_1_5_INTERACTION_FEATURES}
        assert names == {'hawkes_x_session_phase', 'tick_count_x_session_phase'}

    def test_specs_tagged_interaction_and_new(self):
        for s in PHASE_1_5_INTERACTION_FEATURES:
            assert s.family == FAMILY_INTERACTION
            assert s.ic_verdict == IC_VERDICT_NEW

    def test_in_aggregate_catalog(self):
        names = {s.name for s in ALL_FEATURE_SPECS}
        for c in ('hawkes_x_session_phase', 'tick_count_x_session_phase'):
            assert c in names


# ── Functional correctness ────────────────────────────────────────────────
class TestInteractionValues:
    def test_two_interaction_cols_added(self):
        out = pdt._apply_phase1_engineering_fixes(_interaction_input_frame())
        assert 'hawkes_x_session_phase' in out.columns
        assert 'tick_count_x_session_phase' in out.columns

    def test_hawkes_interaction_formula(self):
        df = _interaction_input_frame(n=10)
        out = pdt._apply_phase1_engineering_fixes(df)
        expected = (
            df['hawkes_intrabar_sum'].to_numpy(dtype=np.float32)
            * df['session_phase'].to_numpy(dtype=np.float32)
        )
        np.testing.assert_allclose(
            out['hawkes_x_session_phase'].to_numpy(),
            expected,
            rtol=1e-5,
        )

    def test_tick_count_interaction_formula(self):
        df = _interaction_input_frame(n=10)
        out = pdt._apply_phase1_engineering_fixes(df)
        expected = (
            df['tick_count'].to_numpy(dtype=np.float32)
            * df['session_phase'].to_numpy(dtype=np.float32)
        )
        np.testing.assert_allclose(
            out['tick_count_x_session_phase'].to_numpy(),
            expected,
            rtol=1e-5,
        )

    def test_session_zero_yields_zero_interaction(self):
        """When session_phase=0, the interaction is also 0 — by design."""
        df = _interaction_input_frame(n=6)
        df['session_phase'] = 0
        out = pdt._apply_phase1_engineering_fixes(df)
        assert (out['hawkes_x_session_phase'] == 0).all()
        assert (out['tick_count_x_session_phase'] == 0).all()


# ── Graceful skip when source cols missing ────────────────────────────────
class TestMissingSourceColsHandled:
    def test_missing_session_phase_skips_both(self):
        df = _interaction_input_frame()
        df = df.drop(columns=['session_phase'])
        out = pdt._apply_phase1_engineering_fixes(df)
        assert 'hawkes_x_session_phase' not in out.columns
        assert 'tick_count_x_session_phase' not in out.columns

    def test_missing_hawkes_only_skips_hawkes(self):
        df = _interaction_input_frame()
        df = df.drop(columns=['hawkes_intrabar_sum'])
        out = pdt._apply_phase1_engineering_fixes(df)
        assert 'hawkes_x_session_phase' not in out.columns
        assert 'tick_count_x_session_phase' in out.columns

    def test_does_not_modify_source_cols(self):
        df = _interaction_input_frame()
        before_h = df['hawkes_intrabar_sum'].copy()
        before_t = df['tick_count'].copy()
        before_s = df['session_phase'].copy()
        out = pdt._apply_phase1_engineering_fixes(df)
        pd.testing.assert_series_equal(out['hawkes_intrabar_sum'], before_h)
        pd.testing.assert_series_equal(out['tick_count'], before_t)
        pd.testing.assert_series_equal(out['session_phase'], before_s)


# ── Schema validation: NaN-safe ────────────────────────────────────────────
class TestNanRobust:
    def test_nan_session_phase_treated_as_neg1(self):
        """The fillna(-1) policy — NaN session_phase produces a finite
        negative product, not NaN propagation that would crash downstream."""
        df = _interaction_input_frame(n=5)
        df['session_phase'] = df['session_phase'].astype(float)
        df.loc[2, 'session_phase'] = float('nan')
        out = pdt._apply_phase1_engineering_fixes(df)
        assert np.isfinite(out['hawkes_x_session_phase']).all()
        assert np.isfinite(out['tick_count_x_session_phase']).all()


# ── Integration with Phase 1.2 keep_top ───────────────────────────────────
class TestKeepTopPreservesInteractions:
    def test_keep_top_keeps_both_interaction_cols(self):
        n = 5
        df = pd.DataFrame({
            'ts_event': pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC"),
            'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0,
            'regime_label': "r", 'atr_14': 0.001,
            'is_session_break': np.zeros(n, dtype=np.int8),
            'obi': 0.0, 'obi_net': 0.0, 'cvd': 0.0, 'cvd_cumulative': 0.0,
            # Phase 1.5 interaction outputs
            'hawkes_x_session_phase': np.zeros(n, dtype=np.float32),
            'tick_count_x_session_phase': np.zeros(n, dtype=np.float32),
        })
        out = pdt._apply_phase1_feature_selection(df, mode='keep_top')
        assert 'hawkes_x_session_phase' in out.columns
        assert 'tick_count_x_session_phase' in out.columns


# ── Smoke: II.A with the rest of the engineering fixes (R4 + R5 combo) ────
class TestSmokeFullFixStack:
    def test_all_phase14_outputs_after_one_call(self):
        df = _interaction_input_frame(n=30)
        out = pdt._apply_phase1_engineering_fixes(df, warmup_drop_bars=5)
        # II.A
        assert 'hawkes_x_session_phase' in out.columns
        assert 'tick_count_x_session_phase' in out.columns
        # II.B
        assert 'dist_to_session_high_atr' in out.columns
        assert 'dist_to_vwap_atr' in out.columns
        assert 'dist_to_pdh_atr' in out.columns
        # II.C
        assert 'regime_label_grouped' in out.columns
        # II.D
        assert 'is_warmup' in out.columns
        assert int(out['is_warmup'].sum()) == 5
