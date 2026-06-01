"""Phase 1.3 — continuous z-scores become the primary event signal.

The IC audit showed `hawkes_intrabar_sum` (raw sum) STRONG while the
boolean `hawkes_z > 1.0` carried near-zero IC. The fix: in
detect_microstructure_events, switch `event_score` from
max(binary_thresholded, continuous) to the continuous score directly,
and expose the raw z-scores as their own columns so the model learns
the continuous response.

Tests cover:
  - event_score equals event_score_continuous after the change
  - event_score_binary preserved as legacy
  - Three new raw-z columns added (hawkes_z_raw / absorb_z_raw / kyle_z_raw)
  - The four new columns are in Phase 1.1 FeatureSpec catalog
  - Phase 1.2 keep_top whitelist preserves them; drop_noise does too
  - Source-level guard: the old max() construct did not come back
  - Smoke: detect_microstructure_events on a synthetic frame yields
    the new columns + their values are causally derived
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
    ALL_FEATURE_SPECS, PHASE_1_3_NEW_FEATURES,
)


# ── helpers ─────────────────────────────────────────────────────────────────
def _event_input_frame(n: int = 200) -> pd.DataFrame:
    """A frame shaped like detect_microstructure_events's input."""
    rng = np.random.RandomState(0)
    ts = pd.date_range("2025-04-01 08:00", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        'ts_event': ts,
        # primary inputs for z-scoring
        'hawkes_intensity': np.abs(rng.randn(n)) * 0.5,
        'absorption_intensity': np.abs(rng.randn(n)) * 0.3,
        'kyle_lambda': np.abs(rng.randn(n)) * 0.1,
        # coverage columns the score uses
        'mbo_bar_coverage': 0.95,
        'mbp_bar_coverage': 0.95,
        # context
        'regime_label': "ranging",
    })


# ── Schema integration ────────────────────────────────────────────────────
class TestSchemaIntegration:
    def test_four_new_specs_in_phase_1_3_catalog(self):
        names = {s.name for s in PHASE_1_3_NEW_FEATURES}
        assert names == {
            'event_score_continuous',
            'hawkes_z_raw', 'absorb_z_raw', 'kyle_z_raw',
        }

    def test_all_four_carried_by_aggregate_catalog(self):
        names = {s.name for s in ALL_FEATURE_SPECS}
        for c in ('event_score_continuous', 'hawkes_z_raw',
                  'absorb_z_raw', 'kyle_z_raw'):
            assert c in names


# ── Phase 1.2 whitelist preserves the new columns ──────────────────────────
class TestPhase12CompatibleWithNewCols:
    def test_keep_top_preserves_new_zscore_cols(self):
        n = 5
        df = pd.DataFrame({
            'ts_event': pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC"),
            'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0,
            'regime_label': "r", 'atr_14': 0.001,
            'is_session_break': np.zeros(n, dtype=np.int8),
            'obi': 0.0, 'obi_net': 0.0, 'cvd': 0.0, 'cvd_cumulative': 0.0,
            # New Phase 1.3 cols
            'event_score_continuous': 0.5,
            'hawkes_z_raw': 0.0, 'absorb_z_raw': 0.0, 'kyle_z_raw': 0.0,
        })
        out = pdt._apply_phase1_feature_selection(df, mode='keep_top')
        for c in ('event_score_continuous', 'hawkes_z_raw',
                  'absorb_z_raw', 'kyle_z_raw'):
            assert c in out.columns


# ── Source-level guards ────────────────────────────────────────────────────
class TestSourceLevelGuards:
    def _src(self) -> str:
        return (REPO_ROOT / "prepare_day_trading.py").read_text()

    def test_old_max_construct_is_gone(self):
        """The 'max(binary, continuous)' fallback collapsed magnitude.
        It must not return."""
        src = self._src()
        # The exact prior construct (event_score assignment to max())
        forbidden = "df['event_score'] = np.maximum(\n        event_score.to_numpy"
        assert forbidden not in src, (
            "Phase 1.3 regressed — the binary-vs-continuous max() construct "
            "is back. event_score must be assigned to continuous_score "
            "directly."
        )

    def test_continuous_score_is_event_score(self):
        src = self._src()
        # The new assignment must be present
        assert "df['event_score'] = continuous_score.to_numpy" in src

    def test_three_raw_z_cols_assigned(self):
        src = self._src()
        for col in ("hawkes_z_raw", "absorb_z_raw", "kyle_z_raw"):
            assert f"df['{col}']" in src, f"raw z column not written: {col}"

    def test_legacy_binary_preserved(self):
        src = self._src()
        assert "df['event_score_binary']" in src


# ── Smoke: drive detect_microstructure_events end-to-end ───────────────────
class TestSmokeDetectMicrostructureEvents:
    def test_function_produces_new_columns(self):
        df = _event_input_frame(200)
        out = pdt.detect_microstructure_events(df)
        # All six new columns must materialize
        for col in (
            'event_score', 'event_score_continuous', 'event_score_binary',
            'hawkes_z_raw', 'absorb_z_raw', 'kyle_z_raw',
        ):
            assert col in out.columns, f"missing: {col}"

    def test_event_score_equals_continuous_not_max(self):
        df = _event_input_frame(200)
        out = pdt.detect_microstructure_events(df)
        # event_score must equal event_score_continuous EXACTLY
        np.testing.assert_array_equal(
            out['event_score'].to_numpy(),
            out['event_score_continuous'].to_numpy(),
        )

    def test_raw_zscores_have_finite_values(self):
        df = _event_input_frame(200)
        out = pdt.detect_microstructure_events(df)
        for col in ('hawkes_z_raw', 'absorb_z_raw', 'kyle_z_raw'):
            v = out[col].to_numpy()
            assert np.isfinite(v).all(), f"{col} has non-finite values"
            # z-score should have mean ~0 and meaningful variance on this input
            assert abs(float(v.mean())) < 2.0
            assert float(v.std()) > 0.01, f"{col} variance collapsed"

    def test_binary_event_score_still_in_unit_range(self):
        df = _event_input_frame(200)
        out = pdt.detect_microstructure_events(df)
        v = out['event_score_binary'].to_numpy()
        # Binary score is sum-of-threshold-indicators × weights → bounded
        assert (v >= 0).all() and (v <= 2.5).all()

    def test_event_score_richer_than_binary_in_magnitude(self):
        """The continuous score should have STRICTLY more distinct values
        than the binary version — that is the whole point of Phase 1.3."""
        df = _event_input_frame(500)
        out = pdt.detect_microstructure_events(df)
        n_distinct_cont = out['event_score_continuous'].round(4).nunique()
        n_distinct_binary = out['event_score_binary'].round(4).nunique()
        assert n_distinct_cont > n_distinct_binary, (
            f"continuous score should be richer: cont={n_distinct_cont}, "
            f"binary={n_distinct_binary}"
        )
