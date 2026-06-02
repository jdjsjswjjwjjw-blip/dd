"""Regression for the ORIGINAL_FEATURES whitelist drop bug.

Discovered after running the IC audit on the user's Q2 6B post-cleanup
parquet: 9 of the Phase 1.3/1.4 engineered columns that the audit
expected to find were absent from the output. Root cause was
`finalize_daytrade_parquet_export` restricting the output to columns
in the `ORIGINAL_FEATURES` whitelist — columns the Phase 1.3/1.4 wave
added were not in that list, so they vanished silently. The downstream
Phase 1.4 engineering fix `_apply_phase1_engineering_fixes` then failed
its `if 'bar_cvd_delta' in out.columns` guard, killing 5 more CVD
features.

This file pins down the fix at the source level so the regression
can't sneak back in via a refactor that forgets to update the
whitelist.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as pdt


# Columns that MUST be in ORIGINAL_FEATURES for the parquet to ship the
# Phase 1.3/1.4 engineered signal. If finalize() drops any of these, the
# downstream Phase 1.4 engineering fix silently degrades.
PHASE_1_3_1_4_WHITELIST = frozenset({
    # Phase 1.3 — continuous z-scores (absorb_z_raw is DEPTH → D1-excluded below)
    'event_score_continuous',
    'event_score_binary',
    'hawkes_z_raw',
    'kyle_z_raw',
    # Phase 1.4 — MT5 CVD (source + 5 derived)
    'bar_cvd_delta',
    'cvd_bar_5m',
    'cvd_direction_ratio_5m',
    'cvd_intensity_vs_atr',
    'cvd_divergence_at_level',
    'cvd_consecutive_imbalance',
    # Phase 1.4 post-finalize cols (defensive)
    'dist_to_session_high_atr',
    'dist_to_vwap_atr',
    'dist_to_pdh_atr',
    'regime_label_grouped',
    'is_warmup',
    'hawkes_x_session_phase',
    'tick_count_x_session_phase',
    # NOTE (D1): absorb_z_raw, obi_net, cvd_cumulative, iceberg_count_5m,
    # iceberg_total_volume_5m were here pre-D1. They are DEPTH/duplicate and are
    # now intentionally EXCLUDED from export (_D1_DEPTH_EXCLUDED_FROM_EXPORT).
})


class TestOriginalFeaturesWhitelist:
    def test_phase_1_3_z_scores_in_whitelist(self):
        wl = set(pdt.ORIGINAL_FEATURES)
        for col in (
            'event_score_continuous',
            'event_score_binary',
            'hawkes_z_raw',
            'kyle_z_raw',
        ):
            assert col in wl, (
                f"Phase 1.3 column {col!r} missing from ORIGINAL_FEATURES — "
                f"finalize_daytrade_parquet_export will silently drop it."
            )
        # D1: absorb_z_raw is the absorption z-score (depth) → intentionally excluded
        assert 'absorb_z_raw' not in wl, "absorb_z_raw is depth → must be D1-excluded"

    def test_phase_1_4_cvd_in_whitelist(self):
        wl = set(pdt.ORIGINAL_FEATURES)
        # bar_cvd_delta is THE INPUT — without it, the 5 derived
        # CVD features cannot be computed.
        assert 'bar_cvd_delta' in wl, (
            "bar_cvd_delta missing from ORIGINAL_FEATURES — finalize() "
            "drops it, then _apply_phase1_engineering_fixes silently "
            "skips the 5 CVD features (the user's Q2 6B run hit this)."
        )
        for col in (
            'cvd_bar_5m',
            'cvd_direction_ratio_5m',
            'cvd_intensity_vs_atr',
            'cvd_divergence_at_level',
            'cvd_consecutive_imbalance',
        ):
            assert col in wl, (
                f"Phase 1.4 CVD column {col!r} missing from ORIGINAL_FEATURES"
            )

    def test_canonical_aliases_excluded_by_d1(self):
        """D1: obi_net / cvd_cumulative were EXACT duplicates (= obi / cvd). They
        are now EXCLUDED from export — obi itself is depth (not exported); cvd
        keeps its canonical name only. Assert the duplicates AND obi are gone."""
        wl = set(pdt.ORIGINAL_FEATURES)
        assert 'obi_net' not in wl
        assert 'cvd_cumulative' not in wl
        assert 'obi' not in wl          # obi is depth → D1-excluded
        assert 'cvd' in wl              # cvd (canonical flow) is KEPT

    def test_all_phase_1_x_engineered_cols_present(self):
        """Single shot — every column the Phase 1.X wave produces must
        be in the whitelist OR have a documented reason to be excluded."""
        wl = set(pdt.ORIGINAL_FEATURES)
        missing = PHASE_1_3_1_4_WHITELIST - wl
        assert not missing, (
            f"{len(missing)} Phase 1.X column(s) absent from "
            f"ORIGINAL_FEATURES — they will be dropped by finalize_-"
            f"daytrade_parquet_export. Missing: {sorted(missing)}"
        )


class TestFinalizeBehaviour:
    """Drive finalize_daytrade_parquet_export on a synthetic frame and
    confirm Phase 1.3/1.4 columns survive."""

    def test_phase_1_3_cols_survive_finalize(self):
        import numpy as np
        import pandas as pd

        n = 5
        df = pd.DataFrame({
            'ts_event': pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC"),
            'close': 1.0, 'open': 1.0, 'high': 1.0, 'low': 1.0,
            'event_score_continuous': np.zeros(n, dtype=np.float32),
            'hawkes_z_raw': np.zeros(n, dtype=np.float32),
            'absorb_z_raw': np.zeros(n, dtype=np.float32),
            'kyle_z_raw': np.zeros(n, dtype=np.float32),
            'bar_cvd_delta': np.zeros(n, dtype=np.float32),
        })
        out = pdt.finalize_daytrade_parquet_export(df)
        # Kept Phase 1.3/1.4 inputs must survive
        for col in (
            'event_score_continuous', 'hawkes_z_raw', 'kyle_z_raw', 'bar_cvd_delta',
        ):
            assert col in out.columns, (
                f"{col!r} dropped by finalize() — whitelist regression"
            )
        # D1: absorb_z_raw is depth → must be DROPPED by finalize
        assert 'absorb_z_raw' not in out.columns, (
            "absorb_z_raw should be dropped by finalize() (D1 depth exclusion)"
        )
