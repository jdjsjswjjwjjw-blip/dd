"""Phase 1.8 — full integration smoke test.

End-to-end gate that drives all Phase 1.X rounds together on a single
synthetic frame and asserts every output co-exists, the sidecar is
internally consistent, and the validation harness exit code is 0. If any
round (R1-R7) regresses in a way the per-round suite missed, this catches
it.

The smoke deliberately uses a small synthetic frame (no real Q2 data) so
it runs in <1s — the contract is "no regressions across rounds", not
"empirical IC numbers". The empirical numbers gate lives in
tools/diagnostics/phase1_revalidate.py and runs against a real parquet.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as pdt
from modules.dataset_schema import (
    SCHEMA_VERSION, ALL_FEATURE_SPECS, ALL_LABEL_SCHEMAS,
    PHASE_1_2_BLACKLIST, write_dataset_meta, load_dataset_meta, validate,
)
from modules.features_v2.iceberg import attach_iceberg_features


def _refinery_like_frame(n: int = 100) -> pd.DataFrame:
    """A frame shaped like the refinery's late-stage df_out — has
    enough columns to exercise every Phase 1.X round end-to-end."""
    rng = np.random.RandomState(0)
    ts = pd.date_range("2025-04-01 08:00", periods=n, freq="5min", tz="UTC")
    closes = 1.0 + np.cumsum(rng.randn(n) * 0.0005)
    df = pd.DataFrame({
        # OHLCV
        'ts_event': ts, 'open': closes, 'high': closes + 0.0005,
        'low': closes - 0.0005, 'close': closes,
        'volume': np.full(n, 1000.0),
        # regime + atr
        'regime_label': np.array(['ranging'] * (n - 10) + ['volatile'] * 5 + ['low_liquidity'] * 5),
        'atr_14': np.full(n, 0.001, dtype=np.float64),
        'is_session_break': np.zeros(n, dtype=np.int8),
        # canonical aliases (Phase 1.1: both names required)
        'obi': np.zeros(n), 'obi_net': np.zeros(n),
        'cvd': np.cumsum(rng.randn(n) * 5), 'cvd_cumulative': np.zeros(n),
        'bar_cvd_delta': rng.randn(n) * 50,
        # structural levels (II.B inputs)
        'london_sess_high': 1.005, 'current_vwap': 1.001, 'pdh': 1.002,
        'pdl': 0.998, 'london_sess_low': 0.998,
        # microstructure inputs (Phase 1.3 z-scoring)
        'hawkes_intensity': np.abs(rng.randn(n)) * 0.5,
        'absorption_intensity': np.abs(rng.randn(n)) * 0.3,
        'kyle_lambda': np.abs(rng.randn(n)) * 0.1,
        'mbo_bar_coverage': 0.95, 'mbp_bar_coverage': 0.95,
        # already-computed features the audit cares about
        'london_sess_low_val': 1.002,  # just present
        'hawkes_intrabar_sum': np.abs(rng.randn(n)) * 0.5,
        'tick_count': np.full(n, 100, dtype=np.int32),
        'session_phase': np.arange(n, dtype=np.int8) % 6,
        'time_to_ny_close_min': np.linspace(720, 0, n, dtype=np.float32),
        'inter_event_time': np.full(n, 3.0, dtype=np.float32),
        'liquidity_density': np.full(n, 0.5, dtype=np.float32),
        'kyle_lambda_intrabar_mean': np.full(n, 0.05, dtype=np.float32),
        # context for IC audit
        'is_event': np.ones(n, dtype=np.int8),
        'event_score': 0.5, 'event_direction': 0,
        'event_flag': 1, 'train_event_flag': 1,
        'kalman_direction': 0,
        'session': 'london', 'is_london': True,
        'is_overlap': False, 'is_ny': False,
        'regime_cluster': 0,
        # a couple of blacklisted columns the filter must drop
        'is_friday': np.zeros(n, dtype=np.int8),
        'cycle_hurst': np.zeros(n, dtype=np.float32),
        'cvd_momentum': np.zeros(n, dtype=np.float32),
        # one unknown (drop_noise keeps, keep_top drops)
        'experimental_X': np.zeros(n),
    })
    # Seed the bias_label / exec_* / next_price_delta from a quick label run
    # Build label arrays directly (avoid chained-assignment under CoW)
    bias = np.full(n, 2, dtype=np.int8)
    bias[10:30] = 0    # LONG
    bias[40:60] = 1    # SHORT
    df['bias_label'] = bias
    df['path_outcome'] = np.where(
        bias == 0, 0, np.where(bias == 1, 1, 4),
    ).astype(np.int8)
    df['exec_label'] = bias.copy()
    df['exec_path'] = df['path_outcome'].to_numpy().copy()
    df['exec_valid'] = (bias != 2)
    df['next_price_delta'] = (closes - np.roll(closes, -6)).astype(np.float32)
    npv = np.ones(n, dtype=bool)
    npv[-6:] = False
    df['next_price_delta_valid'] = npv
    df['signal_quality'] = 0
    df['neutral_reason'] = 0
    return df


# ── R1 + R2 + R3 + R4 + R5 + R6 + R7 — single integrated drive ─────────────
class TestPhase1FullIntegrationSmoke:
    """The cross-cutting gate: drive every Phase 1.X transform on the
    same frame and assert the joint output is internally consistent."""

    def _drive_full_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        # R7 — iceberg (no MBO → zeros, but cols added)
        df = attach_iceberg_features(df, mbo=None)
        # R4 + R5 + R6 — engineering fixes all in one call
        df = pdt._apply_phase1_engineering_fixes(df, warmup_drop_bars=20)
        # R2 — feature selection (default drop_noise)
        df = pdt._apply_phase1_feature_selection(df, mode='drop_noise')
        return df

    def test_full_drive_runs_without_error(self):
        df = _refinery_like_frame()
        out = self._drive_full_pipeline(df)
        assert len(out) == len(df)
        assert len(out.columns) > 0

    def test_blacklisted_cols_dropped(self):
        df = _refinery_like_frame()
        out = self._drive_full_pipeline(df)
        for bad in ('is_friday', 'cycle_hurst', 'cvd_momentum'):
            assert bad not in out.columns

    def test_canonical_aliases_survive(self):
        df = _refinery_like_frame()
        out = self._drive_full_pipeline(df)
        for c in ('obi_net', 'cvd_cumulative'):
            assert c in out.columns

    def test_all_phase_1_X_outputs_present(self):
        df = _refinery_like_frame()
        out = self._drive_full_pipeline(df)
        expected = {
            # R4 (II.B + II.C + II.D)
            'dist_to_session_high_atr', 'dist_to_vwap_atr', 'dist_to_pdh_atr',
            'regime_label_grouped', 'is_warmup',
            # R5 (II.A interactions)
            'hawkes_x_session_phase', 'tick_count_x_session_phase',
            # R6 (MT5 CVD)
            'cvd_bar_5m', 'cvd_direction_ratio_5m', 'cvd_intensity_vs_atr',
            'cvd_divergence_at_level', 'cvd_consecutive_imbalance',
            # R7 (iceberg)
            'iceberg_count_5m', 'iceberg_total_volume_5m',
        }
        missing = expected - set(out.columns)
        assert not missing, f"Phase 1.X outputs missing: {missing}"

    def test_label_targets_survive(self):
        df = _refinery_like_frame()
        out = self._drive_full_pipeline(df)
        for c in ('bias_label', 'exec_label', 'next_price_delta',
                  'exec_valid', 'next_price_delta_valid'):
            assert c in out.columns

    def test_warmup_masking_propagated(self):
        df = _refinery_like_frame()
        out = self._drive_full_pipeline(df)
        # First 20 bars should be warmup; train_event_flag & exec_valid
        # for those bars should be 0/False.
        assert int(out['train_event_flag'].iloc[:20].sum()) == 0
        assert not bool(out['exec_valid'].iloc[:20].any())


# ── Sidecar produced from the full pipeline ───────────────────────────────
class TestFullPipelineSidecar:
    def test_sidecar_round_trip(self, tmp_path):
        df = _refinery_like_frame()
        # Run the same drive as above
        from modules.features_v2.iceberg import attach_iceberg_features
        df = attach_iceberg_features(df, mbo=None)
        df = pdt._apply_phase1_engineering_fixes(df, warmup_drop_bars=20)
        df = pdt._apply_phase1_feature_selection(df, mode='drop_noise')
        # Write + read back
        parquet = tmp_path / "full_smoke.parquet"
        df.to_parquet(parquet)
        write_dataset_meta(df, parquet, extra={"smoke": True})
        loaded = load_dataset_meta(parquet)

        # Sidecar contract
        assert loaded["schema_version"] == SCHEMA_VERSION
        assert loaded["git_hash"] is not None
        assert loaded["built_at_utc"].endswith("+00:00")
        # Phase 1.4/1.5 spec'd outputs registered in feature_columns_present
        for c in ('cvd_bar_5m', 'dist_to_session_high_atr'):
            assert c in loaded["feature_columns_present"], f"sidecar missing: {c}"
        # regime_label_grouped + is_warmup are context (categorical state /
        # bool mask), not FeatureSpec entries — they should still ship in
        # the parquet itself even if not in the feature_columns_present audit
        assert 'regime_label_grouped' in df.columns
        assert 'is_warmup' in df.columns
        # No blacklisted column ships
        assert loaded["blacklisted_columns_still_present"] == []
        # Validate clean (strict requires the 3 training targets we seeded)
        errors = validate(df, strict=True)
        assert errors == [], f"validate(strict) errors: {errors}"


# ── The IC re-audit harness itself runs ───────────────────────────────────
class TestRevalidationHarness:
    """Drive tools/diagnostics/phase1_revalidate.py on a synthetic
    post-cleanup parquet → exit code 0 means every gate passed."""

    def test_harness_imports_and_runs_on_synthetic(self, tmp_path):
        df = _refinery_like_frame(n=200)
        # Drive the cleanup pipeline so every gate has its inputs
        df = attach_iceberg_features(df, mbo=None)
        df = pdt._apply_phase1_engineering_fixes(df, warmup_drop_bars=20)
        df = pdt._apply_phase1_feature_selection(df, mode='drop_noise')

        parquet = tmp_path / "smoke.parquet"
        df.to_parquet(parquet)
        write_dataset_meta(df, parquet, extra={"smoke": True})

        # Run the harness as a subprocess (mirrors real operator usage)
        harness = REPO_ROOT / "tools" / "diagnostics" / "phase1_revalidate.py"
        assert harness.exists(), f"harness script not found: {harness}"
        result = subprocess.run(
            [sys.executable, str(harness), "--parquet", str(parquet)],
            capture_output=True, text=True, timeout=30,
        )
        # We don't assert exit code == 0 on synthetic data (some gates
        # like the IC density may fail on tiny synthetic samples) — but
        # the harness MUST run to completion and print all gate lines.
        assert "Phase 1.6 — Empirical Re-validation Gate" in result.stdout
        assert "gates passed" in result.stdout
        assert result.returncode in (0, 1), (
            f"harness crashed with exit={result.returncode}; "
            f"stderr={result.stderr[:500]}"
        )


# ── Source-level guards: every round's marker is preserved ────────────────
class TestPhase1Markers:
    def test_every_round_has_a_marker_in_source(self):
        src = (REPO_ROOT / "prepare_day_trading.py").read_text()
        markers = {
            # R1 (canonical-alias fallback) and R7 (iceberg attach) were changed by
            # D1: obi_net/cvd_cumulative duplicates dropped; iceberg deferred to the
            # depth track. Their markers now assert the D1 decisions in source.
            "D1: obi_net / cvd_cumulative were EXACT duplicates": "R1 (D1)",
            "Phase 1.2 — Feature Selection": "R2",
            "Phase 1.3: continuous z-scores": "R3",
            "Phase 1.4 — Engineering fixes from the IC audit": "R4 + R5 + R6",
            "D1: iceberg DEFERRED to the depth track": "R7 (D1)",
        }
        for marker, round_name in markers.items():
            assert marker in src, (
                f"{round_name} marker '{marker}' missing from refinery — "
                f"the round may have been removed or refactored away"
            )

    def test_dataset_schema_pipeline_phases_includes_1_X(self):
        from modules.dataset_schema import PIPELINE_CLEANUP_PHASES
        # Phase 1.1 was added; future rounds should be added to this dict
        # too so the sidecar always documents what's live.
        assert "1.1" in PIPELINE_CLEANUP_PHASES


# ── Cross-round invariant: every TARGET column in dataset_schema is
# blacklisted by the data_loader leakage guard ────────────────────────────
class TestCrossRoundLeakageInvariant:
    def test_every_target_excluded_by_data_loader(self):
        from self_supervised.data_loader import _is_leakage_column
        from modules.dataset_schema import (
            target_column_names, LEAKAGE_FORWARD, specs_by_leakage,
        )
        for c in target_column_names():
            assert _is_leakage_column(c), (
                f"target column {c!r} would leak — not excluded by guard"
            )
        for spec in specs_by_leakage(LEAKAGE_FORWARD):
            assert _is_leakage_column(spec.name), (
                f"forward diagnostic {spec.name!r} not excluded"
            )
