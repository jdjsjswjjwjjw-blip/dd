"""Phase 1.4 — three IC-audit-driven engineering fixes (II.B / II.C / II.D).

The audit identified three feature-engineering problems that the
schema/leakage cleanup couldn't reach:

  II.B — scale-dependent IR collapse on the 3 STRONG price-related
         features (london_sess_high / current_vwap / pdh). Fix: publish
         ATR-normalised distance versions next to them.

  II.C — regime sparsity. `volatile` had n=18 across all of Q2;
         `low_liquidity` had n=44 with a sign-flipped IC. Fix: a
         regime_label_grouped column that collapses both into 'rare',
         leaving regime_label untouched.

  II.D — WARMUP_RIDER signal on mbp_bar_coverage (IC drops 57.5% after
         the first ~20 bars). Fix: is_warmup mask + train_event_flag /
         exec_valid / next_price_delta_valid all masked to exclude the
         warmup bars from the training pool.

All three are additive — they don't overwrite any existing column.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as pdt


# ── helpers ─────────────────────────────────────────────────────────────────
def _engineering_input_frame(n: int = 50) -> pd.DataFrame:
    """Frame shaped like the refinery's late-stage df_out — has the cols
    the fix function reads + masks it modifies. Regime is built so that
    if n >= 8 the frame includes some volatile + low_liquidity rows the
    II.C grouping test can verify; smaller n falls back to ranging-only."""
    rng = np.random.RandomState(0)
    closes = 1.0 + np.cumsum(rng.randn(n) * 0.0005)
    if n >= 8:
        regime = np.array(
            ['ranging'] * (n - 8) + ['volatile'] * 4 + ['low_liquidity'] * 4
        )
    else:
        regime = np.array(['ranging'] * n)
    return pd.DataFrame({
        'ts_event': pd.date_range("2025-04-01 08:00", periods=n, freq="5min", tz="UTC"),
        'close': closes,
        'atr_14': np.full(n, 0.001, dtype=np.float64),
        'london_sess_high': np.full(n, 1.0050),
        'current_vwap': np.full(n, 1.0010),
        'pdh': np.full(n, 1.0020),
        'regime_label': regime,
        # masks the fix should update
        'train_event_flag': np.ones(n, dtype=np.int8),
        'exec_valid': np.ones(n, dtype=bool),
        'next_price_delta_valid': np.ones(n, dtype=bool),
    })


# ── II.B: dist_to_X_atr ─────────────────────────────────────────────────────
class TestII_B_DistToXAtr:
    def test_all_three_dist_cols_added(self):
        out = pdt._apply_phase1_engineering_fixes(_engineering_input_frame())
        for c in ('dist_to_session_high_atr', 'dist_to_vwap_atr', 'dist_to_pdh_atr'):
            assert c in out.columns

    def test_dist_to_session_high_atr_formula(self):
        df = _engineering_input_frame(n=10)
        out = pdt._apply_phase1_engineering_fixes(df)
        expected = (df['close'].to_numpy() - df['london_sess_high'].to_numpy()) / df['atr_14'].to_numpy()
        np.testing.assert_allclose(
            out['dist_to_session_high_atr'].to_numpy(),
            expected.astype(np.float32),
            rtol=1e-5,
        )

    def test_dist_to_vwap_atr_formula(self):
        df = _engineering_input_frame(n=10)
        out = pdt._apply_phase1_engineering_fixes(df)
        expected = (df['close'].to_numpy() - df['current_vwap'].to_numpy()) / df['atr_14'].to_numpy()
        np.testing.assert_allclose(
            out['dist_to_vwap_atr'].to_numpy(),
            expected.astype(np.float32),
            rtol=1e-5,
        )

    def test_dist_robust_to_zero_atr(self):
        df = _engineering_input_frame(n=5)
        df.loc[2, 'atr_14'] = 0.0       # singular row
        out = pdt._apply_phase1_engineering_fixes(df)
        # The singular row must be NaN — never inf/Series-of-1e12
        assert pd.isna(out['dist_to_vwap_atr'].iloc[2])
        # Other rows must still be finite
        assert np.isfinite(out['dist_to_vwap_atr'].iloc[0])

    def test_missing_source_col_skipped_gracefully(self):
        df = _engineering_input_frame()
        df = df.drop(columns=['pdh'])
        out = pdt._apply_phase1_engineering_fixes(df)
        # The other two MUST be there; pdh dist absent — not an error
        assert 'dist_to_session_high_atr' in out.columns
        assert 'dist_to_vwap_atr' in out.columns
        assert 'dist_to_pdh_atr' not in out.columns

    def test_does_not_modify_source_cols(self):
        df = _engineering_input_frame()
        before_lsh = df['london_sess_high'].copy()
        before_vwap = df['current_vwap'].copy()
        before_pdh = df['pdh'].copy()
        out = pdt._apply_phase1_engineering_fixes(df)
        pd.testing.assert_series_equal(out['london_sess_high'], before_lsh)
        pd.testing.assert_series_equal(out['current_vwap'], before_vwap)
        pd.testing.assert_series_equal(out['pdh'], before_pdh)


# ── II.C: regime_label_grouped ─────────────────────────────────────────────
class TestII_C_RegimeGrouping:
    def test_volatile_collapsed_to_rare(self):
        out = pdt._apply_phase1_engineering_fixes(_engineering_input_frame())
        # In the seeded frame, volatile bars exist; they must become 'rare'
        grouped = out['regime_label_grouped']
        original = out['regime_label']
        # rows that were volatile/low_liquidity in original are now 'rare'
        for idx in range(len(out)):
            if original.iloc[idx] in ('volatile', 'low_liquidity'):
                assert grouped.iloc[idx] == 'rare', (
                    f"row {idx} original={original.iloc[idx]} but grouped={grouped.iloc[idx]}"
                )

    def test_other_regimes_passthrough(self):
        out = pdt._apply_phase1_engineering_fixes(_engineering_input_frame())
        grouped = out['regime_label_grouped']
        original = out['regime_label']
        for idx in range(len(out)):
            if original.iloc[idx] not in ('volatile', 'low_liquidity'):
                assert grouped.iloc[idx] == original.iloc[idx]

    def test_original_regime_label_unchanged(self):
        df = _engineering_input_frame()
        before = df['regime_label'].copy()
        out = pdt._apply_phase1_engineering_fixes(df)
        pd.testing.assert_series_equal(out['regime_label'], before)


# ── II.D: is_warmup mask + downstream masks ────────────────────────────────
class TestII_D_WarmupMask:
    def test_is_warmup_column_added(self):
        out = pdt._apply_phase1_engineering_fixes(_engineering_input_frame())
        assert 'is_warmup' in out.columns
        assert out['is_warmup'].dtype == bool

    def test_first_n_bars_marked(self):
        n = 50
        df = _engineering_input_frame(n=n)
        out = pdt._apply_phase1_engineering_fixes(df, warmup_drop_bars=20)
        assert int(out['is_warmup'].iloc[:20].sum()) == 20
        assert int(out['is_warmup'].iloc[20:].sum()) == 0

    def test_train_event_flag_masked(self):
        df = _engineering_input_frame(n=30)
        # All originally 1
        assert int(df['train_event_flag'].sum()) == 30
        out = pdt._apply_phase1_engineering_fixes(df, warmup_drop_bars=10)
        # First 10 must be 0; the rest must still be 1
        assert int(out['train_event_flag'].iloc[:10].sum()) == 0
        assert int(out['train_event_flag'].iloc[10:].sum()) == 20

    def test_exec_valid_masked(self):
        df = _engineering_input_frame(n=30)
        out = pdt._apply_phase1_engineering_fixes(df, warmup_drop_bars=10)
        assert not out['exec_valid'].iloc[:10].any()
        assert out['exec_valid'].iloc[10:].all()

    def test_next_price_delta_valid_masked(self):
        df = _engineering_input_frame(n=30)
        out = pdt._apply_phase1_engineering_fixes(df, warmup_drop_bars=10)
        assert not out['next_price_delta_valid'].iloc[:10].any()
        assert out['next_price_delta_valid'].iloc[10:].all()

    def test_zero_disables_masking(self):
        df = _engineering_input_frame(n=30)
        out = pdt._apply_phase1_engineering_fixes(df, warmup_drop_bars=0)
        assert not out['is_warmup'].any()
        # Nothing should have been masked
        assert out['train_event_flag'].equals(df['train_event_flag'].astype(np.int8))

    def test_warmup_bigger_than_frame_safe(self):
        df = _engineering_input_frame(n=5)
        out = pdt._apply_phase1_engineering_fixes(df, warmup_drop_bars=100)
        # Whole frame should be warmup but no crash
        assert out['is_warmup'].all()


# ── Integration: refinery signature + CLI + leakage guard ──────────────────
class TestIntegration:
    def test_refinery_signature_carries_warmup_drop_bars(self):
        import inspect
        sig = inspect.signature(pdt.run_day_trading_refinery)
        assert 'warmup_drop_bars' in sig.parameters
        assert sig.parameters['warmup_drop_bars'].default == pdt.DEFAULT_WARMUP_DROP_BARS

    def test_cli_flag_present(self):
        src = (REPO_ROOT / "prepare_day_trading.py").read_text()
        assert "'--warmup-drop-bars'" in src

    def test_engineering_fixes_called_before_filter(self):
        """The fix function must run BEFORE the Phase 1.2 feature selection,
        otherwise the new dist_to_X_atr cols (NEW spec) would be dropped
        by drop_noise's strict whitelist."""
        src = (REPO_ROOT / "prepare_day_trading.py").read_text()
        idx_fix = src.find("_apply_phase1_engineering_fixes(\n        df_out,")
        idx_filter = src.find("_apply_phase1_feature_selection(df_out, mode=feature_selection)")
        assert idx_fix > 0 and idx_filter > 0
        assert idx_fix < idx_filter, (
            "engineering fixes must run BEFORE feature selection"
        )

    def test_regime_label_grouped_excluded_from_features(self):
        """Same kind as regime_label — must be in the leakage guard."""
        from self_supervised.data_loader import _is_leakage_column
        assert _is_leakage_column("regime_label_grouped")

    def test_keep_top_preserves_new_engineering_cols(self):
        n = 10
        # Build a frame the late refinery would have at this point
        df = pd.DataFrame({
            'ts_event': pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC"),
            'open': 1.0, 'high': 1.001, 'low': 0.999, 'close': 1.0, 'volume': 1.0,
            'regime_label': "ranging", 'atr_14': 0.001,
            'is_session_break': np.zeros(n, dtype=np.int8),
            'obi': 0.0, 'obi_net': 0.0, 'cvd': 0.0, 'cvd_cumulative': 0.0,
            # Phase 1.4 outputs the keep_top whitelist must preserve
            'dist_to_session_high_atr': 0.0,
            'dist_to_vwap_atr': 0.0,
            'dist_to_pdh_atr': 0.0,
            'regime_label_grouped': "ranging",
            'is_warmup': np.zeros(n, dtype=bool),
        })
        out = pdt._apply_phase1_feature_selection(df, mode='keep_top')
        for c in ('dist_to_session_high_atr', 'dist_to_vwap_atr',
                  'dist_to_pdh_atr', 'regime_label_grouped', 'is_warmup'):
            assert c in out.columns


# ── Smoke: drive a small end-to-end flow ──────────────────────────────────
class TestSmokeEndToEnd:
    def test_apply_then_filter_then_sidecar(self, tmp_path):
        from modules.dataset_schema import write_dataset_meta
        df = _engineering_input_frame(n=40)
        # Add the canonical aliases the validator demands
        df['obi'] = 0.0; df['obi_net'] = 0.0
        df['cvd'] = 0.0; df['cvd_cumulative'] = 0.0
        df['open'] = 1.0; df['high'] = 1.001; df['low'] = 0.999; df['volume'] = 1.0
        df['is_session_break'] = np.zeros(len(df), dtype=np.int8)

        df = pdt._apply_phase1_engineering_fixes(df, warmup_drop_bars=10)
        df = pdt._apply_phase1_feature_selection(df, mode='drop_noise')

        parquet = tmp_path / "smoke.parquet"
        df.to_parquet(parquet)
        meta_path = write_dataset_meta(df, parquet)

        # All three Phase 1.4 output classes survived through the pipeline
        for c in ('dist_to_session_high_atr', 'dist_to_vwap_atr',
                  'dist_to_pdh_atr', 'regime_label_grouped', 'is_warmup'):
            assert c in df.columns

        # warmup mask is reflected in train_event_flag
        assert int(df['train_event_flag'].iloc[:10].sum()) == 0
        assert int(df['train_event_flag'].iloc[10:].sum()) == 30
