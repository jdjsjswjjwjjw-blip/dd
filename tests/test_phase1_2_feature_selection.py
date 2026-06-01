"""Phase 1.2 — _apply_phase1_feature_selection.

Three modes (keep_top / drop_noise / all), each driven by the
modules/dataset_schema.py source-of-truth (PHASE_1_2_BLACKLIST +
ALL_FEATURE_SPECS + REQUIRED_*). Tests cover:

  - Each mode's behaviour on a frame seeded with one of every column class
  - Required columns are NEVER dropped, regardless of mode
  - Labels (bias_label + exec_* + next_price_delta) are NEVER dropped
  - The blacklist is applied as a SET (no partial-string matches that
    might catch a similarly-named legitimate column)
  - CLI default is 'drop_noise' (the safe-by-default behaviour)
  - Smoke: end-to-end on a synthetic frame, sidecar reflects the drop
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as pdt
from modules.dataset_schema import (
    PHASE_1_2_BLACKLIST, REQUIRED_OHLCV, REQUIRED_REGIME,
    ALL_FEATURE_SPECS, ALL_LABEL_SCHEMAS, write_dataset_meta,
)


# ── helpers ─────────────────────────────────────────────────────────────────
def _seeded_frame() -> pd.DataFrame:
    """Build a frame that contains:
       - all REQUIRED_OHLCV / REQUIRED_REGIME columns
       - canonical aliases (obi/obi_net, cvd/cvd_cumulative)
       - 3 audited STRONG features
       - 3 NEW (Phase 1.4) features
       - 3 blacklisted columns
       - 2 unknown columns (not in whitelist, not in blacklist)
       - all label-target columns
    """
    n = 5
    return pd.DataFrame({
        # OHLCV (required)
        'ts_event': pd.date_range("2025-04-01 08:00", periods=n, freq="5min", tz="UTC"),
        'open': 1.0, 'high': 1.001, 'low': 0.999, 'close': 1.0, 'volume': 1000.0,
        # regime (required)
        'regime_label': "ranging", 'atr_14': 0.001,
        'is_session_break': np.zeros(n, dtype=np.int8),
        # canonical aliases + raw
        'obi': np.zeros(n), 'obi_net': np.zeros(n),
        'cvd': np.zeros(n), 'cvd_cumulative': np.zeros(n),
        # 3 audited STRONG features
        'london_sess_high': 1.005,
        'current_vwap': 1.000,
        'tick_count': np.full(n, 50, dtype=np.int32),
        # 3 NEW (Phase 1.4) features
        'cvd_bar_5m': np.zeros(n, dtype=np.float32),
        'cvd_direction_ratio_5m': np.zeros(n, dtype=np.float32),
        'iceberg_count_5m': np.zeros(n, dtype=np.int32),
        # 3 blacklisted (must be dropped in drop_noise + keep_top)
        'is_friday': np.zeros(n, dtype=np.int8),
        'cycle_hurst': np.zeros(n, dtype=np.float32),
        'cvd_momentum': np.zeros(n, dtype=np.float32),
        # 2 unknown (whitelist drops, blacklist keeps)
        'some_experimental_feature': np.zeros(n),
        'another_unknown': np.zeros(n),
        # All label targets
        'bias_label': np.full(n, 2, dtype=np.int8),
        'exec_label': np.full(n, 2, dtype=np.int8),
        'exec_path': np.zeros(n, dtype=np.int8),
        'exec_valid': np.zeros(n, dtype=bool),
        'next_price_delta': np.zeros(n, dtype=np.float32),
        'next_price_delta_valid': np.zeros(n, dtype=bool),
    })


# ── Mode: 'all' ────────────────────────────────────────────────────────────
class TestModeAll:
    def test_all_keeps_every_column(self):
        df = _seeded_frame()
        out = pdt._apply_phase1_feature_selection(df, mode='all')
        assert list(out.columns) == list(df.columns)

    def test_all_keeps_unknowns(self):
        df = _seeded_frame()
        out = pdt._apply_phase1_feature_selection(df, mode='all')
        assert 'some_experimental_feature' in out.columns
        assert 'is_friday' in out.columns


# ── Mode: 'drop_noise' (the default) ───────────────────────────────────────
class TestModeDropNoise:
    def test_drops_every_blacklisted_column(self):
        df = _seeded_frame()
        out = pdt._apply_phase1_feature_selection(df, mode='drop_noise')
        for bad in ('is_friday', 'cycle_hurst', 'cvd_momentum'):
            assert bad not in out.columns

    def test_keeps_unknown_columns(self):
        """drop_noise is a strict-blacklist mode — unknowns survive."""
        df = _seeded_frame()
        out = pdt._apply_phase1_feature_selection(df, mode='drop_noise')
        assert 'some_experimental_feature' in out.columns
        assert 'another_unknown' in out.columns

    def test_keeps_all_required_columns(self):
        df = _seeded_frame()
        out = pdt._apply_phase1_feature_selection(df, mode='drop_noise')
        for c in (*REQUIRED_OHLCV, *REQUIRED_REGIME):
            assert c in out.columns

    def test_keeps_all_label_targets(self):
        df = _seeded_frame()
        out = pdt._apply_phase1_feature_selection(df, mode='drop_noise')
        for spec in ALL_LABEL_SCHEMAS:
            if spec.name in df.columns:
                assert spec.name in out.columns

    def test_blacklist_uses_set_match_not_substring(self):
        """If a column NAME merely contains a blacklisted substring, the
        match must NOT trigger — only exact-name membership in the set."""
        df = _seeded_frame()
        # `is_friday` is blacklisted; `is_friday_signal` (longer) is not
        df['is_friday_signal'] = np.zeros(len(df))
        out = pdt._apply_phase1_feature_selection(df, mode='drop_noise')
        assert 'is_friday' not in out.columns         # exact-match dropped
        assert 'is_friday_signal' in out.columns      # different name kept


# ── Mode: 'keep_top' (strict whitelist) ────────────────────────────────────
class TestModeKeepTop:
    def test_drops_all_blacklisted(self):
        df = _seeded_frame()
        out = pdt._apply_phase1_feature_selection(df, mode='keep_top')
        for bad in PHASE_1_2_BLACKLIST.keys():
            assert bad not in out.columns

    def test_drops_unknown_columns(self):
        df = _seeded_frame()
        out = pdt._apply_phase1_feature_selection(df, mode='keep_top')
        assert 'some_experimental_feature' not in out.columns
        assert 'another_unknown' not in out.columns

    def test_keeps_all_required_columns(self):
        df = _seeded_frame()
        out = pdt._apply_phase1_feature_selection(df, mode='keep_top')
        for c in (*REQUIRED_OHLCV, *REQUIRED_REGIME):
            assert c in out.columns

    def test_keeps_canonical_aliases(self):
        """obi_net + cvd_cumulative are the data_loader contract."""
        df = _seeded_frame()
        out = pdt._apply_phase1_feature_selection(df, mode='keep_top')
        assert 'obi_net' in out.columns
        assert 'cvd_cumulative' in out.columns

    def test_keeps_all_audited_features(self):
        df = _seeded_frame()
        out = pdt._apply_phase1_feature_selection(df, mode='keep_top')
        # Of the 3 audited features we seeded, all must survive
        for c in ('london_sess_high', 'current_vwap', 'tick_count'):
            assert c in out.columns

    def test_keeps_new_phase_1_4_features(self):
        df = _seeded_frame()
        out = pdt._apply_phase1_feature_selection(df, mode='keep_top')
        for c in ('cvd_bar_5m', 'cvd_direction_ratio_5m', 'iceberg_count_5m'):
            assert c in out.columns

    def test_keeps_all_label_targets(self):
        df = _seeded_frame()
        out = pdt._apply_phase1_feature_selection(df, mode='keep_top')
        for spec in ALL_LABEL_SCHEMAS:
            if spec.name in df.columns:
                assert spec.name in out.columns

    def test_keeps_context_for_ic_audit(self):
        """The IC re-audit splits per-session / per-regime — those cols
        must survive keep_top even though they aren't model features."""
        df = _seeded_frame()
        df['is_event'] = 1
        df['event_score'] = 0.5
        df['session'] = 'london'
        df['regime_cluster'] = 0
        out = pdt._apply_phase1_feature_selection(df, mode='keep_top')
        for c in ('is_event', 'event_score', 'session', 'regime_cluster'):
            assert c in out.columns


# ── Validation ─────────────────────────────────────────────────────────────
class TestModeValidation:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match='feature_selection'):
            pdt._apply_phase1_feature_selection(_seeded_frame(), mode='nonsense')

    def test_default_mode_is_drop_noise(self):
        import inspect
        sig = inspect.signature(pdt._apply_phase1_feature_selection)
        assert sig.parameters['mode'].default == 'drop_noise'

    def test_refinery_signature_carries_param(self):
        import inspect
        sig = inspect.signature(pdt.run_day_trading_refinery)
        assert 'feature_selection' in sig.parameters
        assert sig.parameters['feature_selection'].default == 'drop_noise'


# ── Smoke: end-to-end on a synthetic frame ─────────────────────────────────
class TestSmokeEndToEnd:
    def test_filter_then_sidecar_drop_noise(self, tmp_path):
        df = _seeded_frame()
        n_in = len(df.columns)
        out = pdt._apply_phase1_feature_selection(df, mode='drop_noise')
        assert len(out.columns) < n_in    # something WAS dropped
        # Sidecar reflects the trimmed frame
        parquet = tmp_path / "out.parquet"
        out.to_parquet(parquet)
        meta_path = write_dataset_meta(out, parquet)
        meta = json.loads(meta_path.read_text())
        assert meta["rows"] == len(out)
        # blacklisted_columns_still_present must be empty after the filter
        assert meta["blacklisted_columns_still_present"] == []

    def test_filter_then_sidecar_keep_top(self, tmp_path):
        df = _seeded_frame()
        out = pdt._apply_phase1_feature_selection(df, mode='keep_top')
        parquet = tmp_path / "out.parquet"
        out.to_parquet(parquet)
        meta_path = write_dataset_meta(out, parquet)
        meta = json.loads(meta_path.read_text())
        assert meta["blacklisted_columns_still_present"] == []
        # 3 audited features were seeded; 3 must show up
        for c in ('london_sess_high', 'current_vwap', 'tick_count'):
            assert c in meta["feature_columns_present"]


# ── Source-level guard ─────────────────────────────────────────────────────
class TestPhase12SourceGuards:
    def test_refinery_calls_filter_before_to_parquet(self):
        src = (REPO_ROOT / "prepare_day_trading.py").read_text()
        idx_filter = src.find("_apply_phase1_feature_selection(df_out, mode=feature_selection)")
        idx_write = src.find("df_out.to_parquet(out_path, index=False)")
        assert idx_filter > 0 and idx_write > 0
        assert idx_filter < idx_write, (
            "Phase 1.2 filter must run BEFORE the parquet write — "
            "otherwise blacklisted columns ship to consumers."
        )

    def test_cli_flag_present(self):
        src = (REPO_ROOT / "prepare_day_trading.py").read_text()
        assert "'--feature-selection'" in src
        assert "default='drop_noise'" in src
