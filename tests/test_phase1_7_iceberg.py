"""Phase 1.5 — Iceberg detector (modules/features_v2/iceberg.py).

The detector follows the Korajczyk-Murphy rule strict-style:
    executed_volume_at_level >= 2.5 × max_displayed_size
    AND  at least one refill within 8 s of a fill
    AND  NOT a full clearout (some displayed size remained)

These tests construct synthetic MBO scenarios that exercise each rule
branch + the per-bar aggregation + the refinery integration's
zero-fallback when MBO is not supplied.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.features_v2.iceberg import (
    IcebergDetector, IcebergConfig, attach_iceberg_features,
)
from modules.dataset_schema import (
    PHASE_1_5_NEW_FEATURES, ALL_FEATURE_SPECS,
    FAMILY_ICEBERG, IC_VERDICT_NEW,
)


# ── helpers ─────────────────────────────────────────────────────────────────
def _mbo_iceberg_scenario() -> pd.DataFrame:
    """Synthetic MBO sequence at price 1.05 (bid side) that triggers the
    iceberg rule:
        - Displayed size oscillates: 100 → 0 → 100 → 0 → 100 → 50
        - Each "0" is preceded by a FILL action (size=100), each "100" by
          a REFILL action.
        - Final state: 50 (NOT a clearout).
        - Total executed = 100+100+100 = 300; 300 / max_displayed=100 = 3.0 ≥ 2.5 ✓
        - Each REFILL is 1 second after the FILL ≤ 8s ✓
    """
    base_ts = pd.Timestamp("2025-04-01 08:00:00", tz="UTC")
    rows = [
        # ADD (initial display)
        dict(ts_event=base_ts,                   price=1.05, side='B',
             displayed_size=100.0, executed_size=0.0,  action='ADD'),
        # FILL #1 (100 hit, display drops to 0)
        dict(ts_event=base_ts + pd.Timedelta(seconds=2), price=1.05, side='B',
             displayed_size=0.0,   executed_size=100.0, action='FILL'),
        # REFILL (1s later, display restored to 100)
        dict(ts_event=base_ts + pd.Timedelta(seconds=3), price=1.05, side='B',
             displayed_size=100.0, executed_size=0.0,  action='REFILL'),
        # FILL #2
        dict(ts_event=base_ts + pd.Timedelta(seconds=5), price=1.05, side='B',
             displayed_size=0.0,   executed_size=100.0, action='FILL'),
        # REFILL #2
        dict(ts_event=base_ts + pd.Timedelta(seconds=6), price=1.05, side='B',
             displayed_size=100.0, executed_size=0.0,  action='REFILL'),
        # FILL #3
        dict(ts_event=base_ts + pd.Timedelta(seconds=8), price=1.05, side='B',
             displayed_size=50.0,  executed_size=100.0, action='FILL'),
        # Final partial display remains
        dict(ts_event=base_ts + pd.Timedelta(seconds=9), price=1.05, side='B',
             displayed_size=50.0,  executed_size=0.0,  action='ADD'),
    ]
    return pd.DataFrame(rows)


def _mbo_full_clearout_scenario() -> pd.DataFrame:
    """Same big executed volume but display ENDS at 0 — not an iceberg."""
    df = _mbo_iceberg_scenario()
    df.loc[df.index[-1], 'displayed_size'] = 0.0
    return df


def _mbo_no_refill_scenario() -> pd.DataFrame:
    """Big fill but no REFILL action — not an iceberg."""
    df = _mbo_iceberg_scenario()
    df = df[df['action'] != 'REFILL'].reset_index(drop=True)
    return df


def _mbo_slow_refill_scenario() -> pd.DataFrame:
    """Refill happens but > 8s after the fill — not an iceberg."""
    base_ts = pd.Timestamp("2025-04-01 08:00:00", tz="UTC")
    return pd.DataFrame([
        dict(ts_event=base_ts,                  price=1.05, side='B',
             displayed_size=100.0, executed_size=0.0,  action='ADD'),
        dict(ts_event=base_ts + pd.Timedelta(seconds=2),  price=1.05, side='B',
             displayed_size=0.0,   executed_size=100.0, action='FILL'),
        # REFILL 30s later (> 8s threshold) — too slow
        dict(ts_event=base_ts + pd.Timedelta(seconds=32), price=1.05, side='B',
             displayed_size=100.0, executed_size=0.0,  action='REFILL'),
        dict(ts_event=base_ts + pd.Timedelta(seconds=34), price=1.05, side='B',
             displayed_size=0.0,   executed_size=100.0, action='FILL'),
        dict(ts_event=base_ts + pd.Timedelta(seconds=64), price=1.05, side='B',
             displayed_size=50.0,  executed_size=100.0, action='FILL'),
        dict(ts_event=base_ts + pd.Timedelta(seconds=65), price=1.05, side='B',
             displayed_size=50.0,  executed_size=0.0,  action='ADD'),
    ])


# ── Schema integration ────────────────────────────────────────────────────
class TestSchemaIntegration:
    def test_two_specs_present(self):
        names = {s.name for s in PHASE_1_5_NEW_FEATURES}
        assert names == {'iceberg_count_5m', 'iceberg_total_volume_5m'}

    def test_specs_tagged_iceberg_and_new(self):
        for s in PHASE_1_5_NEW_FEATURES:
            assert s.family == FAMILY_ICEBERG
            assert s.ic_verdict == IC_VERDICT_NEW


# ── Positive detection ────────────────────────────────────────────────────
class TestPositiveDetection:
    def test_iceberg_scenario_detected(self):
        det = IcebergDetector()
        events = det.detect_events(_mbo_iceberg_scenario())
        assert len(events) == 1

    def test_hidden_volume_value(self):
        det = IcebergDetector()
        events = det.detect_events(_mbo_iceberg_scenario())
        # 3 fills × 100 = 300
        assert events['hidden_volume'].iloc[0] == 300.0

    def test_event_carries_price_and_ts(self):
        det = IcebergDetector()
        events = det.detect_events(_mbo_iceberg_scenario())
        assert events['price_level'].iloc[0] == 1.05
        # ts_event is the last tick at the level
        assert events['ts_event'].iloc[0] == pd.Timestamp("2025-04-01 08:00:09", tz="UTC")


# ── Negative cases — rule branches enforced ────────────────────────────────
class TestRuleBranches:
    def test_full_clearout_not_iceberg(self):
        det = IcebergDetector()
        events = det.detect_events(_mbo_full_clearout_scenario())
        assert len(events) == 0

    def test_no_refill_not_iceberg(self):
        det = IcebergDetector()
        events = det.detect_events(_mbo_no_refill_scenario())
        assert len(events) == 0

    def test_slow_refill_not_iceberg(self):
        det = IcebergDetector()
        events = det.detect_events(_mbo_slow_refill_scenario())
        assert len(events) == 0

    def test_small_displayed_below_min_size(self):
        det = IcebergDetector(IcebergConfig(min_displayed_size=200.0))
        # Same iceberg scenario but min_displayed_size raised above the 100 in data
        events = det.detect_events(_mbo_iceberg_scenario())
        assert len(events) == 0


# ── Empty / edge inputs ────────────────────────────────────────────────────
class TestEmptyInputs:
    def test_empty_mbo_returns_empty_events(self):
        det = IcebergDetector()
        out = det.detect_events(pd.DataFrame(columns=(
            'ts_event', 'price', 'side', 'displayed_size',
            'executed_size', 'action',
        )))
        assert len(out) == 0

    def test_missing_column_raises(self):
        det = IcebergDetector()
        try:
            det.detect_events(pd.DataFrame({'ts_event': [pd.Timestamp.now()]}))
        except ValueError as e:
            assert "MBO column missing" in str(e)
            return
        raise AssertionError("expected ValueError on missing column")


# ── Per-bar aggregation ───────────────────────────────────────────────────
class TestAggregatePerBar:
    def test_aggregates_to_correct_bar(self):
        events = pd.DataFrame([
            dict(ts_event=pd.Timestamp("2025-04-01 08:02:30", tz="UTC"),
                 price_level=1.05, hidden_volume=300.0),
            dict(ts_event=pd.Timestamp("2025-04-01 08:07:45", tz="UTC"),
                 price_level=1.06, hidden_volume=500.0),
        ])
        bars = pd.DataFrame({
            'ts_event': pd.date_range("2025-04-01 08:00:00", periods=3,
                                       freq="5min", tz="UTC"),
        })
        det = IcebergDetector()
        out = det.aggregate_per_bar(events, bars)
        # bar 0 (08:00) gets the first event; bar 1 (08:05) gets the second
        assert int(out['iceberg_count_5m'].iloc[0]) == 1
        assert int(out['iceberg_count_5m'].iloc[1]) == 1
        assert int(out['iceberg_count_5m'].iloc[2]) == 0
        assert float(out['iceberg_total_volume_5m'].iloc[0]) == 300.0
        assert float(out['iceberg_total_volume_5m'].iloc[1]) == 500.0

    def test_empty_events_yields_zeros(self):
        bars = pd.DataFrame({
            'ts_event': pd.date_range("2025-04-01 08:00", periods=4,
                                       freq="5min", tz="UTC"),
        })
        det = IcebergDetector()
        out = det.aggregate_per_bar(pd.DataFrame(), bars)
        assert (out['iceberg_count_5m'] == 0).all()
        assert (out['iceberg_total_volume_5m'] == 0.0).all()


# ── Refinery integration (attach_iceberg_features) ────────────────────────
class TestAttachToBars:
    def test_no_mbo_ships_zeros(self):
        bars = pd.DataFrame({
            'ts_event': pd.date_range("2025-04-01 08:00", periods=5,
                                       freq="5min", tz="UTC"),
            'close': 1.0,
        })
        out = attach_iceberg_features(bars, mbo=None)
        assert 'iceberg_count_5m' in out.columns
        assert 'iceberg_total_volume_5m' in out.columns
        assert (out['iceberg_count_5m'] == 0).all()
        assert (out['iceberg_total_volume_5m'] == 0.0).all()
        # Other columns preserved
        assert 'close' in out.columns

    def test_with_mbo_runs_detector(self):
        bars = pd.DataFrame({
            'ts_event': pd.date_range("2025-04-01 08:00", periods=3,
                                       freq="5min", tz="UTC"),
        })
        out = attach_iceberg_features(bars, mbo=_mbo_iceberg_scenario())
        # The synthetic iceberg sits in bar 0 (ts < 08:05)
        assert int(out['iceberg_count_5m'].iloc[0]) == 1
        assert float(out['iceberg_total_volume_5m'].iloc[0]) == 300.0

    def test_keep_top_preserves_iceberg_cols(self):
        import prepare_day_trading as pdt
        n = 5
        df = pd.DataFrame({
            'ts_event': pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC"),
            'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0,
            'regime_label': "r", 'atr_14': 0.001,
            'is_session_break': np.zeros(n, dtype=np.int8),
            'obi': 0.0, 'obi_net': 0.0, 'cvd': 0.0, 'cvd_cumulative': 0.0,
            'iceberg_count_5m': np.zeros(n, dtype=np.int32),
            'iceberg_total_volume_5m': np.zeros(n, dtype=np.float32),
        })
        out = pdt._apply_phase1_feature_selection(df, mode='keep_top')
        assert 'iceberg_count_5m' in out.columns
        assert 'iceberg_total_volume_5m' in out.columns
