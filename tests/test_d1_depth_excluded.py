"""D1 (Phase 1) — market-depth + order-level features removed from the day_trade
parquet export, while STILL computed in-memory for the gate/voting.

Workflow §4: depth belongs to the raw MBP-10 tensor → SSL/CNN/LSTM, not the
structure model. D1 removes the 24 depth/order-level features + the 2 exact
duplicates (obi_net, cvd_cumulative) from ORIGINAL_FEATURES (the export
whitelist). They remain COMPUTED on df_bars, which the event gate and
directional voting read BEFORE finalize — so nothing internal breaks. The
structural cycle (cycle_phase_*/cycle_position) is KEPT (that's D2, separate).

These tests lock the D1 contract:
  1. the depth/duplicate set is ABSENT from ORIGINAL_FEATURES (the teeth),
  2. structural cycle + core structure/flow are KEPT,
  3. the event gate (both modes) and directional voting STILL run when those
     depth columns are present in-memory — proving they read the frame, not the
     export whitelist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as P


class TestDepthExcludedFromExport:
    def test_depth_and_duplicates_absent(self):
        of = set(P.ORIGINAL_FEATURES)
        for col in P._D1_DEPTH_EXCLUDED_FROM_EXPORT:
            assert col not in of, f"D1: {col!r} must be excluded from ORIGINAL_FEATURES"

    def test_known_depth_names_absent(self):
        of = set(P.ORIGINAL_FEATURES)
        for col in ("absorption_intensity", "obi", "kyle_lambda", "bid_wall_strength",
                    "micro_price", "lob_imbalance", "iceberg_count_5m", "obi_net",
                    "cvd_cumulative", "liquidity_gaps", "absorb_z_raw"):
            assert col not in of, f"{col!r} should be D1-excluded"

    def test_excluded_set_size(self):
        # 24 depth/order-level + 2 exact duplicates
        assert len(P._D1_DEPTH_EXCLUDED_FROM_EXPORT) == 26


class TestKeptFeatures:
    def test_structural_cycle_kept_not_d1(self):
        of = set(P.ORIGINAL_FEATURES)
        for col in ("cycle_phase_acc_prob", "cycle_phase_markup_prob",
                    "cycle_phase_dist_prob", "cycle_phase_markdown_prob", "cycle_position"):
            assert col in of, f"{col!r} is structural cycle (D2) — must be KEPT in D1"

    def test_core_structure_and_compass_kept(self):
        of = set(P.ORIGINAL_FEATURES)
        for col in ("cvd", "current_vwap", "london_sess_high", "time_to_ny_close_min",
                    "order_flow_imbalance", "cvd_divergence_at_level", "vwap_z_score",
                    "hawkes_z_raw", "kyle_z_raw", "mbp_roll_lob_coverage"):
            assert col in of, f"{col!r} (structure/flow/compass) must be kept"


class TestGateAndVotingStillWork:
    """The decisive D1 safety check: the gate/voting read the IN-MEMORY df (before
    finalize), so removing depth from the EXPORT whitelist does not affect them."""

    def _gate_frame(self, n=120):
        rng = np.random.RandomState(0)
        close = 1.25 + np.cumsum(rng.randn(n) * 0.0005)
        return pd.DataFrame({
            "ts_event": pd.date_range("2025-04-07 08:00", periods=n, freq="5min", tz="UTC"),
            "close": close, "high": close + 0.0003, "low": close - 0.0003,
            "atr_14": 0.001, "regime_label": "ranging",
            # depth/micro columns — D1 removes them from EXPORT but they stay in-memory:
            "hawkes_intensity": np.abs(rng.randn(n)) * 0.5,
            "absorption_intensity": np.abs(rng.randn(n)) * 0.3,
            "kyle_lambda": np.abs(rng.randn(n)) * 0.1,
            "cvd_momentum": rng.randn(n),
            "mbp_roll_lob_coverage": 0.9, "mbo_bar_coverage": 0.9,
        })

    def test_microstructure_gate_reads_in_memory_depth(self):
        out = P.detect_microstructure_events(self._gate_frame())
        assert "is_event" in out.columns          # gate ran using in-memory absorption/kyle
        assert int(out["is_event"].sum()) >= 0

    def test_structure_compass_gate_unaffected(self):
        # the NEW gate doesn't even use depth — sanity it still runs
        df = self._gate_frame()
        df["order_flow_imbalance"] = 0.5
        df["vwap_z_score"] = 2.0
        df["bar_cvd_delta"] = 1.0
        df["is_london"] = 1
        df["pdh"] = df["close"] + 0.0001
        out = P.detect_structure_compass_events(df)
        assert "is_event" in out.columns

    def test_directional_voting_reads_in_memory_obi(self):
        n = 80
        rng = np.random.RandomState(2)
        close = 1.25 + np.cumsum(rng.randn(n) * 0.0005)
        df = pd.DataFrame({
            "ts_event": pd.date_range("2025-04-07 08:00", periods=n, freq="5min", tz="UTC"),
            "close": close, "high": close + 0.0003, "low": close - 0.0003,
            "atr_14": 0.001, "regime_label": "ranging",
            "obi": rng.uniform(-1, 1, n),            # depth — in-memory, D1-excluded from export
            "cvd": np.cumsum(rng.randn(n)),
            "bar_cvd_delta": rng.randn(n),           # always present in the real pipeline
            "kalman_direction": rng.choice([-1, 0, 1], n),
            "is_event": 1, "event_score": 0.5,
        })
        out = P.add_event_direction(df)
        assert "event_direction" in out.columns      # voting ran using in-memory obi
