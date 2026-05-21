"""tests/test_full_pipeline.py — Phase 7: Full pipeline regression test.

يشغّل كل المراحل المُكتمَلة في pipeline واحد على synthetic data:
  1. Phase 2: label_engine_v2 (Triple Barrier)
  2. Phase 3: feature_enrichment (V19.2 simulators + Bell pairs)
  3. Phase 4: statistical_validation_layer (alpha discovery)
  4. Phase 5: deeplob_v7ch (7-channel tensors)
  5. Phase 6: integration_bridge (decisions)

ضمان عدم وجود regressions في الـ end-to-end flow.
"""
from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd


def _make_full_pipeline_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """يحاكي DataFrame مع كل الأعمدة المطلوبة."""
    rng = np.random.RandomState(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 100.0 + np.cumsum(rng.randn(n) * 0.05)
    high = close + np.abs(rng.randn(n) * 0.1)
    low = close - np.abs(rng.randn(n) * 0.1)

    cols = {
        "ts_event": ts,
        "open": close + rng.randn(n) * 0.02,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.uniform(100, 1000, n),
    }
    # MBP-10
    for i in range(10):
        cols[f"bid_sz_{i:02d}"] = rng.uniform(0, 100, n)
        cols[f"ask_sz_{i:02d}"] = rng.uniform(0, 100, n)
    cols["trade_size"] = rng.uniform(1, 50, n)
    cols["side"] = rng.choice(["B", "S"], n)
    return pd.DataFrame(cols)


class TestFullPipeline(unittest.TestCase):
    """End-to-end pipeline على synthetic data."""

    def test_phase_2_labels(self):
        from modules.label_engine_v2 import (
            TripleBarrierConfig, compute_atr, label_distribution,
            label_triple_barrier_atr_vectorized,
        )
        df = _make_full_pipeline_df(n=500)
        atr = compute_atr(df["high"].to_numpy(), df["low"].to_numpy(),
                          df["close"].to_numpy(), window=14)
        cfg = TripleBarrierConfig(tp_atr_mult=2.0, sl_atr_mult=2.0, horizon_default=20)
        result = label_triple_barrier_atr_vectorized(df["close"].to_numpy(), atr, config=cfg)
        dist = label_distribution(result["bias"])
        # ما يكون كله NEUTRAL (98% bug)
        self.assertLess(dist["neutral"], 0.95, msg=f"too much NEUTRAL: {dist}")

    def test_phase_3_enrichment(self):
        from modules.feature_enrichment import enrich_features, EnrichmentConfig
        df = _make_full_pipeline_df(n=300)
        cfg = EnrichmentConfig(
            add_simulators=True, add_wall_depth=False,
            add_iceberg=False, add_session_mapping=True,
            add_bell_pairs=True,
        )
        out = enrich_features(df, config=cfg)
        # الـ sim_* outputs مضافة (18 من feature_simulators)
        sim_cols = [c for c in out.columns if c.startswith("sim_")]
        self.assertGreaterEqual(len(sim_cols), 18)
        # الـ bp_* المتاحة (بدون wall_depth/iceberg يكون ≥5)
        bp_cols = [c for c in out.columns if c.startswith("bp_")]
        self.assertGreaterEqual(len(bp_cols), 5)
        # session mapping أضاف zone_full
        self.assertIn("zone_full", out.columns)

    def test_phase_4_discovery(self):
        from modules.feature_enrichment import enrich_features, EnrichmentConfig
        from modules.statistical_validation_layer import discover_alphas
        df = _make_full_pipeline_df(n=500)
        cfg = EnrichmentConfig(
            add_simulators=True, add_wall_depth=False, add_iceberg=False,
            add_session_mapping=True, add_bell_pairs=False,
        )
        enriched = enrich_features(df, config=cfg)
        # discovery
        alpha_set = discover_alphas(
            enriched, run_permutation=False, verbose=False,
        )
        # ما يرمي exception، diagnostics موجودة
        self.assertIn("cells_scanned", alpha_set.diagnostics)

    def test_phase_5_tensors(self):
        from modules.deeplob_v7ch import build_7ch_tensor, N_CHANNELS_V7
        df = _make_full_pipeline_df(n=200)
        # محتاج sim_* outputs أيضاً
        rng = np.random.RandomState(0)
        for col in ["sim_iceberg_strength", "sim_wall_persist",
                    "sim_informed_prob", "sim_depth_imbalance"]:
            df[col] = rng.uniform(0, 1, len(df))
        t = build_7ch_tensor(df, time_steps=50)
        self.assertEqual(t.shape[-1], N_CHANNELS_V7)
        self.assertEqual(t.shape[1], 50)

    def test_phase_6_bridge(self):
        from modules.integration_bridge import (
            Action, BridgeConfig, IntegrationBridge,
        )
        from modules.statistical_validation_layer import AlphaSet
        # alpha يدوي
        alpha = {
            "zone": "asia_q1", "level": "PDH", "event": "touch",
            "combo": "LONG_dp_pos", "direction": 1, "horizon": 6,
            "filters": ["depth_pressure_pos"],
        }
        alpha_set = AlphaSet(candidates=[alpha])
        bridge = IntegrationBridge(alpha_set)
        # matching row + DL confirm
        row = pd.Series({
            "zone_full": "asia_q1",
            "PDH_event": "touch",
            "sim_depth_pressure": 0.30,
            "sim_wall_consumed": 0.10,
        })
        d = bridge.evaluate_row(
            row, dl_proba={"long": 0.70, "short": 0.20, "neutral": 0.10}
        )
        self.assertEqual(d.action, Action.OPEN_LONG)

    def test_phases_chained(self):
        """تشغيل phases 2 → 3 → 6 في chain واحد."""
        from modules.feature_enrichment import enrich_features, EnrichmentConfig
        from modules.statistical_validation_layer import AlphaSet
        from modules.integration_bridge import (
            Action, IntegrationBridge,
        )

        df = _make_full_pipeline_df(n=200)
        cfg = EnrichmentConfig(
            add_simulators=True, add_wall_depth=False, add_iceberg=False,
            add_session_mapping=True, add_bell_pairs=False,
        )
        enriched = enrich_features(df, config=cfg)
        # تأكد الأعمدة المطلوبة للـ bridge موجودة
        self.assertIn("zone_full", enriched.columns)
        self.assertIn("sim_depth_pressure", enriched.columns)

        # bridge
        alpha = {
            "zone": enriched["zone_full"].iloc[0],
            "level": "PDH", "event": "touch",
            "combo": "LONG_dp_pos", "direction": 1, "horizon": 6,
            "filters": ["depth_pressure_pos"],
        }
        bridge = IntegrationBridge(AlphaSet(candidates=[alpha]))
        # تشغيل batch
        n = len(enriched)
        proba = np.full((n, 3), [0.7, 0.2, 0.1])
        decisions = bridge.evaluate_batch(enriched, proba)
        self.assertEqual(len(decisions), n)


if __name__ == "__main__":
    unittest.main(verbosity=2)
