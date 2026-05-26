"""tests/test_feature_enrichment.py — Phase 3 tests."""
from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd


def _make_mock_prepare_day_trading_output(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """يحاكي مخرج prepare_day_trading: 87 feature column + market cols."""
    rng = np.random.RandomState(seed)
    ts = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1min", tz="UTC")

    cols = {
        # core market
        "ts_event": ts,
        "open": 100.0 + rng.randn(n) * 0.5,
        "high": 100.5 + rng.randn(n) * 0.5,
        "low": 99.5 + rng.randn(n) * 0.5,
        "close": 100.0 + np.cumsum(rng.randn(n) * 0.1),
        "volume": rng.randint(100, 1000, n).astype(float),
        # MBP fields (لـ wall_depth_simulator)
    }
    # MBP-10 levels (bid/ask px + sz)
    for i in range(10):
        cols[f"bid_px_0{i}"] = 100.0 - 0.01 * (i + 1) + rng.randn(n) * 0.01
        cols[f"ask_px_0{i}"] = 100.0 + 0.01 * (i + 1) + rng.randn(n) * 0.01
        cols[f"bid_sz_0{i}"] = rng.randint(10, 200, n).astype(float)
        cols[f"ask_sz_0{i}"] = rng.randint(10, 200, n).astype(float)
    # MBO-style fields (لـ iceberg)
    cols["bid_qty"] = cols["bid_sz_00"]
    cols["ask_qty"] = cols["ask_sz_00"]
    cols["trade_size"] = rng.randint(1, 50, n).astype(float)
    cols["side"] = rng.choice(["B", "S"], n)

    # Mock "87 features" — placeholder columns
    for i in range(87):
        cols[f"feature_{i:02d}"] = rng.randn(n)

    return pd.DataFrame(cols)


class TestEnrichmentBasic(unittest.TestCase):
    def test_imports(self):
        from modules.feature_enrichment import (
            EnrichmentConfig, enrich_features,
            list_added_columns, feature_count_summary,
        )
        self.assertIsNotNone(enrich_features)

    def test_enrich_with_default_config(self):
        from modules.feature_enrichment import enrich_features
        df = _make_mock_prepare_day_trading_output(n=200)
        original_cols = len(df.columns)
        out = enrich_features(df)
        # العدد الإجمالي يزيد
        self.assertGreater(len(out.columns), original_cols)
        # الأعمدة الأصلية تبقى
        for c in df.columns:
            self.assertIn(c, out.columns)

    def test_non_destructive(self):
        """enrich_features لا يعدّل DataFrame الأصلي."""
        from modules.feature_enrichment import enrich_features
        df = _make_mock_prepare_day_trading_output(n=100)
        original_cols = list(df.columns)
        _ = enrich_features(df)
        self.assertEqual(list(df.columns), original_cols)

    def test_skip_missing_cols(self):
        """لو نواقص MBP، الـ wall_depth يتخطّى بدون crash."""
        from modules.feature_enrichment import enrich_features, EnrichmentConfig
        df = _make_mock_prepare_day_trading_output(n=100)
        # احذف بعض أعمدة الـ MBP
        df = df.drop(columns=[c for c in df.columns if "bid_px" in c or "ask_px" in c])
        cfg = EnrichmentConfig(skip_missing_cols=True, verbose=False)
        # ما يرمي exception
        out = enrich_features(df, config=cfg)
        self.assertIsNotNone(out)


class TestConfigToggle(unittest.TestCase):
    """flags تتحكّم في الـ stages."""

    def test_only_bell_pairs(self):
        from modules.feature_enrichment import enrich_features, EnrichmentConfig
        df = _make_mock_prepare_day_trading_output(n=100)
        # أضف sim_* columns مسبقاً (محاكاة الحالة بعد simulators)
        rng = np.random.RandomState(0)
        for col in ["sim_wall_persist", "sim_iceberg_strength", "sim_depth_pressure",
                    "sim_informed_prob", "sim_sweep_signal", "sim_absorb_intensity",
                    "sim_volatility_regime", "sim_flow_consistency", "sim_wall_real",
                    "sim_wall_consumed", "sim_flow_direction", "sim_depth_imbalance",
                    "sim_iceberg_side", "sim_informed_direction", "sim_wall_growth",
                    "sim_iceberg_replenish", "sim_liquidity_state", "sim_data_quality"]:
            df[col] = rng.rand(len(df))

        cfg = EnrichmentConfig(
            add_simulators=False,
            add_wall_depth=False,
            add_iceberg=False,
            add_session_mapping=False,
            add_bell_pairs=True,
        )
        out = enrich_features(df, config=cfg)
        bp_cols = [c for c in out.columns if c.startswith("bp_")]
        self.assertEqual(len(bp_cols), 9, msg=f"expected 9 bp_*, got {len(bp_cols)}")

    def test_disable_all_returns_same_count(self):
        from modules.feature_enrichment import enrich_features, EnrichmentConfig
        df = _make_mock_prepare_day_trading_output(n=100)
        cfg = EnrichmentConfig(
            add_simulators=False, add_wall_depth=False,
            add_iceberg=False, add_session_mapping=False,
            add_bell_pairs=False,
        )
        out = enrich_features(df, config=cfg)
        self.assertEqual(len(out.columns), len(df.columns))


class TestFeatureCountSummary(unittest.TestCase):
    def test_default_count_summary(self):
        """التقرير: 87 + ~40 = ~127-138 feature."""
        from modules.feature_enrichment import feature_count_summary
        counts = feature_count_summary()
        # 18 simulators + 8 wall + 5 iceberg + 1 zone + 12 levels + 9 bp = 53
        self.assertGreaterEqual(counts["total_added"], 40)
        self.assertEqual(counts["simulators"], 18)
        self.assertEqual(counts["wall_depth"], 8)
        self.assertEqual(counts["iceberg"], 5)
        self.assertEqual(counts["bell_pairs"], 9)

    def test_partial_config_summary(self):
        from modules.feature_enrichment import feature_count_summary, EnrichmentConfig
        cfg = EnrichmentConfig(
            add_simulators=True, add_wall_depth=False,
            add_iceberg=False, add_session_mapping=False,
            add_bell_pairs=True,
        )
        counts = feature_count_summary(cfg)
        self.assertEqual(counts["simulators"], 18)
        self.assertEqual(counts["bell_pairs"], 9)
        self.assertEqual(counts["total_added"], 18 + 9)


class TestListAddedColumns(unittest.TestCase):
    def test_lists_bp_columns(self):
        from modules.feature_enrichment import list_added_columns
        cols = list_added_columns()
        bp_cols = [c for c in cols if c.startswith("bp_")]
        self.assertEqual(len(bp_cols), 9)

    def test_lists_simulator_columns(self):
        from modules.feature_enrichment import list_added_columns
        cols = list_added_columns()
        self.assertIn("sim_depth_pressure", cols)
        self.assertIn("sim_iceberg_strength", cols)


if __name__ == "__main__":
    unittest.main(verbosity=2)
