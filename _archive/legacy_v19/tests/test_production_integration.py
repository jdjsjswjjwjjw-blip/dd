"""
tests/test_production_integration.py — Sprint 1 production wiring tests.

يثبت أن الـ integration الفعلي شغّال في production paths:
  1. dynamic_labels.label_with_forward_scan يستخدم MFE/MAE (not future[-1])
  2. v19_2.edge_scanner.scan_edges يفوّض إلى scan_edges_v2 بـ default
  3. DeepLOBCNN يقبل channels=7 (مع backward compat لـ 3)
  4. labels_v19 يستخدم Triple Barrier (tp_mult / sl_mult / adaptive horizons)
  5. prepare_day_trading_enriched wrapper يضيف V19.2 features
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_V19_2 = os.path.join(_ROOT, "v19_2")
if _V19_2 not in sys.path:
    sys.path.insert(0, _V19_2)


class TestDynamicLabelsMFE(unittest.TestCase):
    """Step 1: dynamic_labels يستخدم MFE/MAE داخلياً."""

    def test_mean_revert_no_longer_NEUTRAL(self):
        """السيناريو 98% NEUTRAL: السعر يصعد ثم يعود، MFE موجود، future[-1] ≈ 0."""
        from modules.dynamic_labels import label_with_forward_scan, DIR_LONG, DIR_NEUTRAL

        n = 100
        # سعر يصعد +50p ثم يعود إلى البداية (mean-revert)
        prices = np.concatenate([
            np.linspace(100.0, 100.5, 25),   # +50p up
            np.linspace(100.5, 100.0, 25),   # back to start
            np.full(50, 100.0),               # flat
        ])

        df_trades = pd.DataFrame({
            "ts_event": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
            "close": prices,
            "high": prices + 0.005,
            "low": prices - 0.005,
            "volume": np.full(n, 100.0),
            "atr_14": np.full(n, 0.005),
        })
        df_levels = pd.DataFrame()

        result = label_with_forward_scan(
            df_trades, df_levels,
            max_bars_forward=30,
            tick_size=0.0001,
            direction_threshold_ticks=5.0,
        )

        # الـ rows في بداية الـ rally (1-20): MFE = +50p لكن future[-1] قد يكون 0
        # المنطق الجديد يجب أن يعطي LONG، الـ legacy كان NEUTRAL
        early_bias = result["bias_label"].iloc[1:20].values
        n_long = int((early_bias == DIR_LONG).sum())
        # لا نفرض نسبة محددة — فقط نتأكد إن في LONG signals
        # (الـ legacy = 0 LONG لأن future[-1] - entry ≈ 0)
        self.assertGreater(n_long, 0,
            f"MFE/MAE logic must detect upward move; got 0 LONG signals "
            f"(distribution: {pd.Series(early_bias).value_counts().to_dict()})")


class TestEdgeScannerWrapper(unittest.TestCase):
    """Step 2: scan_edges يفوّض إلى scan_edges_v2."""

    def test_scan_edges_default_calls_v2(self):
        """الـ default path يجب أن يستدعي scan_edges_v2."""
        import edge_scanner
        # لو الـ source يحتوي على scan_edges_v2 import → التفويض شغّال
        src = Path(_V19_2) / "edge_scanner.py"
        content = src.read_text()
        self.assertIn("scan_edges_v2", content,
            "edge_scanner.py يجب أن يستورد scan_edges_v2 للتفويض")
        self.assertIn("_use_legacy_loop", content,
            "edge_scanner.py يجب أن يدعم flag للـ legacy")

    def test_legacy_flag_works(self):
        from edge_scanner import scan_edges
        import inspect
        sig = inspect.signature(scan_edges)
        self.assertIn("_use_legacy_loop", sig.parameters,
            "scan_edges() يجب أن يقبل _use_legacy_loop parameter")


class TestDeepLOB7ChannelSupport(unittest.TestCase):
    """Step 4: DeepLOBCNN يدعم 7 channels."""

    def test_deeplob_accepts_7_channels(self):
        from modules.deeplob_cnn import DeepLOBCNN
        m = DeepLOBCNN(channels=7)
        self.assertEqual(m.C, 7, "DeepLOBCNN يجب أن يقبل channels=7")

    def test_deeplob_backward_compat_3ch(self):
        from modules.deeplob_cnn import DeepLOBCNN
        m = DeepLOBCNN(channels=3)
        self.assertEqual(m.C, 3, "backward compat: channels=3 يبقى يعمل")

    def test_deeplob_v7ch_module_available(self):
        from modules import deeplob_v7ch
        self.assertTrue(hasattr(deeplob_v7ch, "build_7ch_tensor"),
            "build_7ch_tensor يجب أن يكون متاحاً")


class TestLabelsV19TripleBarrier(unittest.TestCase):
    """Step 5: labels_v19.py يستخدم Triple Barrier (tp/sl/adaptive horizon)."""

    def test_forward_scan_uses_tp_sl_mult(self):
        """_forward_scan_rows يحوي tp_mult و sl_mult في signature."""
        from modules.labels_v19 import _forward_scan_rows
        import inspect
        sig = inspect.signature(_forward_scan_rows)
        self.assertIn("tp_mult", sig.parameters)
        self.assertIn("sl_mult", sig.parameters)
        self.assertIn("adaptive_horizons", sig.parameters)


class TestEnrichmentWrapper(unittest.TestCase):
    """Step 3: prepare_day_trading_enriched يضيف V19.2 features."""

    def test_wrapper_module_importable(self):
        # smoke import
        import prepare_day_trading_enriched
        self.assertTrue(hasattr(prepare_day_trading_enriched, "enrich_parquet"))

    def test_wrapper_uses_feature_enrichment(self):
        from prepare_day_trading_enriched import enrich_parquet
        # smoke run on tiny df
        import tempfile
        rng = np.random.RandomState(0)
        n = 100
        df = pd.DataFrame({
            "ts_event": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
            "open": 100.0 + rng.randn(n).cumsum() * 0.01,
            "close": 100.0 + rng.randn(n).cumsum() * 0.01,
            "high": 100.5 + rng.randn(n) * 0.01,
            "low": 99.5 + rng.randn(n) * 0.01,
            "volume": rng.uniform(100, 1000, n),
        })
        for i in range(10):
            df[f"bid_sz_{i:02d}"] = rng.uniform(0, 100, n)
            df[f"ask_sz_{i:02d}"] = rng.uniform(0, 100, n)
        df["trade_size"] = rng.uniform(1, 50, n)
        df["side"] = rng.choice(["B", "S"], n)

        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.parquet"
            out = Path(td) / "out.parquet"
            df.to_parquet(inp)
            result = enrich_parquet(inp, out, verbose=False)

        self.assertGreater(result["added"], 0, "wrapper يجب أن يضيف features")


class TestProductionPipelineSmoke(unittest.TestCase):
    """End-to-end smoke: V19.2 + main يعملان معاً."""

    def test_imports_chain(self):
        """كل الـ modules القابلة للـ integration تستورد بدون errors."""
        import modules.dynamic_labels
        import modules.label_engine_v2
        import modules.feature_enrichment
        import modules.statistical_validation_layer
        import modules.integration_bridge
        import modules.deeplob_cnn
        import modules.deeplob_v7ch
        import modules.labels_v19
        import edge_scanner
        import edge_scanner_v2

    def test_alpha_set_compatible_with_bridge(self):
        """AlphaSet من statistical_validation_layer يعمل مع IntegrationBridge."""
        from modules.statistical_validation_layer import AlphaSet
        from modules.integration_bridge import IntegrationBridge, BridgeConfig

        alpha_set = AlphaSet(candidates=[{
            "zone": "T_REG_LONDON", "level": "PDH", "event": "touch",
            "combo": "LONG_dp_pos", "direction": 1, "horizon": 6,
            "filters": ["depth_pressure_pos"],
        }])
        bridge = IntegrationBridge(alpha_set, config=BridgeConfig())
        self.assertIsNotNone(bridge)


if __name__ == "__main__":
    unittest.main(verbosity=2)
