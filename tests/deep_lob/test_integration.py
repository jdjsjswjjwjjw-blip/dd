"""Integration tests: adapters + system integration hooks."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestDeepLOBCNNAdapter(unittest.TestCase):
    def test_predict_shape(self):
        from modules.deep_lob import DeepLOBCNNAdapter, DeepLOBConfig
        adapter = DeepLOBCNNAdapter(
            channels=7, config=DeepLOBConfig.small_dev(), device="cpu",
            brain_file="/tmp/nonexistent_for_test.pt",
        )
        # LOB tensor: (N=4, T=10, P=20, C=7)
        lob = np.random.randn(4, 10, 20, 7).astype(np.float32)
        emb = adapter.predict(lob)
        self.assertEqual(emb.shape, (4, 8))

    def test_single_sample(self):
        from modules.deep_lob import DeepLOBCNNAdapter, DeepLOBConfig
        adapter = DeepLOBCNNAdapter(
            channels=7, config=DeepLOBConfig.small_dev(), device="cpu",
            brain_file="/tmp/nonexistent_for_test.pt",
        )
        # Single sample (T, P, C) → adds batch dim
        lob_single = np.random.randn(10, 20, 7).astype(np.float32)
        emb = adapter.predict(lob_single)
        self.assertEqual(emb.shape, (1, 8))


class TestBridgeAdapter(unittest.TestCase):
    def test_predict_proba_shape(self):
        from modules.deep_lob import (
            BridgeAdapter, DeepLOBConfig, HierarchicalLOBTransformer,
            BarLOB, OrderBatch, Order, OrderSide, OrderType,
        )
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        adapter = BridgeAdapter(model, device="cpu")
        orders = [
            Order(i*1000, OrderSide.BUY, OrderType.TRADE, 1.2345, 100)
            for i in range(5)
        ]
        bar = BarLOB(
            bar_start_ns=0, bar_end_ns=60_000_000_000, mid_price=1.2345,
            orders=OrderBatch.from_orders(orders),
            context={f"f{i}": float(i) for i in range(10)},
        )
        proba = adapter.predict_proba(bar)
        self.assertEqual(proba.shape, (3,))
        self.assertAlmostEqual(float(proba.sum()), 1.0, places=4)

    def test_predict_with_targets(self):
        from modules.deep_lob import (
            BridgeAdapter, DeepLOBConfig, HierarchicalLOBTransformer,
            BarLOB, OrderBatch, Order, OrderSide, OrderType,
        )
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        adapter = BridgeAdapter(model, device="cpu")
        orders = [Order(i*1000, OrderSide.BUY, OrderType.TRADE, 1.2345, 100) for i in range(5)]
        bar = BarLOB(0, 60_000_000_000, 1.2345, orders=OrderBatch.from_orders(orders))
        out = adapter.predict_with_targets(bar)
        for key in ("dl_proba", "next_price", "next_imbalance",
                    "next_volatility", "next_regime_probs",
                    "wall_persist", "time_to_event"):
            self.assertIn(key, out)


class TestDynamicTPSL(unittest.TestCase):
    def test_compute_dynamic_tp_sl(self):
        from modules.deep_lob import compute_dynamic_tp_sl
        # High volatility → TP wider
        preds_high_vol = {
            "next_volatility": 2.0,
            "wall_persist": 1.0,
            "time_to_event": 10.0,
        }
        tp_high, sl_high = compute_dynamic_tp_sl(
            preds_high_vol, base_tp_mult=2.0, base_sl_mult=1.0,
        )
        # Low volatility → TP narrower
        preds_low_vol = {
            "next_volatility": 0.1,
            "wall_persist": 1.0,
            "time_to_event": 10.0,
        }
        tp_low, sl_low = compute_dynamic_tp_sl(
            preds_low_vol, base_tp_mult=2.0, base_sl_mult=1.0,
        )
        self.assertGreater(tp_high, tp_low)

    def test_wall_caution_reduces_tp(self):
        from modules.deep_lob import compute_dynamic_tp_sl
        no_wall = {"next_volatility": 1.0, "wall_persist": 1.0, "time_to_event": 10.0}
        long_wall = {"next_volatility": 1.0, "wall_persist": 20.0, "time_to_event": 10.0}
        tp_nw, _ = compute_dynamic_tp_sl(no_wall, base_tp_mult=2.0, base_sl_mult=1.0)
        tp_lw, _ = compute_dynamic_tp_sl(long_wall, base_tp_mult=2.0, base_sl_mult=1.0)
        self.assertLess(tp_lw, tp_nw)


class TestSystemReadinessCheck(unittest.TestCase):
    def test_runs_without_error(self):
        from modules.deep_lob import system_readiness_check
        result = system_readiness_check()
        self.assertIn("ready", result)
        self.assertIn("checks", result)
        self.assertIn("recommendations", result)

    def test_includes_pytorch_check(self):
        from modules.deep_lob import system_readiness_check
        result = system_readiness_check()
        self.assertIn("pytorch", result["checks"])
        # PyTorch is installed in this env
        self.assertTrue(result["checks"]["pytorch"]["available"])


class TestVisualEmbedderFactory(unittest.TestCase):
    def test_factory_transformer_mode(self):
        from modules.deep_lob import get_visual_embedder, DeepLOBConfig
        from modules.deep_lob import DeepLOBCNNAdapter
        embedder = get_visual_embedder(
            use_transformer=True,
            transformer_checkpoint="/tmp/no_such_ckpt.pt",
            channels=7,
            config=DeepLOBConfig.small_dev(),
        )
        self.assertIsInstance(embedder, DeepLOBCNNAdapter)


class TestExtendAlphaLibrary(unittest.TestCase):
    def test_extend_with_empty_patterns(self):
        from modules.deep_lob.system_integration import extend_alpha_library_with_discoveries
        with tempfile.TemporaryDirectory() as td:
            lib_path = Path(td) / "test_library.json"
            result = extend_alpha_library_with_discoveries(
                str(lib_path), [], backup=False,
            )
            self.assertEqual(result["n_added"], 0)
            self.assertTrue(lib_path.exists())

    def test_extend_with_discovered_pattern(self):
        from modules.deep_lob.pattern_discovery import DiscoveredPattern
        from modules.deep_lob.system_integration import extend_alpha_library_with_discoveries

        pattern = DiscoveredPattern(
            pattern_id="test_001",
            method="cluster",
            score=2.5,
            n_examples=30,
            centroid_embedding=np.zeros(8),
            metadata={"win_rate": 0.7, "sharpe": 1.5, "mean_outcome": 0.05,
                      "regime": "trending"},
        )

        with tempfile.TemporaryDirectory() as td:
            lib_path = Path(td) / "lib.json"
            result = extend_alpha_library_with_discoveries(
                str(lib_path), [pattern], min_score=1.0, backup=False,
            )
            self.assertEqual(result["n_added"], 1)
            self.assertEqual(result["library_total"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
