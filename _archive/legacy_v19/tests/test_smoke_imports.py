"""tests/test_smoke_imports.py — Phase 7: smoke tests للـ imports.

يضمن إن كل الـ modules الجديدة تُستورد بدون errors.
يكتشف:
  - circular imports
  - missing dependencies (graceful fallback)
  - typo في import paths
"""
from __future__ import annotations

import importlib
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestProjectModulesImport(unittest.TestCase):
    """الـ modules الجديدة في modules/ تستورد بنجاح."""

    def test_label_engine_v2(self):
        m = importlib.import_module("modules.label_engine_v2")
        self.assertTrue(hasattr(m, "label_triple_barrier_atr"))
        self.assertTrue(hasattr(m, "label_triple_barrier_atr_vectorized"))
        self.assertTrue(hasattr(m, "compute_atr"))

    def test_feature_enrichment(self):
        m = importlib.import_module("modules.feature_enrichment")
        self.assertTrue(hasattr(m, "enrich_features"))
        self.assertTrue(hasattr(m, "EnrichmentConfig"))

    def test_statistical_validation_layer(self):
        m = importlib.import_module("modules.statistical_validation_layer")
        self.assertTrue(hasattr(m, "discover_alphas"))
        self.assertTrue(hasattr(m, "AlphaSet"))

    def test_deeplob_v7ch(self):
        m = importlib.import_module("modules.deeplob_v7ch")
        self.assertTrue(hasattr(m, "build_7ch_tensor"))
        self.assertEqual(m.N_CHANNELS_V7, 7)

    def test_integration_bridge(self):
        m = importlib.import_module("modules.integration_bridge")
        self.assertTrue(hasattr(m, "IntegrationBridge"))
        self.assertTrue(hasattr(m, "TradingDecision"))


class TestV19_2ModulesImport(unittest.TestCase):
    """الـ V19.2 modules تستورد عبر sys.path injection."""

    def setUp(self):
        v19_path = os.path.join(_ROOT, "v19_2")
        if v19_path not in sys.path:
            sys.path.insert(0, v19_path)

    def test_market_specs(self):
        import market_specs
        self.assertTrue(hasattr(market_specs, "MarketSpec"))

    def test_statistics_module(self):
        import statistics_module
        self.assertTrue(hasattr(statistics_module, "benjamini_hochberg"))

    def test_feature_simulators(self):
        import feature_simulators
        self.assertTrue(hasattr(feature_simulators, "run_all_simulators"))
        self.assertEqual(len(feature_simulators.SIMULATOR_OUTPUTS), 18)

    def test_wall_depth_simulator(self):
        import wall_depth_simulator
        self.assertTrue(hasattr(wall_depth_simulator, "simulate_wall_depth"))
        self.assertEqual(len(wall_depth_simulator.WALL_DEPTH_OUTPUTS), 8)

    def test_iceberg_simulator(self):
        import iceberg_simulator
        self.assertTrue(hasattr(iceberg_simulator, "simulate_iceberg"))
        self.assertEqual(len(iceberg_simulator.ICEBERG_OUTPUTS), 5)

    def test_session_mapper(self):
        import session_mapper
        self.assertTrue(hasattr(session_mapper, "map_sessions"))

    def test_edge_scanner(self):
        import edge_scanner
        self.assertTrue(hasattr(edge_scanner, "scan_edges"))

    def test_edge_scanner_v2(self):
        import edge_scanner_v2
        self.assertTrue(hasattr(edge_scanner_v2, "scan_edges_v2"))

    def test_cluster_engine(self):
        import cluster_engine
        self.assertTrue(hasattr(cluster_engine, "select_distinct_alphas"))

    def test_bell_pairs(self):
        from quantum.bell_pairs import bell_pair, STANDARD_BELL_PAIRS
        self.assertEqual(len(STANDARD_BELL_PAIRS), 9)


class TestToolsImport(unittest.TestCase):
    """tools/ packages تستورد."""

    def test_tools_init(self):
        import tools
        self.assertIsNotNone(tools)

    def test_diagnostics_init(self):
        import tools.diagnostics
        self.assertIsNotNone(tools.diagnostics)


class TestNoDuplicatesLeft(unittest.TestCase):
    """تأكد إن duplicates المحذوفة فعلاً غير موجودة."""

    def test_no_dynamic_labels2(self):
        path = os.path.join(_ROOT, "tools", "legacy")
        self.assertFalse(
            os.path.isdir(path),
            msg="tools/legacy/ يجب أن يكون محذوفاً في Phase 7",
        )

    def test_no_catboost_brain2_in_modules(self):
        path = os.path.join(_ROOT, "modules", "catboost_brain2.py")
        self.assertFalse(os.path.exists(path))

    def test_no_catboost_brain3_in_modules(self):
        path = os.path.join(_ROOT, "modules", "catboost_brain3.py")
        self.assertFalse(os.path.exists(path))

    def test_no_dynamic_labels2_in_modules(self):
        path = os.path.join(_ROOT, "modules", "dynamic_labels2.py")
        self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
