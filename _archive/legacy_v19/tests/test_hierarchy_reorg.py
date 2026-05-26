"""
tests/test_hierarchy_reorg.py — Sprint 8 tests.

يحرس الـ Hierarchy الموحد (التقرير 3.2):
    core/, simulators/, context/, discovery/, dl_pipeline/, training/, deployment/

كل package = facade re-export. backward compat 100%.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestHierarchyDirectoriesExist(unittest.TestCase):
    """التقرير 3.2: الـ 7 مجلدات الجديدة (training/ في Sprint 7 separate PR)."""

    # Sprint 8 يبني: core, simulators, context, discovery, dl_pipeline, deployment.
    # training/ يأتي من Sprint 7 (PR منفصل) — لا نُلزِم وجوده هنا.
    EXPECTED = [
        "core", "simulators", "context", "discovery",
        "dl_pipeline", "deployment",
    ]

    def test_all_directories_exist(self):
        for name in self.EXPECTED:
            path = os.path.join(_ROOT, name)
            self.assertTrue(os.path.isdir(path), f"{name}/ مفقود")
            init = os.path.join(path, "__init__.py")
            self.assertTrue(os.path.exists(init), f"{name}/__init__.py مفقود")


class TestCoreFacade(unittest.TestCase):
    def test_core_init_references_v19_2_modules(self):
        src = open(os.path.join(_ROOT, "core", "__init__.py")).read()
        for mod in ["market_specs", "statistics_module", "fix_ohlc",
                    "combine_months", "combine_raw"]:
            self.assertIn(mod, src)

    def test_core_loadable(self):
        # facade يحمّل بدون errors حتى لو بعض الـ modules ناقصة (graceful)
        import core
        self.assertTrue(hasattr(core, "__all__"))


class TestSimulatorsFacade(unittest.TestCase):
    def test_simulators_init(self):
        src = open(os.path.join(_ROOT, "simulators", "__init__.py")).read()
        for mod in ["feature_simulators", "wall_depth_simulator", "iceberg_simulator"]:
            self.assertIn(mod, src)

    def test_simulators_loadable(self):
        import simulators
        self.assertTrue(hasattr(simulators, "__all__"))


class TestContextFacade(unittest.TestCase):
    def test_context_init(self):
        src = open(os.path.join(_ROOT, "context", "__init__.py")).read()
        for mod in ["session_mapper", "liquidity_topology_engine"]:
            self.assertIn(mod, src)

    def test_context_loadable(self):
        import context
        self.assertTrue(hasattr(context, "__all__"))


class TestDiscoveryFacade(unittest.TestCase):
    def test_discovery_init(self):
        src = open(os.path.join(_ROOT, "discovery", "__init__.py")).read()
        # V19.2 discovery
        for mod in ["edge_scanner", "edge_scanner_v2", "cluster_engine",
                    "run_pipeline", "show_candidates"]:
            self.assertIn(mod, src)
        # Phase C quantum
        for mod in ["grover_alpha_search", "qaoa_select_subset",
                    "vqe_find_alpha_mode", "quantum_cluster"]:
            self.assertIn(mod, src)

    def test_discovery_loadable(self):
        import discovery
        self.assertTrue(hasattr(discovery, "__all__"))


class TestDLPipelineFacade(unittest.TestCase):
    def test_dl_pipeline_init(self):
        src = open(os.path.join(_ROOT, "dl_pipeline", "__init__.py")).read()
        for mod in ["tensor_builder", "label_engine_v2", "deeplob_v7ch", "feature_enrichment"]:
            self.assertIn(mod, src)

    def test_dl_pipeline_loadable(self):
        import dl_pipeline
        self.assertTrue(hasattr(dl_pipeline, "__all__"))


class TestDeploymentFacade(unittest.TestCase):
    def test_deployment_init(self):
        src = open(os.path.join(_ROOT, "deployment", "__init__.py")).read()
        for fn in ["get_backtest_v19", "get_walkforward_v19", "get_predict_v19",
                   "get_paper_v19", "get_live_predictor", "get_integration_bridge"]:
            self.assertIn(fn, src)

    def test_deployment_loadable(self):
        import deployment
        self.assertTrue(hasattr(deployment, "__all__"))

    def test_deployment_uses_lazy_pattern(self):
        """deployment/ functions يجب أن lazy-load (heavy deps)."""
        src = open(os.path.join(_ROOT, "deployment", "__init__.py")).read()
        # functions يجب أن لا يكون فيهم top-level imports للـ heavy modules
        top_level_imports = [
            line for line in src.split("\n")
            if line.startswith("import predict_v19") or
               line.startswith("import backtest_v19") or
               line.startswith("import paper_v19")
        ]
        self.assertEqual(len(top_level_imports), 0,
            "deployment/ يجب lazy-load heavy modules داخل functions")


class TestBackwardCompatibility(unittest.TestCase):
    """التأكد إن backward compat 100%."""

    def test_modules_still_at_root(self):
        """الـ modules القديمة لازالت في الجذر (لم نُنقَل)."""
        for name in ["paper_v19.py", "live_predictor.py", "predict_v19.py",
                     "prepare_day_trading.py", "backtest_v19.py"]:
            self.assertTrue(os.path.exists(os.path.join(_ROOT, name)),
                f"{name} يجب أن يبقى في الجذر للـ backward compat")

    def test_v19_2_still_intact(self):
        """v19_2/ لازال موجود (لم نُنقَل)."""
        self.assertTrue(os.path.isdir(os.path.join(_ROOT, "v19_2")))
        self.assertTrue(os.path.exists(os.path.join(_ROOT, "v19_2", "edge_scanner.py")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
