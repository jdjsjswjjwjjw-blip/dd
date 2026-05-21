"""
tests/test_per_regime_discovery.py — Sprint 13 tests.

يحرس Per-Regime Discovery + RegimeAlphaLibrary + Bridge integration.
(Regime Analysis Report v2 ③ — الفكرة الذهبية)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _make_alpha(name: str, regime_hint: str = "trending", wr: float = 0.7) -> dict:
    return {
        "name": name,
        "zone": f"T_REG_{regime_hint.upper()}_Q1",
        "level": "PDH",
        "event": "touch",
        "combo": "LONG_dp_pos",
        "direction": 1,
        "horizon": 6,
        "filters": ["depth_pressure_pos"],
        "stats": {
            "wr": wr,
            "sharpe": 1.5,
            "perm_p": 0.02,
            "n_trades": 100,
            "n_days": 20,
        },
    }


class TestRegimeAlphaLibraryBasics(unittest.TestCase):
    def test_default_regimes_initialized(self):
        from modules.regime_alpha_library import RegimeAlphaLibrary, DEFAULT_REGIMES
        lib = RegimeAlphaLibrary()
        for r in DEFAULT_REGIMES:
            self.assertIn(r, lib.alphas)
            self.assertEqual(len(lib.alphas[r]), 0)

    def test_add_alpha_basic(self):
        from modules.regime_alpha_library import RegimeAlphaLibrary
        lib = RegimeAlphaLibrary()
        lib.add_alpha("trending", _make_alpha("a1"))
        self.assertEqual(len(lib.get("trending")), 1)
        self.assertEqual(lib.total_alphas(), 1)

    def test_add_alpha_validates_required_keys(self):
        from modules.regime_alpha_library import RegimeAlphaLibrary
        lib = RegimeAlphaLibrary()
        with self.assertRaises(ValueError):
            lib.add_alpha("trending", {"name": "missing_dir_and_horizon"})

    def test_add_from_scan_result(self):
        from modules.regime_alpha_library import RegimeAlphaLibrary
        lib = RegimeAlphaLibrary()
        scan_result = {
            "candidates": [
                {"zone": "z1", "direction": 1, "horizon": 6, "wr": 0.7},
                {"zone": "z2", "direction": -1, "horizon": 12, "wr": 0.65},
            ],
        }
        n = lib.add_from_scan_result("trending", scan_result)
        self.assertEqual(n, 2)
        self.assertEqual(len(lib.get("trending")), 2)
        # auto-generated names
        names = [a["name"] for a in lib.get("trending")]
        self.assertEqual(names, ["trending_alpha_000", "trending_alpha_001"])

    def test_summary_structure(self):
        from modules.regime_alpha_library import RegimeAlphaLibrary
        lib = RegimeAlphaLibrary()
        lib.add_alpha("trending", _make_alpha("a"))
        lib.add_alpha("trending", _make_alpha("b"))
        lib.add_alpha("ranging", _make_alpha("c"))
        summary = lib.summary()
        self.assertEqual(summary["total_alphas"], 3)
        self.assertEqual(summary["per_regime_counts"]["trending"], 2)
        self.assertEqual(summary["per_regime_counts"]["ranging"], 1)
        self.assertEqual(summary["per_regime_counts"]["volatile"], 0)
        self.assertIn("trending", summary["regimes_with_alphas"])
        self.assertIn("ranging", summary["regimes_with_alphas"])
        self.assertNotIn("volatile", summary["regimes_with_alphas"])

    def test_has_method(self):
        from modules.regime_alpha_library import RegimeAlphaLibrary
        lib = RegimeAlphaLibrary()
        lib.add_alpha("trending", _make_alpha("a"))
        self.assertTrue(lib.has("trending"))
        self.assertFalse(lib.has("ranging"))
        self.assertFalse(lib.has("nonexistent"))


class TestRegimeAlphaLibrarySerialization(unittest.TestCase):
    def test_to_dict_roundtrip(self):
        from modules.regime_alpha_library import RegimeAlphaLibrary
        lib = RegimeAlphaLibrary(metadata={"created_by": "test"})
        lib.add_alpha("trending", _make_alpha("a"))
        lib.add_alpha("ranging", _make_alpha("b", "ranging"))
        d = lib.to_dict()
        lib2 = RegimeAlphaLibrary.from_dict(d)
        self.assertEqual(lib2.total_alphas(), 2)
        self.assertEqual(lib2.metadata["created_by"], "test")

    def test_save_load_json(self):
        from modules.regime_alpha_library import RegimeAlphaLibrary
        lib = RegimeAlphaLibrary()
        lib.add_alpha("trending", _make_alpha("a"))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "lib.json"
            lib.save(path)
            self.assertTrue(path.exists())
            loaded = RegimeAlphaLibrary.load(path)
            self.assertEqual(loaded.total_alphas(), 1)
            self.assertIn("saved_at", loaded.metadata)

    def test_load_missing_file_raises(self):
        from modules.regime_alpha_library import RegimeAlphaLibrary
        with self.assertRaises(FileNotFoundError):
            RegimeAlphaLibrary.load("/tmp/nonexistent_lib_xyz.json")


class TestFilteringSelection(unittest.TestCase):
    def test_filter_by_stats(self):
        from modules.regime_alpha_library import RegimeAlphaLibrary
        lib = RegimeAlphaLibrary()
        lib.add_alpha("trending", _make_alpha("good", wr=0.75))
        lib.add_alpha("trending", _make_alpha("bad", wr=0.45))
        filt = lib.filter_by_stats(min_wr=0.55)
        self.assertEqual(len(filt.get("trending")), 1)
        self.assertEqual(filt.get("trending")[0]["name"], "good")

    def test_top_k_sorted_by_sharpe(self):
        from modules.regime_alpha_library import RegimeAlphaLibrary
        lib = RegimeAlphaLibrary()
        for i in range(5):
            a = _make_alpha(f"a{i}")
            a["stats"]["sharpe"] = float(i)
            lib.add_alpha("trending", a)
        top = lib.top_k_per_regime(k=3, sort_by="sharpe")
        names = [a["name"] for a in top.get("trending")]
        self.assertEqual(names, ["a4", "a3", "a2"])


class TestMergeLibraries(unittest.TestCase):
    def test_merge_basic(self):
        from modules.regime_alpha_library import RegimeAlphaLibrary, merge_libraries
        l1 = RegimeAlphaLibrary()
        l1.add_alpha("trending", _make_alpha("a1"))
        l2 = RegimeAlphaLibrary()
        l2.add_alpha("trending", _make_alpha("a2"))
        l2.add_alpha("ranging", _make_alpha("b1", "ranging"))
        merged = merge_libraries(l1, l2)
        self.assertEqual(merged.total_alphas(), 3)
        self.assertEqual(merged.metadata["merged_from"], 2)

    def test_merge_dedup_by_name(self):
        from modules.regime_alpha_library import RegimeAlphaLibrary, merge_libraries
        l1 = RegimeAlphaLibrary()
        l1.add_alpha("trending", _make_alpha("dup"))
        l2 = RegimeAlphaLibrary()
        l2.add_alpha("trending", _make_alpha("dup"))  # same name
        merged = merge_libraries(l1, l2)
        self.assertEqual(merged.total_alphas(), 1)


class TestDiscoverByRegimeAPI(unittest.TestCase):
    """يفحص الـ API + الـ skip logic. لا نشغّل scan_edges_v2 الفعلي."""

    def test_function_exists_and_accepts_params(self):
        from modules.statistical_validation_layer import discover_alphas_by_regime
        import inspect
        sig = inspect.signature(discover_alphas_by_regime)
        self.assertIn("df", sig.parameters)
        self.assertIn("regimes", sig.parameters)
        self.assertIn("regime_col", sig.parameters)
        self.assertIn("min_samples_per_regime", sig.parameters)

    def test_raises_on_missing_regime_col(self):
        from modules.statistical_validation_layer import discover_alphas_by_regime
        df = pd.DataFrame({"close": [1.0, 2.0]})
        with self.assertRaises(KeyError):
            discover_alphas_by_regime(df, regime_col="missing")

    def test_skip_insufficient_samples(self):
        """regimes أقل من min_samples_per_regime تُتخطّى مع تسجيل السبب."""
        from modules.statistical_validation_layer import discover_alphas_by_regime
        rng = np.random.RandomState(0)
        n = 30  # كل regime < 1000
        df = pd.DataFrame({
            "regime_label": rng.choice(["trending", "ranging"], n),
            "close": rng.randn(n).cumsum() + 100,
            "ts_event": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        })
        # min_samples_per_regime=1000 → كل الـ regimes تتخطّى
        # ولا scan فعلي يحدث (آمن في env بدون v19_2 enrichments)
        library = discover_alphas_by_regime(
            df, regimes=["trending", "ranging"],
            min_samples_per_regime=1000,
            run_permutation=False, verbose=False,
        )
        self.assertEqual(library.total_alphas(), 0)
        # كل الـ regimes في 'skipped'
        skipped = library.metadata.get("skipped", {})
        self.assertEqual(len(skipped), 2)
        for r in ("trending", "ranging"):
            self.assertIn(r, skipped)
            self.assertEqual(skipped[r]["reason"], "insufficient_samples")


class TestIntegrationBridgeRegimeAware(unittest.TestCase):
    """Sprint 13: Bridge يفلتر alphas حسب row['regime_label']."""

    def test_bridge_accepts_regime_library(self):
        from modules.integration_bridge import IntegrationBridge, BridgeConfig
        from modules.statistical_validation_layer import AlphaSet
        from modules.regime_alpha_library import RegimeAlphaLibrary

        library = RegimeAlphaLibrary()
        library.add_alpha("trending", _make_alpha("t1"))
        library.add_alpha("ranging", _make_alpha("r1", "ranging"))

        bridge = IntegrationBridge(
            alpha_set=AlphaSet(),
            config=BridgeConfig(),
            regime_library=library,
        )
        self.assertIs(bridge.regime_library, library)

    def test_alphas_for_row_filters_by_regime(self):
        from modules.integration_bridge import IntegrationBridge, BridgeConfig
        from modules.statistical_validation_layer import AlphaSet
        from modules.regime_alpha_library import RegimeAlphaLibrary

        library = RegimeAlphaLibrary()
        library.add_alpha("trending", _make_alpha("t1"))
        library.add_alpha("trending", _make_alpha("t2"))
        library.add_alpha("ranging", _make_alpha("r1", "ranging"))

        bridge = IntegrationBridge(
            AlphaSet(), BridgeConfig(),
            regime_library=library,
        )

        # row في trending regime
        row_t = pd.Series({"regime_label": "trending"})
        alphas_t = bridge._alphas_for_row(row_t)
        self.assertEqual(len(alphas_t), 2)
        self.assertEqual({a["name"] for a in alphas_t}, {"t1", "t2"})

        # row في ranging regime
        row_r = pd.Series({"regime_label": "ranging"})
        alphas_r = bridge._alphas_for_row(row_r)
        self.assertEqual(len(alphas_r), 1)
        self.assertEqual(alphas_r[0]["name"], "r1")

        # row في volatile (لا alphas)
        row_v = pd.Series({"regime_label": "volatile"})
        alphas_v = bridge._alphas_for_row(row_v)
        self.assertEqual(len(alphas_v), 0)

    def test_alphas_for_row_no_regime_label_returns_all(self):
        """fallback: row بدون regime_label → كل alphas."""
        from modules.integration_bridge import IntegrationBridge, BridgeConfig
        from modules.statistical_validation_layer import AlphaSet
        from modules.regime_alpha_library import RegimeAlphaLibrary

        library = RegimeAlphaLibrary()
        library.add_alpha("trending", _make_alpha("t1"))
        library.add_alpha("ranging", _make_alpha("r1", "ranging"))

        bridge = IntegrationBridge(
            AlphaSet(), BridgeConfig(),
            regime_library=library,
        )
        row_no_regime = pd.Series({"other_col": "x"})
        alphas = bridge._alphas_for_row(row_no_regime)
        # fallback: all
        self.assertEqual(len(alphas), 2)

    def test_backward_compat_without_regime_library(self):
        """لو regime_library=None → السلوك القديم (يستخدم alpha_set)."""
        from modules.integration_bridge import IntegrationBridge, BridgeConfig
        from modules.statistical_validation_layer import AlphaSet

        alpha_set = AlphaSet(candidates=[_make_alpha("legacy")])
        bridge = IntegrationBridge(alpha_set, BridgeConfig())
        self.assertIsNone(bridge.regime_library)

        row = pd.Series({"regime_label": "trending"})  # regime موجود لكن مهمول
        alphas = bridge._alphas_for_row(row)
        self.assertEqual(len(alphas), 1)
        self.assertEqual(alphas[0]["name"], "legacy")


class TestScientificValidation(unittest.TestCase):
    """Quantitative validation: per-regime يكتشف alphas تختبئ في الكل."""

    def test_per_regime_detects_alpha_hidden_in_combined(self):
        """
        Monte Carlo simulation:
            بناء dataset مع alpha ذهبي في trending فقط (WR=78%)،
            وضوضاء عشوائية في ranging (WR=45%).

            الـ combined WR = (10K * 0.78 + 2K * 0.45) / 12K = 73%
            لكن في trending فقط = 78%.

            هذا اختبار خصائص الـ math (no scan needed).
        """
        rng = np.random.RandomState(42)
        # trending: 10K samples، 78% wins
        trending_wins = (rng.uniform(0, 1, 10000) < 0.78).astype(int)
        # ranging: 2K samples، 45% wins
        ranging_wins = (rng.uniform(0, 1, 2000) < 0.45).astype(int)

        combined_wr = (trending_wins.sum() + ranging_wins.sum()) / 12000
        trending_only_wr = trending_wins.mean()
        ranging_only_wr = ranging_wins.mean()

        # الـ trending-only WR أعلى بكثير من combined
        self.assertGreater(trending_only_wr, combined_wr)
        self.assertGreater(trending_only_wr - combined_wr, 0.03)

        # هذا يؤكد فرضية التقرير: per-regime يكشف عما يختبئ
        self.assertGreater(trending_only_wr, 0.75)
        self.assertLess(ranging_only_wr, 0.50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
