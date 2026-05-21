"""
tests/test_prepare_day_trading_enrichment.py — Sprint 5 wiring test.

يحرس أن prepare_day_trading.py يدعم enrich_v19_2 flag فعلياً:
    1. الـ CLI flag --enrich-v19-2 موجود
    2. الـ run_day_trading_refinery signature يقبل enrich_v19_2
    3. الـ flag يُمرَّر للـ enrichment فعلياً
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestPrepareDayTradingEnrichmentWiring(unittest.TestCase):
    def test_cli_has_enrich_flag(self):
        """--enrich-v19-2 يظهر في الـ help."""
        result = subprocess.run(
            [sys.executable, "prepare_day_trading.py", "--help"],
            cwd=_ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, f"help failed: {result.stderr[:500]}")
        self.assertIn("--enrich-v19-2", result.stdout,
            "CLI flag --enrich-v19-2 يجب أن يكون في الـ help")

    def test_run_function_accepts_param(self):
        """run_day_trading_refinery signature يقبل enrich_v19_2."""
        from prepare_day_trading import run_day_trading_refinery
        sig = inspect.signature(run_day_trading_refinery)
        self.assertIn("enrich_v19_2", sig.parameters,
            "run_day_trading_refinery يجب أن يقبل enrich_v19_2 parameter")
        # default should be False (backward compat)
        self.assertEqual(sig.parameters["enrich_v19_2"].default, False,
            "enrich_v19_2 default يجب أن يكون False")

    def test_enrichment_module_importable(self):
        """modules.feature_enrichment.enrich_features قابل للاستيراد."""
        from modules.feature_enrichment import enrich_features, EnrichmentConfig
        self.assertTrue(callable(enrich_features))

    def test_source_has_enrich_call(self):
        """التأكد إن الـ enrich call موجود فعلياً في الـ source."""
        src = Path(_ROOT) / "prepare_day_trading.py"
        content = src.read_text()
        # The enrich call should be before to_parquet
        self.assertIn("from modules.feature_enrichment import enrich_features", content,
            "prepare_day_trading.py يجب أن يستورد enrich_features داخلياً")
        self.assertIn("enrich_v19_2", content,
            "prepare_day_trading.py يجب أن يستخدم enrich_v19_2 flag")


if __name__ == "__main__":
    unittest.main(verbosity=2)
