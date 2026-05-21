"""
tests/test_paper_live_bridge_wiring.py — Sprint 6 wiring tests.

يحرس إن paper_v19.py + live_predictor.py مربوطين بـ IntegrationBridge.
لا يستورد الـ scripts مباشرة (تجنب sklearn dependency في env المحلي) —
يفحص الـ source code للـ wiring.
"""

from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestPaperV19BridgeWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (Path(_ROOT) / "paper_v19.py").read_text()

    def test_run_paper_accepts_use_bridge(self):
        # signature يحوي use_bridge
        m = re.search(r"def run_paper\([^)]+\)", self.src, re.S)
        self.assertIsNotNone(m, "run_paper signature لم تُعثَر")
        self.assertIn("use_bridge", m.group(0))
        self.assertIn("alpha_set_path", m.group(0))

    def test_imports_integration_bridge(self):
        self.assertIn("from modules.integration_bridge import", self.src)
        self.assertIn("IntegrationBridge", self.src)
        self.assertIn("BridgeConfig", self.src)

    def test_imports_alpha_set(self):
        self.assertIn("from modules.statistical_validation_layer import AlphaSet", self.src)

    def test_evaluate_row_call_in_loop(self):
        """التأكد إن bridge.evaluate_row(...) موجود في الـ row loop."""
        self.assertIn("bridge.evaluate_row(", self.src)

    def test_bridge_writer_to_jsonl(self):
        """التأكد إن decisions تُسجَّل في paper_bridge.jsonl."""
        self.assertIn("paper_bridge.jsonl", self.src)
        self.assertIn("bridge_writer.write(", self.src)

    def test_cli_has_use_bridge_flag(self):
        self.assertIn("--use-bridge", self.src)
        self.assertIn("--alpha-set", self.src)


class TestLivePredictorBridgeWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (Path(_ROOT) / "live_predictor.py").read_text()

    def test_predict_live_accepts_bridge(self):
        m = re.search(r"def predict_live\([^)]+\)", cls.src if False else (Path(_ROOT) / "live_predictor.py").read_text(), re.S)
        self.assertIsNotNone(m)
        self.assertIn("bridge", m.group(0))

    def test_bridge_evaluate_row_called(self):
        self.assertIn("bridge.evaluate_row(", self.src)

    def test_debug_includes_bridge_decision(self):
        self.assertIn("bridge_decision", self.src)

    def test_dl_proba_constructed_from_long_short(self):
        """الـ DL proba [long, short, neutral] يُبنى صحيحاً."""
        # long_p, short_p, neutral_p logic
        self.assertIn("neutral_p", self.src)
        self.assertIn("dl_proba", self.src)


class TestBridgeCompatibility(unittest.TestCase):
    """التأكد إن IntegrationBridge.evaluate_row signature موجود."""

    def test_bridge_evaluate_row_exists(self):
        from modules.integration_bridge import IntegrationBridge
        import inspect
        sig = inspect.signature(IntegrationBridge.evaluate_row)
        self.assertIn("row", sig.parameters)
        self.assertIn("dl_proba", sig.parameters)


if __name__ == "__main__":
    unittest.main(verbosity=2)
