"""tests/test_integration_bridge.py — Phase 6 tests."""
from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from modules.integration_bridge import (
    Action,
    BridgeConfig,
    IntegrationBridge,
    TradingDecision,
)
from modules.statistical_validation_layer import AlphaSet


def _make_row(**kwargs) -> pd.Series:
    base = {
        "zone_full": "asia_q1",
        "PDH_event": "touch",
        "sim_depth_pressure": 0.30,
        "sim_wall_consumed": 0.10,
        "sim_informed_prob": 0.60,
    }
    base.update(kwargs)
    return pd.Series(base)


def _make_alpha(
    zone="asia_q1", level="PDH", event="touch",
    combo="LONG_dp_pos", direction=1, filters=None,
):
    return {
        "zone": zone, "level": level, "event": event,
        "combo": combo, "direction": direction,
        "filters": filters or ["depth_pressure_pos"],
        "horizon": 6,
    }


class TestRowMatching(unittest.TestCase):
    def test_match_long_alpha(self):
        alpha = _make_alpha()
        bridge = IntegrationBridge(AlphaSet(candidates=[alpha]))
        row = _make_row()
        self.assertTrue(bridge._row_matches_alpha(row, alpha))

    def test_no_match_wrong_zone(self):
        alpha = _make_alpha(zone="asia_q1")
        bridge = IntegrationBridge(AlphaSet(candidates=[alpha]))
        row = _make_row(zone_full="ny_q1")
        self.assertFalse(bridge._row_matches_alpha(row, alpha))

    def test_no_match_wrong_event(self):
        alpha = _make_alpha(event="break")
        bridge = IntegrationBridge(AlphaSet(candidates=[alpha]))
        row = _make_row()  # PDH_event="touch"
        self.assertFalse(bridge._row_matches_alpha(row, alpha))

    def test_no_match_filter_fails(self):
        # filter dp_pos يطلب > 0.20؛ السطر فيه 0.10
        alpha = _make_alpha()
        bridge = IntegrationBridge(AlphaSet(candidates=[alpha]))
        row = _make_row(sim_depth_pressure=0.10)
        self.assertFalse(bridge._row_matches_alpha(row, alpha))


class TestEvaluateRow(unittest.TestCase):
    def test_hold_when_no_match(self):
        bridge = IntegrationBridge(AlphaSet(candidates=[]))
        row = _make_row()
        d = bridge.evaluate_row(row, dl_proba={"long": 0.7, "short": 0.2, "neutral": 0.1})
        self.assertEqual(d.action, Action.HOLD)
        self.assertEqual(d.reason, "no alpha match")

    def test_open_long_when_alpha_match_and_dl_confirm(self):
        alpha = _make_alpha(direction=1)
        bridge = IntegrationBridge(
            AlphaSet(candidates=[alpha]),
            config=BridgeConfig(dl_confirm_threshold=0.55),
        )
        row = _make_row()
        d = bridge.evaluate_row(
            row, dl_proba={"long": 0.70, "short": 0.20, "neutral": 0.10}
        )
        self.assertEqual(d.action, Action.OPEN_LONG)
        self.assertEqual(d.matched_alpha, "LONG_dp_pos")
        self.assertAlmostEqual(d.confidence, 0.70)

    def test_open_short(self):
        alpha = _make_alpha(
            zone="ny_q1", combo="SHORT_dp_neg", direction=-1,
            filters=["depth_pressure_neg"],
        )
        bridge = IntegrationBridge(AlphaSet(candidates=[alpha]))
        row = _make_row(zone_full="ny_q1", sim_depth_pressure=-0.30)
        d = bridge.evaluate_row(
            row, dl_proba={"long": 0.10, "short": 0.80, "neutral": 0.10}
        )
        self.assertEqual(d.action, Action.OPEN_SHORT)
        self.assertEqual(d.confidence, 0.80)

    def test_hold_when_alpha_match_but_dl_low(self):
        alpha = _make_alpha(direction=1)
        bridge = IntegrationBridge(
            AlphaSet(candidates=[alpha]),
            config=BridgeConfig(dl_confirm_threshold=0.55),
        )
        row = _make_row()
        d = bridge.evaluate_row(
            row, dl_proba={"long": 0.40, "short": 0.30, "neutral": 0.30}
        )
        self.assertEqual(d.action, Action.HOLD)
        self.assertIn("لم يثبّت", d.reason)


class TestWallExit(unittest.TestCase):
    def test_close_on_high_wall_consumed(self):
        bridge = IntegrationBridge(
            AlphaSet(candidates=[]),
            config=BridgeConfig(wall_consumed_exit=0.70),
        )
        row = _make_row(sim_wall_consumed=0.85)
        d = bridge.evaluate_row(
            row,
            dl_proba={"long": 0.5, "short": 0.5, "neutral": 0.0},
            in_position=Action.OPEN_LONG,
        )
        self.assertEqual(d.action, Action.CLOSE)
        self.assertIn("wall_consumed", d.reason)

    def test_no_close_when_flat(self):
        bridge = IntegrationBridge(AlphaSet(candidates=[]))
        row = _make_row(sim_wall_consumed=0.85)
        d = bridge.evaluate_row(
            row,
            dl_proba={"long": 0.5, "short": 0.5, "neutral": 0.0},
            in_position=None,
        )
        self.assertNotEqual(d.action, Action.CLOSE)


class TestEvaluateBatch(unittest.TestCase):
    def test_batch_decisions_length(self):
        n = 10
        df = pd.DataFrame({
            "zone_full": ["asia_q1"] * n,
            "PDH_event": ["touch"] * n,
            "sim_depth_pressure": np.full(n, 0.30),
            "sim_wall_consumed": np.full(n, 0.10),
        })
        proba = np.array([[0.70, 0.20, 0.10]] * n)
        alpha = _make_alpha()
        bridge = IntegrationBridge(AlphaSet(candidates=[alpha]))
        decisions = bridge.evaluate_batch(df, proba)
        self.assertEqual(len(decisions), n)
        # كلها OPEN_LONG
        for d in decisions:
            self.assertEqual(d.action, Action.OPEN_LONG)

    def test_batch_shape_mismatch_raises(self):
        df = pd.DataFrame({"zone_full": ["asia_q1"] * 5})
        proba = np.zeros((3, 3))  # wrong length
        bridge = IntegrationBridge(AlphaSet(candidates=[]))
        with self.assertRaises(ValueError):
            bridge.evaluate_batch(df, proba)


class TestTradingDecisionSerialization(unittest.TestCase):
    def test_to_dict(self):
        d = TradingDecision(
            action=Action.OPEN_LONG,
            reason="test",
            confidence=0.7,
            matched_alpha="LONG_dp_pos",
        )
        dd = d.to_dict()
        self.assertEqual(dd["action"], "OPEN_LONG")
        self.assertEqual(dd["confidence"], 0.7)
        self.assertEqual(dd["matched_alpha"], "LONG_dp_pos")


if __name__ == "__main__":
    unittest.main(verbosity=2)
