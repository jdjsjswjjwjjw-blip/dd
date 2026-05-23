"""Tests for causal swing analysis."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestSwingDetection(unittest.TestCase):
    def test_detects_obvious_swings(self):
        from modules.price_cycle import detect_swings, SwingConfig
        # Clear zigzag: up 10, down 10, up 10
        prices = np.concatenate([
            np.linspace(100, 110, 20),
            np.linspace(110, 100, 20),
            np.linspace(100, 112, 20),
        ])
        high = prices + 0.1
        low = prices - 0.1
        close = prices
        config = SwingConfig(reversal_pct=2.0, min_bars_between_swings=2)
        swings, confirmations = detect_swings(high, low, close, config)
        # Should detect at least the peak around index 20 and trough around 40
        self.assertGreaterEqual(len(swings), 1)

    def test_confirmations_are_causal(self):
        """Confirmation index must be >= swing index (no look-ahead)."""
        from modules.price_cycle import detect_swings, SwingConfig
        rng = np.random.RandomState(0)
        prices = 100 + np.cumsum(rng.randn(200) * 0.5)
        config = SwingConfig(reversal_pct=1.5)
        swings, confirmations = detect_swings(
            prices + 0.1, prices - 0.1, prices, config,
        )
        for i, sw in enumerate(swings):
            if i < len(confirmations):
                self.assertGreaterEqual(
                    confirmations[i], sw.bar_index,
                    "Confirmation must come at or after the swing (causal)",
                )

    def test_empty_short_series(self):
        from modules.price_cycle import detect_swings
        swings, conf = detect_swings(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]),
        )
        self.assertEqual(len(swings), 0)


class TestSwingClassification(unittest.TestCase):
    def test_classify_hh_hl(self):
        from modules.price_cycle import SwingPoint, classify_swings, SwingType
        # Uptrend: high, low, higher-high, higher-low
        swings = [
            SwingPoint(0, 100.0, is_high=True),
            SwingPoint(5, 95.0, is_high=False),
            SwingPoint(10, 105.0, is_high=True),   # HH
            SwingPoint(15, 98.0, is_high=False),   # HL
        ]
        classified = classify_swings(swings)
        self.assertEqual(classified[2].swing_type, SwingType.HH)
        self.assertEqual(classified[3].swing_type, SwingType.HL)

    def test_classify_lh_ll(self):
        from modules.price_cycle import SwingPoint, classify_swings, SwingType
        swings = [
            SwingPoint(0, 100.0, is_high=True),
            SwingPoint(5, 95.0, is_high=False),
            SwingPoint(10, 98.0, is_high=True),    # LH
            SwingPoint(15, 90.0, is_high=False),   # LL
        ]
        classified = classify_swings(swings)
        self.assertEqual(classified[2].swing_type, SwingType.LH)
        self.assertEqual(classified[3].swing_type, SwingType.LL)


class TestStructureScore(unittest.TestCase):
    def test_uptrend_positive(self):
        from modules.price_cycle import SwingPoint, classify_swings, swing_structure_score
        swings = classify_swings([
            SwingPoint(0, 100.0, is_high=True),
            SwingPoint(5, 95.0, is_high=False),
            SwingPoint(10, 105.0, is_high=True),
            SwingPoint(15, 98.0, is_high=False),
        ])
        score = swing_structure_score(swings, lookback=4)
        self.assertGreater(score, 0)

    def test_downtrend_negative(self):
        from modules.price_cycle import SwingPoint, classify_swings, swing_structure_score
        swings = classify_swings([
            SwingPoint(0, 100.0, is_high=True),
            SwingPoint(5, 95.0, is_high=False),
            SwingPoint(10, 98.0, is_high=True),
            SwingPoint(15, 90.0, is_high=False),
        ])
        score = swing_structure_score(swings, lookback=4)
        self.assertLess(score, 0)

    def test_empty_zero(self):
        from modules.price_cycle import swing_structure_score
        self.assertEqual(swing_structure_score([], lookback=4), 0.0)


class TestTrendMaturity(unittest.TestCase):
    def test_returns_valid_result(self):
        from modules.price_cycle import (
            detect_swings, classify_swings, estimate_trend_maturity, SwingConfig,
        )
        rng = np.random.RandomState(1)
        close = 100 + np.cumsum(rng.randn(150) * 0.3)
        swings, _ = detect_swings(close + 0.1, close - 0.1, close, SwingConfig(reversal_pct=1.5))
        result = estimate_trend_maturity(close, classify_swings(swings))
        self.assertIn(result.maturity.name, ("YOUNG", "MATURE", "EXHAUSTED"))
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_insufficient_data(self):
        from modules.price_cycle import estimate_trend_maturity
        result = estimate_trend_maturity(np.array([1.0, 2.0]), [])
        self.assertEqual(result.age_bars, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
