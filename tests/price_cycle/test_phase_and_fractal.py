"""Tests for Wyckoff phase classifier + fractal features."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestPhaseClassifier(unittest.TestCase):
    def test_markup_detected_in_uptrend(self):
        from modules.price_cycle import classify_phase_sequence
        # Strong uptrend
        n = 200
        close = np.linspace(100, 150, n)
        high = close + 0.5
        low = close - 0.5
        volume = np.full(n, 1000.0)
        assessments = classify_phase_sequence(high, low, close, volume)
        # In a clear uptrend, later bars should mostly be MARKUP
        from modules.price_cycle import WyckoffPhase
        late_phases = [a.phase for a in assessments[-50:]]
        markup_count = sum(1 for p in late_phases if p == WyckoffPhase.MARKUP)
        self.assertGreater(markup_count, 10, "Uptrend should yield MARKUP phases")

    def test_markdown_detected_in_downtrend(self):
        from modules.price_cycle import classify_phase_sequence, WyckoffPhase
        n = 200
        close = np.linspace(150, 100, n)
        assessments = classify_phase_sequence(
            close + 0.5, close - 0.5, close, np.full(n, 1000.0),
        )
        late = [a.phase for a in assessments[-50:]]
        markdown_count = sum(1 for p in late if p == WyckoffPhase.MARKDOWN)
        self.assertGreater(markdown_count, 10)

    def test_phase_probs_sum_to_one(self):
        from modules.price_cycle import classify_phase_sequence
        n = 100
        rng = np.random.RandomState(0)
        close = 100 + np.cumsum(rng.randn(n) * 0.3)
        assessments = classify_phase_sequence(
            close + 0.1, close - 0.1, close, rng.uniform(500, 1500, n),
        )
        for a in assessments:
            self.assertAlmostEqual(float(a.phase_probs.sum()), 1.0, places=5)

    def test_cycle_position_in_range(self):
        from modules.price_cycle import classify_phase_sequence
        n = 100
        rng = np.random.RandomState(2)
        close = 100 + np.cumsum(rng.randn(n) * 0.3)
        assessments = classify_phase_sequence(
            close + 0.1, close - 0.1, close, rng.uniform(500, 1500, n),
        )
        for a in assessments:
            self.assertGreaterEqual(a.cycle_position, 0.0)
            self.assertLessEqual(a.cycle_position, 1.0)


class TestHurstExponent(unittest.TestCase):
    def test_trending_series_high_hurst(self):
        from modules.price_cycle import hurst_exponent
        # Strong trend → H > 0.5
        trend = np.cumsum(np.full(500, 0.1) + np.random.RandomState(0).randn(500) * 0.01)
        H = hurst_exponent(trend, max_lag=20)
        self.assertGreater(H, 0.5)

    def test_random_walk_near_half(self):
        from modules.price_cycle import hurst_exponent
        rng = np.random.RandomState(0)
        rw = np.cumsum(rng.randn(1000))
        H = hurst_exponent(rw, max_lag=20)
        # Random walk → H ≈ 0.5 (with tolerance)
        self.assertGreater(H, 0.3)
        self.assertLess(H, 0.7)

    def test_mean_reverting_low_hurst(self):
        from modules.price_cycle import hurst_exponent
        # Mean-reverting (anti-persistent): oscillating series
        rng = np.random.RandomState(0)
        n = 1000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = -0.5 * x[i-1] + rng.randn()  # AR(1) with negative coef
        H = hurst_exponent(x, max_lag=20)
        self.assertLess(H, 0.5)

    def test_insufficient_data_returns_half(self):
        from modules.price_cycle import hurst_exponent
        H = hurst_exponent(np.array([1.0, 2.0, 3.0]), max_lag=20)
        self.assertEqual(H, 0.5)


class TestMultiTimeframeAlignment(unittest.TestCase):
    def test_aligned_uptrend(self):
        from modules.price_cycle import multi_timeframe_alignment
        close = np.linspace(100, 120, 200)
        result = multi_timeframe_alignment(close, (1, 4, 16))
        # All timeframes bullish → alignment near +1
        self.assertGreater(result["alignment_score"], 0.5)

    def test_aligned_downtrend(self):
        from modules.price_cycle import multi_timeframe_alignment
        close = np.linspace(120, 100, 200)
        result = multi_timeframe_alignment(close, (1, 4, 16))
        self.assertLess(result["alignment_score"], -0.5)


class TestFractalFeatures(unittest.TestCase):
    def test_compute_all_features(self):
        from modules.price_cycle import compute_fractal_features
        rng = np.random.RandomState(0)
        n = 300
        close = 100 + np.cumsum(rng.randn(n) * 0.2)
        feats = compute_fractal_features(
            close, close + 0.1, close - 0.1, close, rng.uniform(500, 1500, n),
        )
        self.assertIn("hurst", feats)
        self.assertIn("fractal_dim", feats)
        self.assertIn("mtf_alignment", feats)
        for key, arr in feats.items():
            self.assertEqual(len(arr), n)
            self.assertTrue(np.all(np.isfinite(arr)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
