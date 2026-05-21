"""
tests/test_regime_probabilities.py — Sprint 14 tests.

يحرس Soft Regime Probabilities (Regime Analysis Report v2 ④):
    P1: Normalization (Σ P = 1)
    P2: Non-negativity (P ≥ 0)
    P3: Convergence to hard label (max P → 1 لو signal واضح)
    P4: Convergence to uniform (max P → 1/k لو signal فوضوي)
    P5: Temporal smoothness (|P_t - P_{t-1}| ≤ (1-α))
    P6: Monotonicity TP (TP_w monotone في P(trending))
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestSoftmax(unittest.TestCase):
    def test_softmax_normalizes(self):
        from modules.regime_probabilities import softmax
        out = softmax(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(out.sum()), 1.0, places=10)

    def test_softmax_temperature_sharpening(self):
        from modules.regime_probabilities import softmax
        scores = np.array([1.0, 2.0, 3.0])
        smooth = softmax(scores, temperature=2.0)
        sharp = softmax(scores, temperature=0.5)
        # sharp يقترب من one-hot (max أكبر)
        self.assertGreater(sharp.max(), smooth.max())

    def test_softmax_invalid_temperature(self):
        from modules.regime_probabilities import softmax
        with self.assertRaises(ValueError):
            softmax(np.array([1.0]), temperature=0.0)

    def test_softmax_2d_batch(self):
        from modules.regime_probabilities import softmax
        scores = np.array([[1.0, 2.0], [3.0, 1.0]])
        out = softmax(scores)
        self.assertEqual(out.shape, (2, 2))
        self.assertTrue(np.allclose(out.sum(axis=-1), 1.0))


class TestFromHardLabels(unittest.TestCase):
    def test_full_confidence_one_hot(self):
        from modules.regime_probabilities import from_hard_labels
        labels = np.array(["trending", "ranging", "volatile"])
        out = from_hard_labels(labels, confidence=1.0)
        self.assertEqual(out.shape, (3, 3))
        # diagonal = 1
        self.assertAlmostEqual(out[0, 0], 1.0)
        self.assertAlmostEqual(out[1, 1], 1.0)
        self.assertAlmostEqual(out[2, 2], 1.0)

    def test_partial_confidence_distributes(self):
        from modules.regime_probabilities import from_hard_labels
        out = from_hard_labels(["trending"], confidence=0.7)
        # trending = 0.7, others = (1-0.7)/2 = 0.15 each
        self.assertAlmostEqual(out[0, 0], 0.7, places=6)
        self.assertAlmostEqual(out[0, 1], 0.15, places=6)
        self.assertAlmostEqual(out[0, 2], 0.15, places=6)
        # Σ = 1
        self.assertAlmostEqual(out.sum(), 1.0, places=6)

    def test_unknown_label_uniform(self):
        from modules.regime_probabilities import from_hard_labels
        out = from_hard_labels(["unknown_regime"], confidence=0.9)
        # uniform fallback
        self.assertTrue(np.allclose(out[0], 1/3))


class TestFromIndicators(unittest.TestCase):
    def test_high_adx_implies_trending(self):
        from modules.regime_probabilities import from_indicators
        # ADX=50 (high) → trending dominates
        adx = np.array([50.0])
        atr_z = np.array([0.0])
        out = from_indicators(adx, atr_z)
        # P(trending) = max
        self.assertEqual(np.argmax(out[0]), 0)
        self.assertGreater(out[0, 0], 0.5)

    def test_low_adx_implies_ranging(self):
        from modules.regime_probabilities import from_indicators
        # ADX=10 (low) + ATR_z normal → ranging dominates
        adx = np.array([10.0])
        atr_z = np.array([0.0])
        out = from_indicators(adx, atr_z)
        # P(ranging) = max
        self.assertEqual(np.argmax(out[0]), 1)

    def test_high_atr_implies_volatile(self):
        from modules.regime_probabilities import from_indicators
        # ATR_z=5 (extreme) → volatile high
        out = from_indicators(np.array([20.0]), np.array([5.0]))
        # volatile = nonzero (not necessarily max لو ADX قريب من threshold)
        self.assertGreater(out[0, 2], 0.4)

    def test_property_P1_normalization(self):
        """P1: each row sums to 1."""
        from modules.regime_probabilities import from_indicators
        rng = np.random.RandomState(0)
        n = 1000
        adx = rng.uniform(0, 80, n)
        atr_z = rng.uniform(-3, 5, n)
        out = from_indicators(adx, atr_z)
        sums = out.sum(axis=-1)
        self.assertTrue(np.allclose(sums, 1.0, atol=1e-6))

    def test_property_P2_non_negativity(self):
        from modules.regime_probabilities import from_indicators
        rng = np.random.RandomState(7)
        n = 500
        adx = rng.uniform(0, 80, n)
        atr_z = rng.uniform(-3, 5, n)
        out = from_indicators(adx, atr_z)
        self.assertTrue((out >= 0).all())


class TestSmoothProbabilities(unittest.TestCase):
    def test_alpha_one_no_smoothing(self):
        from modules.regime_probabilities import smooth_probabilities
        probs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        out = smooth_probabilities(probs, alpha=1.0)
        # α=1 → no smoothing
        np.testing.assert_array_almost_equal(out, probs)

    def test_alpha_zero_pure_history(self):
        from modules.regime_probabilities import smooth_probabilities
        # we don't allow alpha=0 (only 0 < alpha <= 1)
        with self.assertRaises(ValueError):
            smooth_probabilities(np.array([[1.0, 0.0]]), alpha=0.0)

    def test_smoothing_reduces_jumps(self):
        from modules.regime_probabilities import smooth_probabilities, transition_smoothness
        # discrete jumps
        probs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        raw_smoothness = transition_smoothness(probs)
        smoothed = smooth_probabilities(probs, alpha=0.3)
        smooth_smoothness = transition_smoothness(smoothed)
        # smoothing must reduce L1 between consecutive rows
        self.assertLess(smooth_smoothness, raw_smoothness)

    def test_property_P5_smoothness_bound(self):
        """P5: |P_t - P_{t-1}|_∞ ≤ 2α (loose bound)."""
        from modules.regime_probabilities import smooth_probabilities
        rng = np.random.RandomState(1)
        # random probability matrix
        raw = rng.dirichlet([1, 1, 1], size=100)
        alpha = 0.2
        out = smooth_probabilities(raw, alpha=alpha)
        diffs = np.abs(out[1:] - out[:-1])
        max_per_step = diffs.max(axis=-1)
        # loose bound: 2α (since L_inf diff per step at most)
        self.assertTrue((max_per_step <= 2 * alpha + 1e-6).all(),
            f"smoothness bound violated: max={max_per_step.max()}, bound={2*alpha}")

    def test_renormalization_after_smoothing(self):
        from modules.regime_probabilities import smooth_probabilities
        rng = np.random.RandomState(0)
        raw = rng.dirichlet([1, 1, 1], size=50)
        out = smooth_probabilities(raw, alpha=0.5)
        sums = out.sum(axis=-1)
        self.assertTrue(np.allclose(sums, 1.0, atol=1e-6))


class TestWeightedTPSL(unittest.TestCase):
    def test_weighted_tp_with_dict_input(self):
        from modules.regime_probabilities import weighted_tp_sl
        probs = {"trending": 0.65, "ranging": 0.30, "volatile": 0.05}
        tp, sl = weighted_tp_sl(probs)
        # TP = 0.65*2.0 + 0.30*1.1 + 0.05*1.8 = 1.30 + 0.33 + 0.09 = 1.72
        self.assertAlmostEqual(tp, 1.72, places=4)
        # SL = 0.65*1.0 + 0.30*1.0 + 0.05*1.2 = 1.01
        self.assertAlmostEqual(sl, 1.01, places=4)

    def test_weighted_tp_with_array_input(self):
        from modules.regime_probabilities import weighted_tp_sl
        # (trending, ranging, volatile) order
        probs = np.array([0.65, 0.30, 0.05])
        tp, sl = weighted_tp_sl(probs)
        self.assertAlmostEqual(tp, 1.72, places=4)
        self.assertAlmostEqual(sl, 1.01, places=4)

    def test_property_P6_monotonicity_in_trending(self):
        """TP_w must be monotone in P(trending) keeping others proportional."""
        from modules.regime_probabilities import weighted_tp_sl
        tps = []
        for p_trend in np.linspace(0.0, 1.0, 20):
            remaining = (1 - p_trend) / 2
            probs = np.array([p_trend, remaining, remaining])
            tp, _ = weighted_tp_sl(probs)
            tps.append(tp)
        # TP increases with P(trending) (because tp_trending=2 > tp_ranging=1.1 and tp_volatile=1.8)
        diffs = np.diff(tps)
        self.assertTrue((diffs >= -1e-9).all(),
            f"TP_w must be monotonically increasing in P(trending), got diffs={diffs}")

    def test_batch_consistent_with_scalar(self):
        from modules.regime_probabilities import weighted_tp_sl, weighted_tp_sl_batch
        rng = np.random.RandomState(0)
        probs = rng.dirichlet([1, 1, 1], size=10)
        tp_batch, sl_batch = weighted_tp_sl_batch(probs)
        for i in range(10):
            tp_i, sl_i = weighted_tp_sl(probs[i])
            self.assertAlmostEqual(tp_batch[i], tp_i, places=6)
            self.assertAlmostEqual(sl_batch[i], sl_i, places=6)


class TestWeightedHorizon(unittest.TestCase):
    def test_weighted_horizon_dict(self):
        from modules.regime_probabilities import weighted_horizon
        probs = {"trending": 1.0, "ranging": 0.0, "volatile": 0.0}
        h = weighted_horizon(probs)
        self.assertEqual(h, 12)

    def test_weighted_horizon_array(self):
        from modules.regime_probabilities import weighted_horizon
        # equally mixed: 12*0.5 + 6*0.5 = 9
        h = weighted_horizon(np.array([0.5, 0.5, 0.0]))
        self.assertEqual(h, 9)


class TestEntropy(unittest.TestCase):
    def test_entropy_one_hot_is_zero(self):
        from modules.regime_probabilities import entropy
        probs = np.array([[1.0, 0.0, 0.0]])
        self.assertAlmostEqual(float(entropy(probs)[0]), 0.0, places=6)

    def test_entropy_uniform_is_log_k(self):
        from modules.regime_probabilities import entropy
        probs = np.full((1, 3), 1/3)
        expected = np.log(3)
        self.assertAlmostEqual(float(entropy(probs)[0]), expected, places=6)


class TestTransitionSmoothness(unittest.TestCase):
    def test_no_transitions_zero(self):
        from modules.regime_probabilities import transition_smoothness
        probs = np.array([[1.0, 0.0, 0.0]] * 5)
        self.assertEqual(transition_smoothness(probs), 0.0)

    def test_full_flip_high(self):
        from modules.regime_probabilities import transition_smoothness
        probs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        s = transition_smoothness(probs)
        # each step changes L1 = 2 (one full flip)
        self.assertAlmostEqual(s, 2.0, places=6)


class TestValidation(unittest.TestCase):
    def test_validates_correct_probs(self):
        from modules.regime_probabilities import validate_probability_matrix
        probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
        result = validate_probability_matrix(probs)
        self.assertTrue(result["valid"])

    def test_detects_non_normalized(self):
        from modules.regime_probabilities import validate_probability_matrix
        probs = np.array([[0.5, 0.3, 0.5]])  # sums to 1.3
        result = validate_probability_matrix(probs)
        self.assertFalse(result["valid"])

    def test_detects_negatives(self):
        from modules.regime_probabilities import validate_probability_matrix
        probs = np.array([[-0.1, 0.5, 0.6]])  # sums to 1 but has -0.1
        result = validate_probability_matrix(probs)
        self.assertTrue(result["has_negatives"])
        self.assertFalse(result["valid"])


class TestComputeSoftRegimePipeline(unittest.TestCase):
    """High-level pipeline integration."""

    def test_from_labels_method(self):
        from modules.regime_probabilities import compute_soft_regime_pipeline
        df = pd.DataFrame({
            "regime_label": ["trending"] * 10 + ["ranging"] * 10,
        })
        out = compute_soft_regime_pipeline(df, method="from_labels")
        self.assertEqual(out.probs.shape, (20, 3))
        self.assertEqual(len(out.tp_weighted), 20)
        self.assertGreater(out.mean_entropy, 0.0)
        # validation passes
        self.assertTrue(out.diagnostics["validation"]["valid"])

    def test_from_indicators_method(self):
        from modules.regime_probabilities import compute_soft_regime_pipeline
        rng = np.random.RandomState(0)
        df = pd.DataFrame({
            "adx_14": rng.uniform(10, 60, 50),
            "atr_z": rng.uniform(-2, 3, 50),
        })
        out = compute_soft_regime_pipeline(df, method="from_indicators")
        self.assertEqual(out.probs.shape, (50, 3))
        self.assertTrue(out.diagnostics["validation"]["valid"])

    def test_smoothing_reduces_jumpiness(self):
        from modules.regime_probabilities import compute_soft_regime_pipeline
        # alternating labels
        df = pd.DataFrame({
            "regime_label": ["trending", "ranging", "volatile"] * 20,
        })
        out_strong = compute_soft_regime_pipeline(df, smoothing_alpha=0.1)
        out_weak = compute_soft_regime_pipeline(df, smoothing_alpha=0.9)
        # strong smoothing → less jumpiness
        self.assertLess(out_strong.smoothness, out_weak.smoothness)

    def test_raises_on_missing_column(self):
        from modules.regime_probabilities import compute_soft_regime_pipeline
        df = pd.DataFrame({"other": [1, 2]})
        with self.assertRaises(KeyError):
            compute_soft_regime_pipeline(df, method="from_labels")


class TestScientificScenarios(unittest.TestCase):
    """Validate the TP smoothness improvement claimed in the report."""

    def test_smooth_transition_TP_no_sudden_jump(self):
        """Scenario from التقرير: TP linearly interpolates during transition."""
        from modules.regime_probabilities import smooth_probabilities, weighted_tp_sl_batch

        # transitions trending → ranging gradually
        n = 50
        # linear ramp from trending=1 to trending=0
        ramps = np.linspace(1.0, 0.0, n)
        raw = np.zeros((n, 3))
        raw[:, 0] = ramps           # trending decreasing
        raw[:, 1] = 1.0 - ramps     # ranging increasing
        raw[:, 2] = 0.0             # volatile = 0

        smooth = smooth_probabilities(raw, alpha=0.5)
        tp_w, _ = weighted_tp_sl_batch(smooth)

        # TP must be monotonically non-increasing (decay)
        diffs = np.diff(tp_w)
        # allow small numerical noise
        self.assertTrue((diffs <= 1e-6).all(),
            f"TP should decay during trending→ranging: diffs[max]={diffs.max()}")

        # TP at start ≈ 2.0 (pure trending)
        # TP at end ≈ 1.1 (pure ranging)
        # mid ≈ 1.55
        self.assertGreater(tp_w[0], 1.7)
        self.assertLess(tp_w[-1], 1.3)
        self.assertAlmostEqual(tp_w[n // 2], 1.55, delta=0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
