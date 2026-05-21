"""
tests/test_quantum_core.py — Sprint 2 quantum_core tests.

يحرس primitives الـ quantum (state, gates, amplifier, measurement, decoherence).
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestStatePreparation(unittest.TestCase):
    def test_normalize_amplitudes_unit_norm(self):
        from modules.quantum_core import normalize_amplitudes
        v = np.array([3.0, 4.0])  # norm = 5
        out = normalize_amplitudes(v)
        self.assertAlmostEqual(float(np.sum(out ** 2)), 1.0, places=10)

    def test_normalize_zero_vector(self):
        from modules.quantum_core import normalize_amplitudes
        out = normalize_amplitudes(np.zeros(5))
        self.assertTrue(np.allclose(out, 0.0))

    def test_normalize_batched(self):
        from modules.quantum_core import normalize_amplitudes
        v = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
        out = normalize_amplitudes(v)
        # row 0: norm 1
        self.assertAlmostEqual(float(np.sum(out[0] ** 2)), 1.0, places=10)
        # row 2: zero stays zero
        self.assertTrue(np.allclose(out[2], 0.0))

    def test_hadamard_uniform(self):
        from modules.quantum_core import hadamard_transform
        out = hadamard_transform(np.array([1.0, 2.0, 3.0, 4.0]))
        # 4 elements → each = 1/2
        self.assertTrue(np.allclose(out, 0.5))
        # ‖ψ‖ = 1
        self.assertAlmostEqual(float(np.sum(out ** 2)), 1.0, places=10)

    def test_quantum_state_class(self):
        from modules.quantum_core import QuantumState
        psi = QuantumState(np.array([0.6, 0.8]))
        self.assertEqual(psi.n, 2)
        p = psi.probabilities()
        self.assertAlmostEqual(float(p[0]), 0.36, places=6)
        self.assertAlmostEqual(float(p[1]), 0.64, places=6)


class TestEntanglementGates(unittest.TestCase):
    def test_cnot_amplifies_when_control_high(self):
        from modules.quantum_core import cnot_gate
        out = cnot_gate(control=np.array([1.0]), target=np.array([0.5]), threshold=0.5, amplification=2.0)
        self.assertAlmostEqual(float(out[0]), 1.0)

    def test_cnot_deflates_when_control_low(self):
        from modules.quantum_core import cnot_gate
        out = cnot_gate(control=np.array([0.1]), target=np.array([0.5]), threshold=0.5, deflation=0.5)
        self.assertAlmostEqual(float(out[0]), 0.25)

    def test_bell_phi_plus_high_correlation(self):
        """|Φ+⟩: كلاهما high أو كلاهما low → output high."""
        from modules.quantum_core import bell_state
        out = bell_state(np.array([1.0, 0.0, 0.5]), np.array([1.0, 0.0, 0.5]))
        # (1,1) → joint=1, anti=0, mixed=0 → 1
        self.assertAlmostEqual(float(out[0]), 1.0)
        # (0,0) → joint=0, anti=1, mixed=0 → 1
        self.assertAlmostEqual(float(out[1]), 1.0)

    def test_bell_anti_correlation(self):
        """(1,0) أو (0,1) → mixed → Φ+ = -1."""
        from modules.quantum_core import bell_state
        out = bell_state(np.array([1.0]), np.array([0.0]))
        self.assertAlmostEqual(float(out[0]), -1.0)

    def test_ghz_all_high(self):
        from modules.quantum_core import ghz_state
        feats = [np.array([1.0]), np.array([1.0]), np.array([1.0])]
        out = ghz_state(feats)
        # all_high=1, all_low=0 → 1
        self.assertAlmostEqual(float(out[0]), 1.0)

    def test_ghz_all_low(self):
        from modules.quantum_core import ghz_state
        feats = [np.array([0.0]), np.array([0.0]), np.array([0.0])]
        out = ghz_state(feats)
        self.assertAlmostEqual(float(out[0]), 1.0)  # all_low = 1

    def test_ghz_mixed(self):
        from modules.quantum_core import ghz_state
        feats = [np.array([1.0]), np.array([0.0]), np.array([1.0])]
        out = ghz_state(feats)
        # all_high = 0, all_low = 0 → 0
        self.assertAlmostEqual(float(out[0]), 0.0)

    def test_toffoli_activation(self):
        from modules.quantum_core import toffoli_gate
        # c1=1, c2=1 → target * (1 + 1) = 2*target
        out = toffoli_gate(np.array([1.0]), np.array([1.0]), np.array([3.0]))
        self.assertAlmostEqual(float(out[0]), 6.0)

    def test_toffoli_no_activation(self):
        from modules.quantum_core import toffoli_gate
        # c1=1, c2=0 → no amplification
        out = toffoli_gate(np.array([1.0]), np.array([0.0]), np.array([3.0]))
        self.assertAlmostEqual(float(out[0]), 3.0)


class TestInterferenceAmplifier(unittest.TestCase):
    def test_optimal_iterations(self):
        from modules.quantum_core import optimal_iterations
        # N=100, M=1 → ≈ (π/4) * 10 = ~7.85 → 8
        n = optimal_iterations(100, 1)
        self.assertIn(n, [7, 8, 9])

    def test_grover_amplifies_good_states(self):
        from modules.quantum_core import grover_iteration
        N = 64
        state = np.full(N, 1.0 / np.sqrt(N))
        mask = np.zeros(N, dtype=bool)
        mask[5] = True  # one "good" state
        out = grover_iteration(state, mask)
        # The good state's amplitude should be much larger
        good_amp_sq = out[5] ** 2
        avg_others_sq = np.mean(out[~mask] ** 2)
        self.assertGreater(good_amp_sq, avg_others_sq * 5,
            f"Grover should amplify: good={good_amp_sq:.4f}, avg_others={avg_others_sq:.6f}")

    def test_grover_preserves_norm(self):
        from modules.quantum_core import grover_iteration
        N = 32
        state = np.full(N, 1.0 / np.sqrt(N))
        mask = np.zeros(N, dtype=bool)
        mask[[3, 7, 15]] = True
        out = grover_iteration(state, mask)
        self.assertAlmostEqual(float(np.sum(out ** 2)), 1.0, places=4)

    def test_grover_alpha_search(self):
        import pandas as pd
        from modules.quantum_core import GroverAlphaSearch
        # Synthetic: 100 alphas, ONLY 5 explicitly marked pass all criteria
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "perm_p": rng.uniform(0.10, 0.20, 100),       # all fail (> 0.05)
            "win_rate": rng.uniform(0.40, 0.50, 100),     # all fail (< 0.55)
            "sharpe": rng.uniform(0.5, 1.0, 100),         # all borderline
            "n_days": rng.randint(1, 5, 100),             # all fail (< 5)
        })
        good_idx = [10, 30, 50, 70, 90]
        df.loc[good_idx, "perm_p"] = 0.01
        df.loc[good_idx, "win_rate"] = 0.65
        df.loc[good_idx, "sharpe"] = 2.0
        df.loc[good_idx, "n_days"] = 20

        search = GroverAlphaSearch()
        top10 = search.select_top_k(df, k=10)
        # All 5 good should be in top-10
        for g in good_idx:
            self.assertIn(g, top10.tolist(),
                f"Grover should rank good idx {g} in top-10, got {top10.tolist()}")


class TestMeasurementEngine(unittest.TestCase):
    def test_measure_greedy(self):
        from modules.quantum_core import measure_greedy
        idx = measure_greedy(np.array([0.1, 0.9, 0.3]))
        self.assertEqual(idx, 1)

    def test_born_probabilities_sum_to_one(self):
        from modules.quantum_core import born_probabilities
        p = born_probabilities(np.array([0.6, 0.8]))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)

    def test_confidence_pure_state(self):
        from modules.quantum_core import confidence_from_state
        # [1, 0, 0]: pure state → conf = 1
        c = confidence_from_state(np.array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(c), 1.0, places=4)

    def test_confidence_uniform_state(self):
        from modules.quantum_core import confidence_from_state
        N = 4
        c = confidence_from_state(np.full(N, 1.0 / np.sqrt(N)))
        self.assertAlmostEqual(float(c), 0.0, places=4)

    def test_fidelity_same_state(self):
        from modules.quantum_core import fidelity
        psi = np.array([0.6, 0.8])
        self.assertAlmostEqual(float(fidelity(psi, psi)), 1.0, places=6)

    def test_fidelity_orthogonal(self):
        from modules.quantum_core import fidelity
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(float(fidelity(a, b)), 0.0, places=6)

    def test_measurement_engine_decide(self):
        from modules.quantum_core import MeasurementEngine
        eng = MeasurementEngine(mode="greedy", confidence_threshold=0.5)
        # High confidence
        idx, conf = eng.decide(np.array([1.0, 0.0, 0.0]))
        self.assertEqual(idx, 0)
        # Low confidence (uniform) → -1 (HOLD)
        idx, conf = eng.decide(np.full(8, 1.0 / np.sqrt(8)))
        self.assertEqual(idx, -1)


class TestDecoherenceHandler(unittest.TestCase):
    def test_low_pass_smooths_noise(self):
        from modules.quantum_core import low_pass_filter
        rng = np.random.RandomState(0)
        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        noisy = signal + rng.normal(0, 0.3, 100)
        smooth = low_pass_filter(noisy, alpha=0.3)
        # smoothed should be closer to signal than noisy
        err_noisy = np.mean((noisy - signal) ** 2)
        err_smooth = np.mean((smooth - signal) ** 2)
        self.assertLess(err_smooth, err_noisy)

    def test_threshold_filter_zeros_small(self):
        from modules.quantum_core import threshold_filter
        arr = np.array([0.001, 0.5, -0.002, 1.0])
        out = threshold_filter(arr, threshold=0.01)
        self.assertEqual(float(out[0]), 0.0)
        self.assertEqual(float(out[2]), 0.0)
        self.assertEqual(float(out[1]), 0.5)
        self.assertEqual(float(out[3]), 1.0)

    def test_renormalize(self):
        from modules.quantum_core import renormalize
        arr = np.array([3.0, 4.0])
        out = renormalize(arr)
        self.assertAlmostEqual(float(np.sum(out ** 2)), 1.0, places=6)

    def test_decoherence_handler_pipeline(self):
        from modules.quantum_core import DecoherenceHandler
        rng = np.random.RandomState(1)
        arr = np.array([0.001, 0.5, 0.6, 0.005, 0.8])
        handler = DecoherenceHandler(lp_alpha=1.0, threshold=0.01)
        out = handler.apply(arr)
        # ‖out‖ ≈ 1
        self.assertAlmostEqual(float(np.sum(out ** 2)), 1.0, places=3)
        # small values are zeroed
        self.assertEqual(float(out[0]), 0.0)
        self.assertEqual(float(out[3]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
