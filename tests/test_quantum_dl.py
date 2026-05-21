"""
tests/test_quantum_dl.py — Sprint 4 quantum_dl tests.

يحرس QNN circuit + QLSTM + VQE portfolio + quantum loss functions.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestQNNCircuit(unittest.TestCase):
    def test_rotation_gates_shape(self):
        from modules.quantum_dl import rx_gate, ry_gate, rz_gate
        x = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertEqual(rx_gate(x, 0.5).shape, (4,))
        self.assertEqual(ry_gate(x, 0.5).shape, (4,))
        self.assertEqual(rz_gate(x, 0.5).shape, (4,))

    def test_rotation_zero_theta_identity(self):
        """θ=0 → ~identity (مع rounding)."""
        from modules.quantum_dl import rx_gate
        x = np.array([0.5, 0.5, 0.5, 0.5])
        out = rx_gate(x, 0.0)
        # cos(0) = 1, sin(0) = 0 → output ≈ x
        self.assertTrue(np.allclose(out, x))

    def test_qnn_layer_normalizes(self):
        from modules.quantum_dl import QNNLayer
        layer = QNNLayer(n_qubits=4)
        state = np.array([0.5, 0.5, 0.5, 0.5])
        out = layer.forward(state)
        # ‖out‖ should be ≈ 1
        self.assertAlmostEqual(float(np.sum(out ** 2)), 1.0, places=4)

    def test_qnn_circuit_forward(self):
        from modules.quantum_dl import QNNCircuit
        circuit = QNNCircuit(n_qubits=4, n_layers=3)
        x = np.random.RandomState(0).randn(4)
        out = circuit.forward(x)
        self.assertEqual(out.shape, (4,))
        # Normalized
        self.assertAlmostEqual(float(np.sum(out ** 2)), 1.0, places=4)

    def test_qnn_n_parameters(self):
        from modules.quantum_dl import QNNCircuit
        circuit = QNNCircuit(n_qubits=4, n_layers=3)
        # 4 params per layer × 3 layers = 12
        self.assertEqual(circuit.n_parameters, 12)

    def test_qnn_get_set_parameters(self):
        from modules.quantum_dl import QNNCircuit
        circuit = QNNCircuit(n_qubits=4, n_layers=2)
        old_params = circuit.get_parameters().copy()
        new_params = old_params + 0.1
        circuit.set_parameters(new_params)
        self.assertTrue(np.allclose(circuit.get_parameters(), new_params))

    def test_qnn_input_padding(self):
        """input أقصر من n_qubits يُحشى بصفر."""
        from modules.quantum_dl import QNNCircuit
        circuit = QNNCircuit(n_qubits=4, n_layers=1)
        x = np.array([1.0, 0.5])  # only 2
        out = circuit.forward(x)
        self.assertEqual(out.shape, (4,))


class TestQLSTM(unittest.TestCase):
    def test_qlstm_cell_step(self):
        from modules.quantum_dl import QLSTMCell
        cell = QLSTMCell(input_dim=3, hidden_dim=4)
        x = np.array([0.5, -0.2, 0.1])
        h0 = np.zeros(4)
        c0 = np.zeros(4)
        h, c = cell.step(x, h0, c0)
        self.assertEqual(h.shape, (4,))
        self.assertEqual(c.shape, (4,))
        # h finite
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertTrue(np.all(np.isfinite(c)))

    def test_qlstm_forward_sequence(self):
        from modules.quantum_dl import QLSTM
        rnn = QLSTM(input_dim=3, hidden_dim=4, n_layers=1, n_qnn_layers=2)
        T = 10
        seq = np.random.RandomState(0).randn(T, 3)
        outputs, final_states = rnn.forward(seq)
        self.assertEqual(outputs.shape, (T, 4))
        self.assertEqual(len(final_states), 1)
        self.assertEqual(final_states[0][0].shape, (4,))

    def test_qlstm_empty_sequence(self):
        from modules.quantum_dl import QLSTM
        rnn = QLSTM(input_dim=3, hidden_dim=4)
        outputs, states = rnn.forward(np.zeros((0, 3)))
        self.assertEqual(outputs.shape, (0, 4))
        self.assertEqual(len(states), 0)

    def test_qlstm_multi_layer(self):
        from modules.quantum_dl import QLSTM
        rnn = QLSTM(input_dim=3, hidden_dim=4, n_layers=2)
        seq = np.random.RandomState(0).randn(5, 3)
        outputs, states = rnn.forward(seq)
        self.assertEqual(outputs.shape, (5, 4))
        self.assertEqual(len(states), 2)


class TestVQEPortfolio(unittest.TestCase):
    def test_build_hamiltonian(self):
        from modules.quantum_dl import build_portfolio_hamiltonian
        mu = np.array([0.1, 0.2, 0.15])
        sigma = np.array([
            [0.01, 0.001, 0.002],
            [0.001, 0.015, 0.001],
            [0.002, 0.001, 0.012],
        ])
        H, info = build_portfolio_hamiltonian(mu, sigma, risk_aversion=1.0)
        self.assertEqual(H.shape, (3, 3))
        self.assertEqual(info["n_assets"], 3)
        # Symmetric
        self.assertTrue(np.allclose(H, H.T))

    def test_expectation_via_state(self):
        from modules.quantum_dl import expectation_via_state
        H = np.diag([1.0, 2.0, 3.0])
        state = np.array([1.0, 0.0, 0.0])
        # ⟨ψ|H|ψ⟩ = 1.0 for state [1,0,0]
        self.assertAlmostEqual(expectation_via_state(state, H), 1.0, places=6)

    def test_vqe_optimize_smoke(self):
        from modules.quantum_dl import vqe_portfolio_optimize, VQEPortfolioConfig
        rng = np.random.RandomState(0)
        N = 4
        mu = rng.uniform(0.001, 0.01, N)
        # Make cov well-conditioned
        A = rng.randn(N, N)
        sigma = (A @ A.T) * 0.01 + np.eye(N) * 0.001
        result = vqe_portfolio_optimize(
            mu, sigma, VQEPortfolioConfig(n_qubits=4, n_layers=2, max_iter=5),
        )
        self.assertEqual(result["weights"].shape, (N,))
        # Weights sum to 1
        self.assertAlmostEqual(float(np.sum(result["weights"])), 1.0, places=4)
        # All non-negative (from |α|²)
        self.assertTrue(np.all(result["weights"] >= 0))


class TestQuantumLoss(unittest.TestCase):
    def test_fidelity_loss_same_state(self):
        from modules.quantum_dl import fidelity_loss
        psi = np.array([0.6, 0.8])
        self.assertAlmostEqual(float(fidelity_loss(psi, psi)), 0.0, places=6)

    def test_fidelity_loss_orthogonal(self):
        from modules.quantum_dl import fidelity_loss
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(float(fidelity_loss(a, b)), 1.0, places=6)

    def test_trace_distance_same_state(self):
        from modules.quantum_dl import trace_distance
        psi = np.array([0.6, 0.8])
        self.assertAlmostEqual(float(trace_distance(psi, psi)), 0.0, places=6)

    def test_trace_distance_orthogonal(self):
        from modules.quantum_dl import trace_distance
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(float(trace_distance(a, b)), 1.0, places=6)

    def test_quantum_kl_divergence(self):
        from modules.quantum_dl import quantum_kl_divergence
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        # Same distributions → KL = 0
        self.assertAlmostEqual(float(quantum_kl_divergence(p, q)), 0.0, places=6)

    def test_quantum_kl_nonneg(self):
        from modules.quantum_dl import quantum_kl_divergence
        p = np.array([0.7, 0.3])
        q = np.array([0.4, 0.6])
        self.assertGreater(float(quantum_kl_divergence(p, q)), 0.0)

    def test_state_mse(self):
        from modules.quantum_dl import state_mse
        a = np.array([0.5, 0.5])
        b = np.array([0.5, 0.5])
        self.assertAlmostEqual(float(state_mse(a, b)), 0.0, places=6)

    def test_amplitude_log_loss(self):
        from modules.quantum_dl import amplitude_log_loss
        # state [1, 0, 0]: prob class 0 = 1 → log_loss(0) = 0
        state = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(amplitude_log_loss(state, 0), 0.0, places=4)

    def test_hellinger_symmetry(self):
        from modules.quantum_dl import hellinger_distance
        p = np.array([0.3, 0.7])
        q = np.array([0.6, 0.4])
        d_pq = float(hellinger_distance(p, q))
        d_qp = float(hellinger_distance(q, p))
        self.assertAlmostEqual(d_pq, d_qp, places=6)

    def test_infidelity_gradient_direction(self):
        """Moving state_pred toward state_target should decrease infidelity."""
        from modules.quantum_dl import fidelity_loss, infidelity_gradient
        pred = np.array([0.8, 0.6])
        pred = pred / np.linalg.norm(pred)
        target = np.array([0.6, 0.8])
        target = target / np.linalg.norm(target)

        loss_before = float(fidelity_loss(pred, target))
        grad = infidelity_gradient(pred, target)
        new_pred = pred - 0.1 * grad
        new_pred = new_pred / np.linalg.norm(new_pred)
        loss_after = float(fidelity_loss(new_pred, target))
        self.assertLess(loss_after, loss_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
