"""
modules/quantum_dl/variational_quantum_eigen.py
───────────────────────────────────────────────
Sprint 4 (Phase D - التقرير 6.5): VQE for portfolio optimization.

Extends quantum_discovery.vqe_alpha_finder بـ:
    - Parametrized ansatz (variational form)
    - Cost function = ⟨ψ(θ)|H|ψ(θ)⟩
    - Optimizer: gradient descent على θ

في الـ trading context:
    H = Markowitz Hamiltonian = w^T Σ w - λ * μ^T w
    minimize ⟨ψ|H|ψ⟩ subject to ‖ψ‖ = 1

التقرير: ⚛ quantum_dl/variational_quantum_eigen.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.quantum_dl.qnn_circuit import QNNCircuit


@dataclass
class VQEPortfolioConfig:
    """إعدادات VQE portfolio optimizer."""

    n_qubits: int  # ≥ N (assets)
    n_layers: int = 3
    risk_aversion: float = 1.0
    max_iter: int = 50
    learning_rate: float = 0.05
    finite_diff_eps: float = 1e-4
    seed: int = 42


def build_portfolio_hamiltonian(
    returns_mean: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Portfolio Hamiltonian.

    Markowitz objective: maximize μᵀw - λ wᵀΣw  ⟺  minimize -μᵀw + λ wᵀΣw.

    As a diagonal observable (per-asset cost), we use:
        Hᵢⱼ = λ Σᵢⱼ - μᵢ δᵢⱼ
    """
    mu = np.asarray(returns_mean, dtype=np.float64)
    sigma = np.asarray(covariance, dtype=np.float64)
    n = mu.size
    H = risk_aversion * sigma - np.diag(mu)
    return (H + H.T) / 2.0, {"n_assets": n, "risk_aversion": risk_aversion}


def expectation_via_state(state: np.ndarray, hamiltonian: np.ndarray) -> float:
    """⟨ψ|H|ψ⟩ (real-valued)."""
    psi = np.asarray(state, dtype=np.float64)
    H = np.asarray(hamiltonian, dtype=np.float64)
    # If state larger than H, truncate
    n = H.shape[0]
    if psi.size != n:
        if psi.size > n:
            psi = psi[:n]
        else:
            padded = np.zeros(n)
            padded[: psi.size] = psi
            psi = padded
        # Renormalize
        nrm = np.linalg.norm(psi)
        if nrm > 1e-12:
            psi = psi / nrm
    return float(psi @ H @ psi)


def vqe_portfolio_optimize(
    returns_mean: np.ndarray,
    covariance: np.ndarray,
    config: VQEPortfolioConfig | None = None,
) -> dict:
    """VQE-style portfolio optimization.

    Variational ansatz: QNN circuit مع n_qubits قابلة للتعديل.
    Loss: ⟨ψ(θ)|H|ψ(θ)⟩.

    Returns
    -------
    dict مع:
        weights : (N,) النسبة لكل asset
        final_loss : float
        history : list of losses per iteration
        circuit : QNNCircuit (مع θ المُحسَّن)
    """
    if config is None:
        config = VQEPortfolioConfig(n_qubits=max(4, returns_mean.size))

    mu = np.asarray(returns_mean, dtype=np.float64)
    sigma = np.asarray(covariance, dtype=np.float64)
    n_assets = mu.size

    H, _info = build_portfolio_hamiltonian(mu, sigma, config.risk_aversion)

    circuit = QNNCircuit(
        n_qubits=max(config.n_qubits, n_assets),
        n_layers=config.n_layers,
        seed=config.seed,
    )

    # Initial input: uniform
    x0 = np.full(n_assets, 1.0 / np.sqrt(n_assets))

    def loss_fn(params):
        circuit.set_parameters(params)
        state = circuit.forward(x0)
        return expectation_via_state(state, H)

    params = circuit.get_parameters()
    history = []

    for it in range(config.max_iter):
        # Finite difference gradient
        grad = np.zeros_like(params)
        eps = config.finite_diff_eps
        base = loss_fn(params)
        for i in range(params.size):
            p_plus = params.copy()
            p_plus[i] += eps
            grad[i] = (loss_fn(p_plus) - base) / eps
        params -= config.learning_rate * grad
        history.append(float(base))

    circuit.set_parameters(params)
    final_state = circuit.forward(x0)
    # Project to n_assets via |α|²
    if final_state.size > n_assets:
        weights = final_state[:n_assets] ** 2
    else:
        weights = final_state ** 2
    # Normalize to sum to 1
    weights = weights / max(weights.sum(), 1e-12)

    return {
        "weights": weights,
        "final_loss": float(loss_fn(params)),
        "history": history,
        "circuit": circuit,
        "n_parameters": int(circuit.n_parameters),
    }
