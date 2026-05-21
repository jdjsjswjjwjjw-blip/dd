"""
modules/quantum_dl — Phase D: Quantum-Inspired Neural Networks.

من التقرير الفصل 6.5 (Phase D، اختياري):
    - qnn_circuit:               Quantum NN layers (parametrized rotations + entanglement)
    - quantum_lstm:               QLSTM (QNN-gated cells with memory)
    - variational_quantum_eigen:  VQE for portfolio optimization
    - quantum_inspired_loss:      Fidelity, KL, trace distance, Hellinger losses

quantum-inspired numpy. لا tensorflow/pytorch dependency.
"""

from .qnn_circuit import (
    QNNCircuit,
    QNNLayer,
    rx_gate,
    ry_gate,
    rz_gate,
)
from .quantum_lstm import (
    QLSTM,
    QLSTMCell,
)
from .variational_quantum_eigen import (
    VQEPortfolioConfig,
    build_portfolio_hamiltonian,
    expectation_via_state,
    vqe_portfolio_optimize,
)
from .quantum_inspired_loss import (
    amplitude_log_loss,
    fidelity_loss,
    hellinger_distance,
    infidelity_gradient,
    quantum_kl_divergence,
    state_mse,
    trace_distance,
)

__all__ = [
    # qnn_circuit
    "QNNCircuit", "QNNLayer", "rx_gate", "ry_gate", "rz_gate",
    # quantum_lstm
    "QLSTM", "QLSTMCell",
    # variational_quantum_eigen
    "VQEPortfolioConfig", "build_portfolio_hamiltonian",
    "expectation_via_state", "vqe_portfolio_optimize",
    # quantum_inspired_loss
    "amplitude_log_loss", "fidelity_loss", "hellinger_distance",
    "infidelity_gradient", "quantum_kl_divergence", "state_mse", "trace_distance",
]
