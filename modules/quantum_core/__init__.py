"""
modules/quantum_core — Quantum-inspired core primitives.

من التقرير الفصل 6:
    - state_preparation:     DataFrame → |ψ⟩ tensor superposition
    - entanglement_gates:    CNOT, Bell, GHZ, Toffoli, SWAP, controlled-phase
    - interference_amplifier: Grover, amplitude amplification
    - measurement_engine:    wave function collapse → decision
    - decoherence_handler:   noise filtering, error correction
"""

from .state_preparation import (
    QuantumState,
    hadamard_transform,
    normalize_amplitudes,
    prepare_state_from_dataframe,
    superposition_from_scores,
)
from .entanglement_gates import (
    bell_state,
    cnot_gate,
    controlled_phase,
    ghz_state,
    swap_gate,
    toffoli_gate,
)
from .interference_amplifier import (
    GroverAlphaSearch,
    amplitude_amplification,
    diffusion_operator,
    grover_iteration,
    grover_oracle,
    optimal_iterations,
)
from .measurement_engine import (
    MeasurementEngine,
    born_probabilities,
    confidence_from_state,
    expectation_value,
    fidelity,
    inverse_participation_ratio,
    measure_greedy,
    measure_probabilistic,
    von_neumann_entropy,
)
from .decoherence_handler import (
    DecoherenceHandler,
    detect_outliers,
    error_correction,
    low_pass_filter,
    renormalize,
    threshold_filter,
    winsorize_amplitudes,
)

__all__ = [
    # state_preparation
    "QuantumState", "hadamard_transform", "normalize_amplitudes",
    "prepare_state_from_dataframe", "superposition_from_scores",
    # entanglement_gates
    "bell_state", "cnot_gate", "controlled_phase", "ghz_state",
    "swap_gate", "toffoli_gate",
    # interference_amplifier
    "GroverAlphaSearch", "amplitude_amplification", "diffusion_operator",
    "grover_iteration", "grover_oracle", "optimal_iterations",
    # measurement_engine
    "MeasurementEngine", "born_probabilities", "confidence_from_state",
    "expectation_value", "fidelity", "inverse_participation_ratio",
    "measure_greedy", "measure_probabilistic", "von_neumann_entropy",
    # decoherence_handler
    "DecoherenceHandler", "detect_outliers", "error_correction",
    "low_pass_filter", "renormalize", "threshold_filter", "winsorize_amplitudes",
]
