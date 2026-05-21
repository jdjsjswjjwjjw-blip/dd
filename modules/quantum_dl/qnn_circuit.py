"""
modules/quantum_dl/qnn_circuit.py
────────────────────────────────
Sprint 4 (Phase D - التقرير 6.5): Quantum Neural Network circuits.

QNN في الكم:
    1. Encoding: classical x → quantum |ψ(x)⟩ بـ rotation gates
    2. Parametrized layers: U(θ) = Π Rᵢ(θᵢ) Cⱼₖ
    3. Measurement: ⟨ψ|O|ψ⟩ → classical output

quantum-inspired (real-valued numpy):
    - State: ℝ^{2^n} (بدل ℂ)
    - Rotations: classical sin/cos parametrization
    - Entanglement gates: من quantum_core.entanglement_gates
    - Loss: MSE أو fidelity-based

التقرير: ⚛ quantum_dl/qnn_circuit.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from modules.quantum_core.entanglement_gates import bell_state, cnot_gate
from modules.quantum_core.state_preparation import normalize_amplitudes


def rx_gate(amplitude: np.ndarray, theta: float) -> np.ndarray:
    """Rotation-X (classical analog): mix with shifted version بـ rotation angle.

    Classical:
        cos(θ/2) * x + sin(θ/2) * shift(x)
    """
    arr = np.asarray(amplitude, dtype=np.float64)
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    # shift = roll
    if arr.ndim == 1:
        shifted = np.roll(arr, 1)
    else:
        shifted = np.roll(arr, 1, axis=-1)
    return c * arr + s * shifted


def ry_gate(amplitude: np.ndarray, theta: float) -> np.ndarray:
    """Rotation-Y: mix بـ negative shift."""
    arr = np.asarray(amplitude, dtype=np.float64)
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    if arr.ndim == 1:
        shifted = np.roll(arr, -1)
    else:
        shifted = np.roll(arr, -1, axis=-1)
    return c * arr + s * shifted


def rz_gate(amplitude: np.ndarray, theta: float) -> np.ndarray:
    """Rotation-Z: phase rotation. Real-valued = sign-modulation."""
    arr = np.asarray(amplitude, dtype=np.float64)
    n = arr.shape[-1]
    phases = np.cos(theta * np.arange(n, dtype=np.float64) / n)
    return arr * phases


@dataclass
class QNNLayer:
    """Single parametrized QNN layer."""

    n_qubits: int
    rotation_gate: str = "ry"  # 'rx' | 'ry' | 'rz'
    entangle: bool = True
    theta: np.ndarray | None = None  # length n_qubits

    def __post_init__(self):
        if self.theta is None:
            rng = np.random.RandomState(42)
            self.theta = rng.uniform(-np.pi, np.pi, self.n_qubits)
        else:
            self.theta = np.asarray(self.theta, dtype=np.float64)

    def _rotation(self, state: np.ndarray, t: float) -> np.ndarray:
        if self.rotation_gate == "rx":
            return rx_gate(state, t)
        if self.rotation_gate == "ry":
            return ry_gate(state, t)
        if self.rotation_gate == "rz":
            return rz_gate(state, t)
        raise ValueError(f"Unknown gate: {self.rotation_gate}")

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Apply rotation gates then entanglement."""
        x = np.asarray(state, dtype=np.float64)
        for t in self.theta:
            x = self._rotation(x, float(t))
        if self.entangle and x.shape[-1] >= 2:
            # Apply CNOT-like coupling between adjacent qubits via bell_state interaction
            half = x.shape[-1] // 2
            a = x[..., :half]
            b = x[..., half : 2 * half]
            # mix via bell_state output as gate
            mixed = bell_state(np.abs(a), np.abs(b), kind="phi+") * 0.1
            # apply as additive update
            x = x.copy()
            x[..., :half] = a + mixed
            x[..., half : 2 * half] = b - mixed
        return normalize_amplitudes(x)


@dataclass
class QNNCircuit:
    """Multi-layer Quantum Neural Network."""

    n_qubits: int
    n_layers: int = 3
    rotation_gate: str = "ry"
    layers: list[QNNLayer] = field(default_factory=list)
    seed: int = 42

    def __post_init__(self):
        if not self.layers:
            rng = np.random.RandomState(self.seed)
            self.layers = [
                QNNLayer(
                    n_qubits=self.n_qubits,
                    rotation_gate=self.rotation_gate,
                    entangle=(i < self.n_layers - 1),
                    theta=rng.uniform(-np.pi, np.pi, self.n_qubits),
                )
                for i in range(self.n_layers)
            ]

    def encode_input(self, x: np.ndarray) -> np.ndarray:
        """Classical x → quantum state via amplitude encoding."""
        arr = np.asarray(x, dtype=np.float64)
        # Reshape to n_qubits-sized state per sample
        if arr.ndim == 1:
            # pad or truncate
            if arr.size < self.n_qubits:
                padded = np.zeros(self.n_qubits)
                padded[: arr.size] = arr
                arr = padded
            elif arr.size > self.n_qubits:
                arr = arr[: self.n_qubits]
            return normalize_amplitudes(arr)
        if arr.shape[-1] != self.n_qubits:
            if arr.shape[-1] < self.n_qubits:
                pad = np.zeros((*arr.shape[:-1], self.n_qubits - arr.shape[-1]))
                arr = np.concatenate([arr, pad], axis=-1)
            else:
                arr = arr[..., : self.n_qubits]
        return normalize_amplitudes(arr)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass."""
        state = self.encode_input(x)
        for layer in self.layers:
            state = layer.forward(state)
        return state

    def get_parameters(self) -> np.ndarray:
        """Flatten parameters."""
        return np.concatenate([layer.theta for layer in self.layers])

    def set_parameters(self, params: np.ndarray) -> None:
        """Set from flattened array."""
        params = np.asarray(params, dtype=np.float64)
        offset = 0
        for layer in self.layers:
            n = layer.n_qubits
            layer.theta = params[offset : offset + n].copy()
            offset += n
        if offset != len(params):
            raise ValueError(f"Parameter size mismatch: expected {offset}, got {len(params)}")

    @property
    def n_parameters(self) -> int:
        return sum(layer.n_qubits for layer in self.layers)
