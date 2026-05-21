"""
modules/quantum_dl/quantum_lstm.py
─────────────────────────────────
Sprint 4 (Phase D - التقرير 6.5): Quantum LSTM (parametrized circuits).

LSTM التقليدي:
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   # forget
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   # input
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   # output
    g_t = tanh(W_g · [h_{t-1}, x_t] + b_g) # cell candidate

QLSTM (هذا الـ implementation):
    استبدال gates الخطية بـ QNN circuits.
    كل gate = parametrized rotation + entanglement.

    forget = qnn_f(concat(h, x))
    input  = qnn_i(concat(h, x))
    ...

quantum-inspired (real-valued).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from modules.quantum_dl.qnn_circuit import QNNCircuit


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -50, 50))


@dataclass
class QLSTMCell:
    """Single QLSTM cell."""

    input_dim: int
    hidden_dim: int
    n_qnn_layers: int = 2
    seed: int = 42

    forget_qnn: QNNCircuit = field(init=False)
    input_qnn: QNNCircuit = field(init=False)
    output_qnn: QNNCircuit = field(init=False)
    cell_qnn: QNNCircuit = field(init=False)

    # Output projection (h matches hidden_dim)
    h_proj: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.RandomState(self.seed)
        n_qubits = self.input_dim + self.hidden_dim
        self.forget_qnn = QNNCircuit(n_qubits=n_qubits, n_layers=self.n_qnn_layers, seed=self.seed)
        self.input_qnn = QNNCircuit(n_qubits=n_qubits, n_layers=self.n_qnn_layers, seed=self.seed + 1)
        self.output_qnn = QNNCircuit(n_qubits=n_qubits, n_layers=self.n_qnn_layers, seed=self.seed + 2)
        self.cell_qnn = QNNCircuit(n_qubits=n_qubits, n_layers=self.n_qnn_layers, seed=self.seed + 3)
        # Projection from n_qubits → hidden_dim
        self.h_proj = rng.randn(n_qubits, self.hidden_dim) * 0.1

    def step(
        self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """One QLSTM step.

        Parameters
        ----------
        x : (input_dim,)
        h_prev : (hidden_dim,)
        c_prev : (hidden_dim,)

        Returns
        -------
        h, c : new hidden and cell state
        """
        combined = np.concatenate([h_prev, x])  # (input + hidden,)
        n_qubits = self.input_dim + self.hidden_dim

        f_state = self.forget_qnn.forward(combined)
        i_state = self.input_qnn.forward(combined)
        o_state = self.output_qnn.forward(combined)
        g_state = self.cell_qnn.forward(combined)

        # Project to hidden_dim
        f = _sigmoid(f_state @ self.h_proj)
        i = _sigmoid(i_state @ self.h_proj)
        o = _sigmoid(o_state @ self.h_proj)
        g = _tanh(g_state @ self.h_proj)

        # Standard LSTM update
        c = f * c_prev + i * g
        h = o * _tanh(c)
        return h, c

    @property
    def n_parameters(self) -> int:
        return (
            self.forget_qnn.n_parameters
            + self.input_qnn.n_parameters
            + self.output_qnn.n_parameters
            + self.cell_qnn.n_parameters
            + self.h_proj.size
        )


@dataclass
class QLSTM:
    """Multi-layer QLSTM."""

    input_dim: int
    hidden_dim: int
    n_layers: int = 1
    n_qnn_layers: int = 2
    seed: int = 42

    cells: list[QLSTMCell] = field(init=False)

    def __post_init__(self):
        self.cells = []
        for i in range(self.n_layers):
            inp = self.input_dim if i == 0 else self.hidden_dim
            self.cells.append(QLSTMCell(
                input_dim=inp,
                hidden_dim=self.hidden_dim,
                n_qnn_layers=self.n_qnn_layers,
                seed=self.seed + i * 10,
            ))

    def forward(self, sequence: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Forward pass على sequence.

        Parameters
        ----------
        sequence : (T, input_dim)

        Returns
        -------
        outputs : (T, hidden_dim) — outputs من الـ last layer
        final_states : list of (h, c) tuples per layer
        """
        arr = np.asarray(sequence, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        T = arr.shape[0]
        if T == 0:
            return np.zeros((0, self.hidden_dim)), []

        outputs = np.zeros((T, self.hidden_dim))
        layer_input = arr
        final_states = []

        for li, cell in enumerate(self.cells):
            h = np.zeros(self.hidden_dim)
            c = np.zeros(self.hidden_dim)
            layer_out = np.zeros((T, self.hidden_dim))
            for t in range(T):
                h, c = cell.step(layer_input[t], h, c)
                layer_out[t] = h
            final_states.append((h.copy(), c.copy()))
            layer_input = layer_out
            outputs = layer_out
        return outputs, final_states

    @property
    def n_parameters(self) -> int:
        return sum(cell.n_parameters for cell in self.cells)
