"""
modules/quantum_core/state_preparation.py
─────────────────────────────────────────
Sprint 2 (Quantum Core, التقرير 6.4): تحويل DataFrame → quantum state |ψ⟩.

في الحوسبة الكمية:
    |ψ⟩ = Σᵢ αᵢ |i⟩    حيث Σᵢ |αᵢ|² = 1

الـ basis states |i⟩ هنا تمثل alphas/scenarios المختلفة،
و αᵢ هي amplitudes معتمدة على values الـ DataFrame.

لا يوجد quantum hardware — هذا quantum-inspired numpy implementation
طبقاً للتقرير الفصل 6.1 (Superposition).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_amplitudes(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """تطبيع amplitudes بحيث Σ|αᵢ|² = 1."""
    arr = np.asarray(values, dtype=np.float64)
    norm = np.sqrt(np.sum(arr ** 2, axis=-1, keepdims=True))
    safe = np.where(norm < eps, 1.0, norm)
    out = arr / safe
    out = np.where(norm < eps, 0.0, out)
    return out


def hadamard_transform(amplitudes: np.ndarray) -> np.ndarray:
    """تحويل Hadamard-like: equal superposition (كل alpha بـ amplitude متساوية).

    في الكم: H|0⟩ = (|0⟩ + |1⟩)/√2.
    هنا: نخلق uniform superposition بعد التطبيع.
    """
    arr = np.asarray(amplitudes, dtype=np.float64)
    if arr.ndim == 1:
        n = arr.size
        if n == 0:
            return arr
        return np.full(n, 1.0 / np.sqrt(n))
    n = arr.shape[-1]
    shape = list(arr.shape)
    return np.full(shape, 1.0 / np.sqrt(n))


def superposition_from_scores(scores: np.ndarray) -> np.ndarray:
    """تحويل scores (الأكبر = أفضل) إلى amplitudes بتطبيع L2.

    Negative scores يتحولون إلى 0 (لا يمكن أن يكون amplitude سالباً
    في الـ probability sense).
    """
    arr = np.asarray(scores, dtype=np.float64)
    arr = np.where(arr > 0, arr, 0.0)
    return normalize_amplitudes(arr)


def prepare_state_from_dataframe(
    df: pd.DataFrame,
    alpha_cols: list[str],
    method: str = "abs_norm",
) -> np.ndarray:
    """DataFrame → state tensor.

    Parameters
    ----------
    df : pandas DataFrame
    alpha_cols : list[str]
        الأعمدة التي تمثل الـ basis states (alphas).
    method : {'abs_norm', 'hadamard', 'softmax'}
        - 'abs_norm': amplitudes = |value| / ‖values‖
        - 'hadamard': uniform superposition
        - 'softmax': softmax-normalized

    Returns
    -------
    state : np.ndarray shape (n_samples, n_alphas)
    """
    missing = [c for c in alpha_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    sub = df[alpha_cols].fillna(0.0).values.astype(np.float64)

    if method == "abs_norm":
        return normalize_amplitudes(np.abs(sub))
    if method == "hadamard":
        return hadamard_transform(sub)
    if method == "softmax":
        m = sub.max(axis=-1, keepdims=True)
        ex = np.exp(sub - m)
        z = ex.sum(axis=-1, keepdims=True)
        p = ex / np.where(z < 1e-12, 1.0, z)
        return np.sqrt(p)
    raise ValueError(f"Unknown method: {method}")


class QuantumState:
    """يحفظ |ψ⟩ مع operations عليه."""

    __slots__ = ("amplitudes",)

    def __init__(self, amplitudes: np.ndarray):
        self.amplitudes = np.asarray(amplitudes, dtype=np.float64)

    @property
    def n(self) -> int:
        return self.amplitudes.shape[-1]

    @property
    def shape(self) -> tuple:
        return self.amplitudes.shape

    def probabilities(self) -> np.ndarray:
        """|αᵢ|² (Born rule)."""
        return self.amplitudes ** 2

    def norm(self) -> float | np.ndarray:
        return np.linalg.norm(self.amplitudes, axis=-1)

    def normalize(self) -> "QuantumState":
        return QuantumState(normalize_amplitudes(self.amplitudes))

    def entropy(self) -> float | np.ndarray:
        """Shannon entropy للـ probability distribution."""
        p = self.probabilities()
        log_p = np.where(p > 1e-12, np.log(np.where(p > 1e-12, p, 1.0)), 0.0)
        return -np.sum(p * log_p, axis=-1)

    def __repr__(self) -> str:
        return f"QuantumState(n={self.n}, ‖ψ‖={float(np.mean(self.norm())):.4f})"
