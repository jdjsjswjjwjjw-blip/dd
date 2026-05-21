"""
modules/quantum_core/measurement_engine.py
──────────────────────────────────────────
Sprint 2 (التقرير 6.4): Wave function collapse → trading decision.

Measurement في الكم:
    |ψ⟩ = Σᵢ αᵢ |i⟩  →  basis |k⟩ with probability |αₖ|²

ثلاثة modes:
    - greedy: argmax |αᵢ|² (deterministic)
    - probabilistic: sample من Born distribution
    - expectation: ⟨ψ|O|ψ⟩ للـ observable O

+ confidence metrics (inverse participation ratio, von Neumann entropy).
"""

from __future__ import annotations

import numpy as np


def born_probabilities(amplitudes: np.ndarray) -> np.ndarray:
    """|αᵢ|² with per-row normalization."""
    arr = np.asarray(amplitudes, dtype=np.float64)
    p = arr ** 2
    s = p.sum(axis=-1, keepdims=True)
    safe = np.where(s < 1e-12, 1.0, s)
    return p / safe


def measure_greedy(amplitudes: np.ndarray) -> np.ndarray | int:
    """Greedy collapse: argmax |αᵢ|²."""
    arr = np.asarray(amplitudes, dtype=np.float64)
    p = arr ** 2
    if p.ndim == 1:
        return int(np.argmax(p))
    return np.argmax(p, axis=-1)


def measure_probabilistic(
    amplitudes: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray | int:
    """Probabilistic collapse: sample basis k with probability |αₖ|²."""
    if rng is None:
        rng = np.random.default_rng()
    p = born_probabilities(amplitudes)
    if p.ndim == 1:
        return int(rng.choice(p.size, p=p))
    return np.array([rng.choice(p.shape[-1], p=row) for row in p])


def expectation_value(amplitudes: np.ndarray, observable: np.ndarray) -> float | np.ndarray:
    """⟨ψ|O|ψ⟩ = Σᵢ |αᵢ|² Oᵢᵢ (diagonal observable)."""
    p = born_probabilities(amplitudes)
    obs = np.asarray(observable, dtype=np.float64)
    return np.sum(p * obs, axis=-1)


def inverse_participation_ratio(amplitudes: np.ndarray) -> float | np.ndarray:
    """IPR = Σᵢ |αᵢ|⁴.

    IPR = 1/N → uniform (maximally delocalized)
    IPR = 1.0  → fully localized (one basis dominates)
    """
    p = born_probabilities(amplitudes)
    return np.sum(p ** 2, axis=-1)


def confidence_from_state(amplitudes: np.ndarray) -> float | np.ndarray:
    """Confidence ∈ [0, 1]: 0 = uniform, 1 = pure single-basis state.

    Based on normalized IPR:
        conf = (N * IPR − 1) / (N − 1)
    """
    arr = np.asarray(amplitudes, dtype=np.float64)
    N = arr.shape[-1]
    if N <= 1:
        return np.ones(arr.shape[:-1])
    ipr = inverse_participation_ratio(arr)
    return (ipr * N - 1.0) / (N - 1.0)


def von_neumann_entropy(amplitudes: np.ndarray) -> float | np.ndarray:
    """S = −Σᵢ pᵢ log pᵢ.

    S = 0 → pure state
    S = log N → maximally mixed
    """
    p = born_probabilities(amplitudes)
    log_p = np.where(p > 1e-12, np.log(np.where(p > 1e-12, p, 1.0)), 0.0)
    return -np.sum(p * log_p, axis=-1)


def fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float | np.ndarray:
    """F(|ψa⟩, |ψb⟩) = |⟨ψa|ψb⟩|².

    Real-valued amplitudes: F = (Σᵢ αᵢ βᵢ)².
    """
    a = np.asarray(state_a, dtype=np.float64)
    b = np.asarray(state_b, dtype=np.float64)
    overlap = np.sum(a * b, axis=-1)
    return overlap ** 2


class MeasurementEngine:
    """يجمع mode + post-processing في pipeline واحد."""

    def __init__(
        self,
        mode: str = "greedy",
        confidence_threshold: float = 0.2,
        rng_seed: int | None = None,
    ):
        if mode not in {"greedy", "probabilistic", "expectation"}:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.confidence_threshold = float(confidence_threshold)
        self.rng = np.random.default_rng(rng_seed)

    def measure(self, amplitudes: np.ndarray, observable: np.ndarray | None = None):
        if self.mode == "greedy":
            return measure_greedy(amplitudes)
        if self.mode == "probabilistic":
            return measure_probabilistic(amplitudes, rng=self.rng)
        # expectation
        if observable is None:
            raise ValueError("expectation mode requires observable")
        return expectation_value(amplitudes, observable)

    def decide(self, amplitudes: np.ndarray):
        """Returns (basis_index, confidence). If conf < threshold → -1 (HOLD)."""
        idx = self.measure(amplitudes)
        conf = confidence_from_state(amplitudes)
        if np.isscalar(conf):
            return (int(idx), float(conf)) if conf >= self.confidence_threshold else (-1, float(conf))
        # batched
        mask = conf >= self.confidence_threshold
        idx_out = np.where(mask, idx, -1)
        return idx_out, conf
