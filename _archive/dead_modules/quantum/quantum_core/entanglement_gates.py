"""
modules/quantum_core/entanglement_gates.py
──────────────────────────────────────────
Sprint 2 (التقرير 6.2): Entanglement gates.

Gates:
    - CNOT (Controlled-NOT)
    - Bell state construction (Φ+, Φ-, Ψ+, Ψ-)
    - GHZ (3+ qubit entanglement)
    - Toffoli (CCNOT, 3-qubit controlled)

quantum-inspired numpy implementation. يحول correlations الإحصائية
إلى structurally embedded features.
"""

from __future__ import annotations

import numpy as np


def cnot_gate(
    control: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    amplification: float = 2.0,
    deflation: float = 0.5,
) -> np.ndarray:
    """CNOT: target يتحول IFF control في حالة |1⟩.

    Classical interpretation:
        if control > threshold:
            target' = target * amplification
        else:
            target' = target * deflation
    """
    c = np.asarray(control, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    return np.where(c > threshold, t * amplification, t * deflation)


def bell_state(a: np.ndarray, b: np.ndarray, kind: str = "phi+") -> np.ndarray:
    """Bell state construction.

    The 4 Bell basis states:
        |Φ+⟩ = (|00⟩ + |11⟩)/√2  — كلاهما high أو كلاهما low
        |Φ-⟩ = (|00⟩ - |11⟩)/√2  — مثل Φ+ لكن بـ phase معكوسة
        |Ψ+⟩ = (|01⟩ + |10⟩)/√2  — anti-correlated
        |Ψ-⟩ = (|01⟩ - |10⟩)/√2  — anti-correlated بـ phase معكوسة

    Returns scalar feature ∈ [-1, +1] حسب الـ kind:
        joint = a * b              (كلاهما مرتفع)
        anti  = (1-a) * (1-b)      (كلاهما منخفض)
        mixed = a*(1-b) + (1-a)*b  (مختلطان)

        Φ+ = joint + anti - mixed   (positive correlation)
        Φ- = joint - anti - mixed   (mixed phase)
        Ψ+ = mixed - joint - anti   (negative correlation)
        Ψ- = mixed - joint + anti
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    joint = a * b
    anti = (1.0 - a) * (1.0 - b)
    mixed = a * (1.0 - b) + (1.0 - a) * b

    if kind == "phi+":
        return joint + anti - mixed
    if kind == "phi-":
        return joint - anti - mixed
    if kind == "psi+":
        return mixed - joint - anti
    if kind == "psi-":
        return mixed - joint + anti
    raise ValueError(f"Unknown Bell kind: {kind!r}")


def ghz_state(features: list[np.ndarray]) -> np.ndarray:
    """GHZ state: |GHZ⟩ = (|0...0⟩ + |1...1⟩)/√2.

    إجمالي N qubits متشابكة. الـ feature output:
        all_high - mixed - none_above_partial + all_low

    الصيغة المبسّطة: all_high + all_low (n-particle correlation).

    Parameters
    ----------
    features : list of arrays (مساوية في الشكل)
        كل array قيمها في [0, 1] (نُطبَّع إن لزم).
    """
    if len(features) < 2:
        raise ValueError("GHZ requires ≥ 2 features")
    arrs = [np.asarray(f, dtype=np.float64) for f in features]
    all_high = arrs[0].copy()
    all_low = (1.0 - arrs[0]).copy()
    for f in arrs[1:]:
        all_high *= f
        all_low *= (1.0 - f)
    return all_high + all_low


def toffoli_gate(c1: np.ndarray, c2: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Toffoli (CCNOT): target يتحول IFF c1 AND c2 = |1⟩.

    Classical:
        activation = c1 * c2
        target' = target * (1 + activation)
    """
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    return t * (1.0 + c1 * c2)


def swap_gate(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SWAP: تبادل state بين qubitين. Classical = (b, a)."""
    return np.asarray(b).copy(), np.asarray(a).copy()


def controlled_phase(
    control: np.ndarray, target: np.ndarray, phase: float = -1.0, threshold: float = 0.5,
) -> np.ndarray:
    """Controlled phase: target *= phase IFF control > threshold."""
    c = np.asarray(control, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    return np.where(c > threshold, t * phase, t)
