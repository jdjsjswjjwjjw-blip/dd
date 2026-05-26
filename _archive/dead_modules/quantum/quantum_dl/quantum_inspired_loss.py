"""
modules/quantum_dl/quantum_inspired_loss.py
───────────────────────────────────────────
Sprint 4 (Phase D - التقرير 6.5): Fidelity-based loss functions.

في DL التقليدي:
    MSE = mean((y_pred - y_true)²)
    CE  = -Σ y_true * log(y_pred)

في الـ QDL:
    Fidelity: F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|² ∈ [0, 1]
    Infidelity loss: L = 1 - F

    Trace distance: D(ρ, σ) = ½ tr|ρ - σ|
    State distance loss: تعميم MSE على quantum states.

تطبيقات:
    - تدريب QNN classifier بـ infidelity
    - تشابه state-to-state بدل value-to-value
    - QAOA cost evaluation
"""

from __future__ import annotations

import numpy as np


def fidelity_loss(state_pred: np.ndarray, state_target: np.ndarray) -> float | np.ndarray:
    """L = 1 - |⟨ψ|φ⟩|².

    For real-valued amplitudes:
        L = 1 - (Σᵢ αᵢ βᵢ)²

    Both states should be normalized.
    """
    a = np.asarray(state_pred, dtype=np.float64)
    b = np.asarray(state_target, dtype=np.float64)
    overlap = np.sum(a * b, axis=-1)
    return 1.0 - overlap ** 2


def infidelity_gradient(
    state_pred: np.ndarray, state_target: np.ndarray
) -> np.ndarray:
    """∂L/∂state_pred = -2 * ⟨ψ|φ⟩ * state_target."""
    a = np.asarray(state_pred, dtype=np.float64)
    b = np.asarray(state_target, dtype=np.float64)
    overlap = np.sum(a * b, axis=-1, keepdims=True)
    return -2.0 * overlap * b


def quantum_kl_divergence(p_pred: np.ndarray, p_target: np.ndarray, eps: float = 1e-12) -> float | np.ndarray:
    """KL divergence على Born probabilities.

    KL(p || q) = Σ p log(p/q)
    """
    p = np.asarray(p_pred, dtype=np.float64)
    q = np.asarray(p_target, dtype=np.float64)
    # Normalize
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), eps)
    q = q / np.maximum(q.sum(axis=-1, keepdims=True), eps)
    safe_q = np.maximum(q, eps)
    log_ratio = np.log(np.maximum(p, eps)) - np.log(safe_q)
    return np.sum(p * log_ratio, axis=-1)


def trace_distance(state_a: np.ndarray, state_b: np.ndarray) -> float | np.ndarray:
    """D(|ψa⟩, |ψb⟩) = √(1 − F(|ψa⟩, |ψb⟩)).

    Equivalent to half the L2 distance for pure states.
    """
    a = np.asarray(state_a, dtype=np.float64)
    b = np.asarray(state_b, dtype=np.float64)
    overlap = np.sum(a * b, axis=-1)
    F = overlap ** 2
    return np.sqrt(np.maximum(1.0 - F, 0.0))


def state_mse(state_pred: np.ndarray, state_target: np.ndarray) -> float | np.ndarray:
    """MSE على state amplitudes (alternative for fidelity)."""
    a = np.asarray(state_pred, dtype=np.float64)
    b = np.asarray(state_target, dtype=np.float64)
    return np.mean((a - b) ** 2, axis=-1)


def amplitude_log_loss(
    state_pred: np.ndarray, target_label: int | np.ndarray, eps: float = 1e-12
) -> float | np.ndarray:
    """Cross-entropy على Born probabilities للـ classification.

    L = -log(|αₜₐᵣ𝓰ₑₜ|²)
    """
    a = np.asarray(state_pred, dtype=np.float64)
    p = a ** 2
    p_norm = p / np.maximum(p.sum(axis=-1, keepdims=True), eps)
    if np.isscalar(target_label):
        target = int(target_label)
        return -float(np.log(max(float(p_norm[..., target]), eps)))
    # batched
    labels = np.asarray(target_label, dtype=int)
    if p_norm.ndim == 1:
        return -np.log(max(float(p_norm[int(labels)]), eps))
    out = -np.log(np.maximum(p_norm[np.arange(p_norm.shape[0]), labels], eps))
    return out


def hellinger_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float | np.ndarray:
    """Hellinger: H(p, q) = (1/√2) ‖√p − √q‖₂.

    Symmetric distance على probability distributions.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), eps)
    q = q / np.maximum(q.sum(axis=-1, keepdims=True), eps)
    diff = np.sqrt(np.maximum(p, 0.0)) - np.sqrt(np.maximum(q, 0.0))
    return np.sqrt(0.5 * np.sum(diff ** 2, axis=-1))
