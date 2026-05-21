"""
modules/quantum_core/decoherence_handler.py
───────────────────────────────────────────
Sprint 2 (التقرير 6.4): Noise filtering + error correction.

Decoherence في الكم = فقدان معلومات الـ quantum state بسبب التفاعل
مع البيئة. هنا نستخدم classical filters لمحاكاة:

    - low_pass_filter: تنعيم temporal لمحاربة noise spikes
    - threshold_filter: zero-out amplitudes تحت threshold
    - renormalize: استعادة ‖ψ‖ = 1 بعد gates
    - error_correction: detect outliers + correct

quantum-inspired (numpy-based).
"""

from __future__ import annotations

import numpy as np


def low_pass_filter(amplitudes: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """EMA filter: y[t] = α * x[t] + (1-α) * y[t-1].

    α = 1.0 → no smoothing
    α = 0.0 → pure history
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    arr = np.asarray(amplitudes, dtype=np.float64)
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        out = np.zeros_like(arr)
        out[0] = arr[0]
        for i in range(1, arr.size):
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
        return out
    # 2D: filter along time axis (axis=0)
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1, arr.shape[0]):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def threshold_filter(amplitudes: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Zero-out amplitudes بـ |α| < threshold."""
    arr = np.asarray(amplitudes, dtype=np.float64)
    mask = np.abs(arr) >= float(threshold)
    return arr * mask


def renormalize(amplitudes: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """استعادة ‖ψ‖² = Σ|αᵢ|² = 1.

    Per-row normalization إذا batched.
    """
    arr = np.asarray(amplitudes, dtype=np.float64)
    sq_sum = np.sum(arr ** 2, axis=-1, keepdims=True)
    norm = np.sqrt(sq_sum)
    safe = np.where(norm < eps, 1.0, norm)
    return arr / safe


def winsorize_amplitudes(
    amplitudes: np.ndarray, lower_q: float = 0.01, upper_q: float = 0.99
) -> np.ndarray:
    """Cap outliers بـ quantile-based winsorization."""
    arr = np.asarray(amplitudes, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = np.quantile(arr, lower_q)
    hi = np.quantile(arr, upper_q)
    return np.clip(arr, lo, hi)


def detect_outliers(amplitudes: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """Boolean mask للـ outliers (|x − μ| > n_sigma * σ)."""
    arr = np.asarray(amplitudes, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)
    mu = arr.mean()
    sd = arr.std()
    if sd < 1e-12:
        return np.zeros_like(arr, dtype=bool)
    return np.abs(arr - mu) > n_sigma * sd


def error_correction(
    amplitudes: np.ndarray, n_sigma: float = 3.0, replace_with: str = "median"
) -> np.ndarray:
    """يكتشف outliers ويستبدلها.

    replace_with:
        - 'median': median للـ array
        - 'mean': mean
        - 'zero': 0
    """
    arr = np.asarray(amplitudes, dtype=np.float64).copy()
    outliers = detect_outliers(arr, n_sigma=n_sigma)
    if not outliers.any():
        return arr
    if replace_with == "median":
        replacement = np.median(arr[~outliers]) if (~outliers).any() else 0.0
    elif replace_with == "mean":
        replacement = arr[~outliers].mean() if (~outliers).any() else 0.0
    elif replace_with == "zero":
        replacement = 0.0
    else:
        raise ValueError(f"Unknown replace_with: {replace_with!r}")
    arr[outliers] = replacement
    return arr


class DecoherenceHandler:
    """يجمع filters في pipeline."""

    def __init__(
        self,
        lp_alpha: float = 0.7,
        threshold: float = 0.01,
        n_sigma: float = 3.0,
        apply_lp: bool = True,
        apply_threshold: bool = True,
        apply_error_correction: bool = False,
    ):
        self.lp_alpha = lp_alpha
        self.threshold = threshold
        self.n_sigma = n_sigma
        self.apply_lp = apply_lp
        self.apply_threshold = apply_threshold
        self.apply_error_correction = apply_error_correction

    def apply(self, amplitudes: np.ndarray) -> np.ndarray:
        x = np.asarray(amplitudes, dtype=np.float64).copy()
        if self.apply_lp:
            x = low_pass_filter(x, self.lp_alpha)
        if self.apply_threshold:
            x = threshold_filter(x, self.threshold)
        if self.apply_error_correction:
            x = error_correction(x, n_sigma=self.n_sigma)
        x = renormalize(x)
        return x
