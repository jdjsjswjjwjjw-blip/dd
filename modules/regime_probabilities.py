"""
modules/regime_probabilities.py
───────────────────────────────
Sprint 14 (Regime Analysis Report v2، التعديل ④):
Soft Regime Probabilities — smooth transitions في exit logic.

═════════════════════════════════════════════════════════════════════════════
الفكرة العلمية (التقرير ص 10، 20)
═════════════════════════════════════════════════════════════════════════════

المشكلة في الـ hard labels (الحالي):
    شمعة 100: regime = trending  → TP = 2.0 × ATR
    شمعة 101: regime = ranging   → TP = 1.1 × ATR  ← قفزة حادة!

    في trades تحت الانتقال: false exits + slippage في live.

الحل:
    P(regime | features) كـ distribution، ليس argmax label.

    regime_probs = {
        'trending': 0.65,
        'ranging':  0.30,
        'volatile': 0.05,
    }

    TP_weighted = Σ P(r) × TP(r) = 0.65×2.0 + 0.30×1.1 + 0.05×1.8 = 1.72 × ATR

═════════════════════════════════════════════════════════════════════════════
Mathematical Foundation
═════════════════════════════════════════════════════════════════════════════

1. Softmax over signal scores (calibrated):
       P(r_i | x) = exp(z_i / T) / Σ exp(z_j / T)
   T = temperature (T=1 default، T<1 = sharper، T>1 = smoother).

2. Membership ratios from continuous indicators (ADX, ATR, Hurst):
       Trending strength    = sigmoid((ADX - 25) / 10)
       Ranging strength     = 1 - Trending - Volatile
       Volatile strength    = sigmoid((ATR_z - 2) / 1)

3. Temporal smoothing (EMA):
       P_smooth_t = α × P_raw_t + (1-α) × P_smooth_{t-1}

4. Weighted aggregation للـ TP/SL:
       TP_w = Σ_r P(r) × TP_r
       SL_w = Σ_r P(r) × SL_r
       Horizon_w = round(Σ_r P(r) × Horizon_r)

═════════════════════════════════════════════════════════════════════════════
Properties (verified in tests)
═════════════════════════════════════════════════════════════════════════════

P1 (Normalization): Σ P(r) = 1, ∀ rows
P2 (Non-negativity): P(r) ≥ 0
P3 (Convergence to hard): max P(r) → 1 لو signal واضح
P4 (Convergence to uniform): max P(r) → 1/k لو signal فوضوي
P5 (Temporal smoothness): |P_t - P_{t-1}| ≤ (1-α) للـ EMA
P6 (Monotonicity TP): TP_w monotone في P(trending)

═════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REGIMES: tuple[str, ...] = ("trending", "ranging", "volatile")

# Default TP/SL multipliers per regime (يطابق REGIME_TP_SL في النظام)
DEFAULT_REGIME_TP_SL: dict[str, tuple[float, float]] = {
    "trending": (2.0, 1.0),  # (tp_mult, sl_mult)
    "ranging":  (1.1, 1.0),
    "volatile": (1.8, 1.2),
}

DEFAULT_REGIME_HORIZON: dict[str, int] = {
    "trending": 12,
    "ranging":  6,
    "volatile": 3,
}


# ════════════════════════════════════════════════════════════════════════════
# Probability computation
# ════════════════════════════════════════════════════════════════════════════


def softmax(scores: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Softmax مع temperature scaling (numerically stable).

    P_i = exp(z_i / T) / Σ exp(z_j / T)

    Parameters
    ----------
    scores : (..., k) array
    temperature : T > 0. T<1 = sharper، T>1 = smoother
    axis : axis للـ normalization
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    arr = np.asarray(scores, dtype=np.float64) / float(temperature)
    arr = arr - arr.max(axis=axis, keepdims=True)
    ex = np.exp(arr)
    return ex / ex.sum(axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def from_hard_labels(
    labels: pd.Series | np.ndarray,
    regimes: tuple[str, ...] = DEFAULT_REGIMES,
    confidence: float = 1.0,
) -> np.ndarray:
    """تحويل hard labels إلى soft (one-hot أو near-one-hot).

    confidence = 1.0 → exact one-hot
    confidence < 1.0 → distribute mass to other regimes uniformly

    Returns
    -------
    (N, k) array
    """
    if not 0.0 < confidence <= 1.0:
        raise ValueError(f"confidence must be in (0, 1], got {confidence}")

    arr = np.asarray(labels)
    n = arr.shape[0]
    k = len(regimes)
    result = np.full((n, k), (1.0 - confidence) / max(k - 1, 1))

    regime_to_idx = {r: i for i, r in enumerate(regimes)}
    for i, label in enumerate(arr):
        if label in regime_to_idx:
            result[i, :] = (1.0 - confidence) / max(k - 1, 1)
            result[i, regime_to_idx[label]] = confidence
        else:
            # unknown label → uniform
            result[i, :] = 1.0 / k
    return result


def from_indicators(
    adx: np.ndarray,
    atr_zscore: np.ndarray,
    regimes: tuple[str, ...] = DEFAULT_REGIMES,
    adx_threshold: float = 25.0,
    adx_scale: float = 10.0,
    atr_threshold: float = 2.0,
    atr_scale: float = 1.0,
) -> np.ndarray:
    """Soft probabilities من ADX + ATR Z-score.

    Membership functions:
        trending = sigmoid((ADX - threshold) / scale)
        volatile = sigmoid((ATR_z - threshold) / scale)
        ranging  = 1 - trending - volatile  (clamped to [0, 1])

    Renormalized via softmax-like rescale.

    Returns
    -------
    (N, 3) array بـ regimes order = (trending, ranging, volatile)
    """
    if regimes != DEFAULT_REGIMES:
        warnings.warn(
            f"from_indicators expects regimes={DEFAULT_REGIMES}, got {regimes}",
            stacklevel=2,
        )

    adx = np.asarray(adx, dtype=np.float64)
    atr_z = np.asarray(atr_zscore, dtype=np.float64)

    trending = sigmoid((adx - adx_threshold) / adx_scale)
    volatile = sigmoid((atr_z - atr_threshold) / atr_scale)
    ranging = np.clip(1.0 - trending - volatile, 0.0, 1.0)

    stacked = np.stack([trending, ranging, volatile], axis=-1)

    # Renormalize so each row sums to 1
    row_sum = stacked.sum(axis=-1, keepdims=True)
    row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)
    return stacked / row_sum


# ════════════════════════════════════════════════════════════════════════════
# Temporal smoothing (EMA)
# ════════════════════════════════════════════════════════════════════════════


def smooth_probabilities(
    probs: np.ndarray,
    alpha: float = 0.3,
) -> np.ndarray:
    """EMA على probability matrix.

    P_smooth_t = α × P_raw_t + (1-α) × P_smooth_{t-1}

    α = 1.0 → no smoothing
    α = 0.0 → pure history

    Returns: (N, k) من normalized smoothed probabilities.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    arr = np.asarray(probs, dtype=np.float64)
    if arr.size == 0:
        return arr
    if arr.ndim != 2:
        raise ValueError(f"probs must be 2D (N, k), got shape {arr.shape}")

    out = np.zeros_like(arr)
    out[0] = arr[0]
    for t in range(1, arr.shape[0]):
        out[t] = alpha * arr[t] + (1.0 - alpha) * out[t - 1]

    # Renormalize (drift correction)
    row_sum = out.sum(axis=-1, keepdims=True)
    row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)
    return out / row_sum


# ════════════════════════════════════════════════════════════════════════════
# Weighted TP/SL aggregation
# ════════════════════════════════════════════════════════════════════════════


def weighted_tp_sl(
    probs: dict[str, float] | np.ndarray,
    regime_tp_sl: dict[str, tuple[float, float]] = None,
    regimes: tuple[str, ...] = DEFAULT_REGIMES,
) -> tuple[float, float]:
    """Weighted average للـ TP/SL multipliers.

    TP_w = Σ_r P(r) × TP_r
    SL_w = Σ_r P(r) × SL_r

    Parameters
    ----------
    probs : dict {regime: prob} أو 1D array بـ regimes order
    regime_tp_sl : dict {regime: (tp_mult, sl_mult)}، default = DEFAULT_REGIME_TP_SL

    Returns
    -------
    (tp_mult_weighted, sl_mult_weighted)
    """
    if regime_tp_sl is None:
        regime_tp_sl = DEFAULT_REGIME_TP_SL

    if isinstance(probs, dict):
        tp = sum(probs.get(r, 0.0) * regime_tp_sl[r][0] for r in regimes)
        sl = sum(probs.get(r, 0.0) * regime_tp_sl[r][1] for r in regimes)
    else:
        arr = np.asarray(probs, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"probs must be 1D for single-row, got shape {arr.shape}")
        tp = sum(arr[i] * regime_tp_sl[r][0] for i, r in enumerate(regimes))
        sl = sum(arr[i] * regime_tp_sl[r][1] for i, r in enumerate(regimes))

    return float(tp), float(sl)


def weighted_tp_sl_batch(
    probs: np.ndarray,
    regime_tp_sl: dict[str, tuple[float, float]] = None,
    regimes: tuple[str, ...] = DEFAULT_REGIMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized weighted TP/SL على batch.

    Parameters
    ----------
    probs : (N, k) array

    Returns
    -------
    (tp_array, sl_array) كل واحد (N,)
    """
    if regime_tp_sl is None:
        regime_tp_sl = DEFAULT_REGIME_TP_SL

    arr = np.asarray(probs, dtype=np.float64)
    tp_vec = np.array([regime_tp_sl[r][0] for r in regimes])
    sl_vec = np.array([regime_tp_sl[r][1] for r in regimes])

    tp_w = arr @ tp_vec
    sl_w = arr @ sl_vec
    return tp_w, sl_w


def weighted_horizon(
    probs: dict[str, float] | np.ndarray,
    regime_horizon: dict[str, int] = None,
    regimes: tuple[str, ...] = DEFAULT_REGIMES,
) -> int:
    """Weighted horizon (rounded to nearest integer)."""
    if regime_horizon is None:
        regime_horizon = DEFAULT_REGIME_HORIZON

    if isinstance(probs, dict):
        h = sum(probs.get(r, 0.0) * regime_horizon[r] for r in regimes)
    else:
        arr = np.asarray(probs, dtype=np.float64)
        h = sum(arr[i] * regime_horizon[r] for i, r in enumerate(regimes))

    return int(round(h))


# ════════════════════════════════════════════════════════════════════════════
# Diagnostics & validation
# ════════════════════════════════════════════════════════════════════════════


def entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per row.

    H = -Σ p log p

    H = 0 → pure label
    H = log(k) → uniform
    """
    arr = np.asarray(probs, dtype=np.float64)
    safe = np.where(arr > 1e-12, arr, 1.0)
    log_p = np.log(safe)
    return -np.sum(arr * log_p, axis=-1)


def transition_smoothness(probs: np.ndarray) -> float:
    """متوسط |P_t - P_{t-1}| (L1 norm per step).

    قيمة منخفضة = smooth.
    قيمة عالية = jumpy.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.shape[0] < 2:
        return 0.0
    diffs = np.abs(arr[1:] - arr[:-1]).sum(axis=-1)
    return float(diffs.mean())


def validate_probability_matrix(
    probs: np.ndarray,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """يفحص خصائص P1, P2 (normalization + non-negativity)."""
    arr = np.asarray(probs, dtype=np.float64)
    row_sums = arr.sum(axis=-1)
    max_dev = float(np.max(np.abs(row_sums - 1.0))) if arr.size > 0 else 0.0
    has_neg = bool(np.any(arr < -tolerance)) if arr.size > 0 else False

    return {
        "valid": (max_dev < tolerance) and not has_neg,
        "max_deviation_from_unity": max_dev,
        "has_negatives": has_neg,
        "shape": tuple(arr.shape),
        "min_value": float(arr.min()) if arr.size > 0 else 0.0,
        "max_value": float(arr.max()) if arr.size > 0 else 0.0,
    }


# ════════════════════════════════════════════════════════════════════════════
# High-level pipeline integration
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class SoftRegimeOutput:
    """مخرج weighted_tp_sl_pipeline."""

    probs: np.ndarray  # (N, 3)
    tp_weighted: np.ndarray  # (N,)
    sl_weighted: np.ndarray  # (N,)
    smoothness: float
    mean_entropy: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def compute_soft_regime_pipeline(
    df: pd.DataFrame,
    method: str = "from_labels",
    smoothing_alpha: float = 0.3,
    label_col: str = "regime_label",
    adx_col: str = "adx_14",
    atr_zscore_col: str = "atr_z",
    confidence_for_labels: float = 0.85,
) -> SoftRegimeOutput:
    """High-level pipeline: features → probs → smoothing → TP/SL.

    methods:
        'from_labels'    : hard labels → soft (confidence-controlled)
        'from_indicators': ADX + ATR_z → soft via sigmoid membership

    Returns
    -------
    SoftRegimeOutput dataclass
    """
    if method == "from_labels":
        if label_col not in df.columns:
            raise KeyError(f"label_col '{label_col}' غير موجود")
        probs_raw = from_hard_labels(
            df[label_col].values,
            confidence=confidence_for_labels,
        )
    elif method == "from_indicators":
        for col in (adx_col, atr_zscore_col):
            if col not in df.columns:
                raise KeyError(f"column '{col}' غير موجود (method=from_indicators)")
        probs_raw = from_indicators(
            df[adx_col].values,
            df[atr_zscore_col].values,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    probs_smooth = smooth_probabilities(probs_raw, alpha=smoothing_alpha)
    tp_w, sl_w = weighted_tp_sl_batch(probs_smooth)

    return SoftRegimeOutput(
        probs=probs_smooth,
        tp_weighted=tp_w,
        sl_weighted=sl_w,
        smoothness=transition_smoothness(probs_smooth),
        mean_entropy=float(entropy(probs_smooth).mean()),
        diagnostics={
            "method": method,
            "smoothing_alpha": smoothing_alpha,
            "n_rows": int(len(probs_smooth)),
            "validation": validate_probability_matrix(probs_smooth),
            "tp_range": [float(tp_w.min()), float(tp_w.max())],
            "sl_range": [float(sl_w.min()), float(sl_w.max())],
        },
    )
