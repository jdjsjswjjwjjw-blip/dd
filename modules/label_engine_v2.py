"""
label_engine_v2.py — Triple Barrier labeling مع MFE/MAE fallback.

يحلّ مشاكل dynamic_labels.py (السطر 779):
    1. استخدام future[-1] فقط → ضياع الحركة في المسار (mean-revert يخفي MFE).
    2. عتبة ثابتة (5 pips) → لا تتكيّف مع ATR.
    3. أفق طويل ثابت (50 شمعة) → يخلط trending و mean-revert.

المنهج الجديد:
    لكل صف t:
        upper = entry + tp_atr_mult * atr[t]
        lower = entry - sl_atr_mult * atr[t]
        horizon = adaptive_horizons[t]
    أول barrier يُلامس يحدد الاتجاه (LONG/SHORT).
    timeout → فحص MFE vs MAE: الفائز ضِعف الخاسر يُحدد الاتجاه.

API:
    label_triple_barrier_atr(prices, atr, horizons, ...) -> dict of arrays
    compute_atr(highs, lows, closes, window=14) -> np.ndarray

Constants:
    DIR_LONG, DIR_SHORT, DIR_NEUTRAL (متوافقة مع modules.dynamic_labels)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── Constants (مطابقة لـ modules.dynamic_labels) ─────────────────────────────
DIR_LONG: int = 0
DIR_SHORT: int = 1
DIR_NEUTRAL: int = 2

# Outcome path codes (للتشخيص)
PATH_TP_LONG: int = 1   # هبط على upper barrier
PATH_TP_SHORT: int = 2  # هبط على lower barrier
PATH_TIMEOUT_MFE_LONG: int = 3   # timeout + MFE > 2*MAE
PATH_TIMEOUT_MFE_SHORT: int = 4  # timeout + MAE > 2*MFE
PATH_TIMEOUT_NEUTRAL: int = 5    # timeout بدون قرار


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Configuration للـ Triple Barrier labeling.

    tp_atr_mult : float
        مضاعف ATR لـ take-profit barrier (upper للـ LONG).
    sl_atr_mult : float
        مضاعف ATR لـ stop-loss barrier (lower للـ LONG).
    mfe_mae_ratio : float
        نسبة MFE/MAE المطلوبة لاتخاذ قرار اتجاهي عند timeout.
        2.0 = الفائز يجب أن يكون ضِعف الخاسر على الأقل.
    min_move_atr_mult : float
        الحد الأدنى للحركة كمضاعف ATR (يمنع labeling لحركات أصغر من noise).
    horizon_default : int
        أفق افتراضي إذا لم تُمرَّر horizons.
    """
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0
    mfe_mae_ratio: float = 2.0
    min_move_atr_mult: float = 1.0
    horizon_default: int = 20


DEFAULT_CONFIG = TripleBarrierConfig()


# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """ATR (Wilder smoothing) - causal، يستخدم بيانات سابقة فقط.

    Returns
    -------
    np.ndarray
        ATR per row. القيم الأولى (قبل window) تُملأ بـ TR المتوسط لتجنّب NaN.
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)

    if n == 0:
        return np.zeros(0, dtype=np.float64)

    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = closes[0]
    prev_close[1:] = closes[:-1]

    tr = np.maximum.reduce([
        highs - lows,
        np.abs(highs - prev_close),
        np.abs(lows - prev_close),
    ])

    atr = np.empty(n, dtype=np.float64)
    if n < window:
        # fallback: cumulative mean
        atr[:] = np.cumsum(tr) / np.maximum(np.arange(1, n + 1), 1)
        return atr

    # initial seed = simple mean of first `window` TRs
    seed = tr[:window].mean()
    atr[:window] = seed
    alpha = 1.0 / window
    for i in range(window, n):
        atr[i] = (1 - alpha) * atr[i - 1] + alpha * tr[i]
    return atr


# ── Core labeling ────────────────────────────────────────────────────────────
def label_triple_barrier_atr(
    prices: np.ndarray,
    atr: np.ndarray,
    horizons: Optional[np.ndarray] = None,
    config: TripleBarrierConfig = DEFAULT_CONFIG,
) -> dict[str, np.ndarray]:
    """Triple Barrier labeling مع MFE/MAE fallback.

    Parameters
    ----------
    prices : np.ndarray (n,)
        أسعار الإغلاق.
    atr : np.ndarray (n,)
        ATR per row (من compute_atr أو مصدر آخر).
    horizons : np.ndarray (n,) أو None
        أفق per-row (شمعات). إذا None → config.horizon_default للكل.
    config : TripleBarrierConfig

    Returns
    -------
    dict مع المفاتيح:
        bias       : int8 (n,)   - DIR_LONG / DIR_SHORT / DIR_NEUTRAL
        path       : int8 (n,)   - PATH_* (للتشخيص)
        mfe        : float64 (n,) - max favorable excursion (مطلقة)
        mae        : float64 (n,) - max adverse excursion (مطلقة)
        end_idx    : int32 (n,)  - الصف الذي اتخذ فيه القرار (barrier hit أو نهاية الأفق)
    """
    prices = np.asarray(prices, dtype=np.float64)
    atr = np.asarray(atr, dtype=np.float64)
    n = len(prices)

    if n == 0:
        return {
            "bias": np.zeros(0, dtype=np.int8),
            "path": np.zeros(0, dtype=np.int8),
            "mfe": np.zeros(0, dtype=np.float64),
            "mae": np.zeros(0, dtype=np.float64),
            "end_idx": np.zeros(0, dtype=np.int32),
        }

    if len(atr) != n:
        raise ValueError(f"atr length {len(atr)} != prices length {n}")

    if horizons is None:
        horizons = np.full(n, config.horizon_default, dtype=np.int32)
    else:
        horizons = np.asarray(horizons, dtype=np.int32)
        if len(horizons) != n:
            raise ValueError(f"horizons length {len(horizons)} != prices length {n}")

    bias = np.full(n, DIR_NEUTRAL, dtype=np.int8)
    path = np.full(n, PATH_TIMEOUT_NEUTRAL, dtype=np.int8)
    mfe_arr = np.zeros(n, dtype=np.float64)
    mae_arr = np.zeros(n, dtype=np.float64)
    end_idx_arr = np.arange(n, dtype=np.int32)

    tp_mult = float(config.tp_atr_mult)
    sl_mult = float(config.sl_atr_mult)
    ratio = float(config.mfe_mae_ratio)
    min_mult = float(config.min_move_atr_mult)

    for t in range(n - 1):
        entry = prices[t]
        atr_t = max(atr[t], 1e-12)
        upper = entry + tp_mult * atr_t
        lower = entry - sl_mult * atr_t
        min_move = min_mult * atr_t

        h = int(horizons[t])
        if h <= 0:
            continue
        end = min(t + 1 + h, n)
        if end <= t + 1:
            continue

        # Scan forward, tracking MFE/MAE في نفس اللوب لتفادي passes متعددة
        mfe = 0.0
        mae = 0.0
        hit_idx = -1
        hit_direction = DIR_NEUTRAL

        for j in range(t + 1, end):
            move = prices[j] - entry
            if move > mfe:
                mfe = move
            if -move > mae:
                mae = -move

            if move >= upper - entry:
                hit_idx = j
                hit_direction = DIR_LONG
                break
            if move <= lower - entry:
                hit_idx = j
                hit_direction = DIR_SHORT
                break

        mfe_arr[t] = mfe
        mae_arr[t] = mae

        if hit_direction == DIR_LONG:
            bias[t] = DIR_LONG
            path[t] = PATH_TP_LONG
            end_idx_arr[t] = hit_idx
            continue
        if hit_direction == DIR_SHORT:
            bias[t] = DIR_SHORT
            path[t] = PATH_TP_SHORT
            end_idx_arr[t] = hit_idx
            continue

        # Timeout — MFE/MAE fallback
        end_idx_arr[t] = end - 1

        # noise filter: لو الحركتين أصغر من min_move، خليها NEUTRAL
        if mfe < min_move and mae < min_move:
            continue

        # الفائز يجب أن يكون ratio× الخاسر
        if mfe > ratio * mae and mfe >= min_move:
            bias[t] = DIR_LONG
            path[t] = PATH_TIMEOUT_MFE_LONG
        elif mae > ratio * mfe and mae >= min_move:
            bias[t] = DIR_SHORT
            path[t] = PATH_TIMEOUT_MFE_SHORT
        # else: يبقى NEUTRAL

    return {
        "bias": bias,
        "path": path,
        "mfe": mfe_arr,
        "mae": mae_arr,
        "end_idx": end_idx_arr,
    }


def label_distribution(bias: np.ndarray) -> dict[str, float]:
    """نسب التوزيع لـ bias array (للتشخيص السريع)."""
    n = max(len(bias), 1)
    return {
        "long": float((bias == DIR_LONG).sum()) / n,
        "short": float((bias == DIR_SHORT).sum()) / n,
        "neutral": float((bias == DIR_NEUTRAL).sum()) / n,
        "n": int(len(bias)),
    }


__all__ = [
    "DIR_LONG",
    "DIR_SHORT",
    "DIR_NEUTRAL",
    "PATH_TP_LONG",
    "PATH_TP_SHORT",
    "PATH_TIMEOUT_MFE_LONG",
    "PATH_TIMEOUT_MFE_SHORT",
    "PATH_TIMEOUT_NEUTRAL",
    "TripleBarrierConfig",
    "DEFAULT_CONFIG",
    "compute_atr",
    "label_triple_barrier_atr",
    "label_distribution",
]
