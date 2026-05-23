"""
modules/price_cycle/swing_analysis.py
──────────────────────────────────────
Causal swing detection + classification + trend maturity estimation.

CRITICAL: all functions are CAUSAL — no look-ahead. A swing at bar t is only
confirmed once price has reversed sufficiently AFTER t. We return the
confirmation index separately so callers never leak future information.

Methods:
    - detect_swings:        ZigZag-style causal swing detection
    - classify_swings:      HH/HL/LH/LL classification
    - trend_maturity:       young/mature/exhausted estimation
    - swing_structure_score: net structural bias [-1, +1]

References:
    - Wyckoff (1931): market structure theory
    - Lo, Mamaysky, Wang (2000): "Foundations of Technical Analysis" — formalized
      swing-point pattern recognition with kernel smoothing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import SwingConfig
from .data_structures import SwingPoint, SwingType, TrendMaturity


# ════════════════════════════════════════════════════════════════════════════
# Causal swing detection
# ════════════════════════════════════════════════════════════════════════════


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """Causal ATR."""
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    out = np.zeros_like(tr)
    acc = 0.0
    for i in range(len(tr)):
        if i == 0:
            acc = tr[0]
        else:
            acc = (acc * (window - 1) + tr[i]) / window
        out[i] = acc
    return out


def detect_swings(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    config: Optional[SwingConfig] = None,
) -> tuple[list[SwingPoint], np.ndarray]:
    """Detect swing highs/lows causally (ZigZag algorithm).

    A swing high is confirmed when price falls by `reversal_threshold` below it.
    A swing low is confirmed when price rises by `reversal_threshold` above it.

    Parameters
    ----------
    high, low, close : (T,) price arrays
    config : SwingConfig

    Returns
    -------
    swings : list[SwingPoint] — confirmed swing points (in order)
    confirmation_index : (n_swings,) — bar at which each swing was CONFIRMED
        (>= swing bar_index). Use this to avoid look-ahead in training.
    """
    if config is None:
        config = SwingConfig()

    T = len(close)
    if T < 3:
        return [], np.zeros(0, dtype=np.int64)

    # Reversal threshold per bar
    if config.reversal_pct > 0:
        threshold = close * (config.reversal_pct / 100.0)
    else:
        atr = _atr(high, low, close, config.atr_window)
        threshold = atr * config.reversal_atr_mult

    swings: list[SwingPoint] = []
    confirmations: list[int] = []

    # State: are we tracking an up-leg (looking for high) or down-leg?
    direction = 0  # 0 = undecided, +1 = up-leg, -1 = down-leg
    extreme_idx = 0
    extreme_price = close[0]

    for t in range(1, T):
        thr = threshold[t]
        if thr <= 0:
            thr = 1e-9

        if direction >= 0:
            # tracking up-leg, watching for new highs
            if high[t] > extreme_price:
                extreme_price = high[t]
                extreme_idx = t
                direction = 1
            elif direction == 1 and (extreme_price - low[t]) >= thr:
                # reversal confirmed → extreme_idx was a swing HIGH
                last_swing_idx = swings[-1].bar_index if swings else -999
                if extreme_idx - last_swing_idx >= config.min_bars_between_swings:
                    swings.append(SwingPoint(
                        bar_index=extreme_idx, price=extreme_price, is_high=True,
                    ))
                    confirmations.append(t)
                # flip to down-leg
                direction = -1
                extreme_price = low[t]
                extreme_idx = t

        if direction <= 0:
            # tracking down-leg, watching for new lows
            if low[t] < extreme_price:
                extreme_price = low[t]
                extreme_idx = t
                direction = -1 if direction != 1 else direction
                if direction != 1:
                    direction = -1
            elif direction == -1 and (high[t] - extreme_price) >= thr:
                # reversal confirmed → extreme_idx was a swing LOW
                last_swing_idx = swings[-1].bar_index if swings else -999
                if extreme_idx - last_swing_idx >= config.min_bars_between_swings:
                    swings.append(SwingPoint(
                        bar_index=extreme_idx, price=extreme_price, is_high=False,
                    ))
                    confirmations.append(t)
                direction = 1
                extreme_price = high[t]
                extreme_idx = t

    return swings, np.array(confirmations, dtype=np.int64)


# ════════════════════════════════════════════════════════════════════════════
# Swing classification (HH/HL/LH/LL)
# ════════════════════════════════════════════════════════════════════════════


def classify_swings(swings: list[SwingPoint]) -> list[SwingPoint]:
    """Classify each swing as HH/HL/LH/LL relative to prior same-type swings.

    Returns a new list with `swing_type` filled.
    """
    if not swings:
        return []

    classified: list[SwingPoint] = []
    last_high: Optional[float] = None
    last_low: Optional[float] = None

    for sw in swings:
        stype = SwingType.UNDEFINED
        if sw.is_high:
            if last_high is not None:
                stype = SwingType.HH if sw.price > last_high else SwingType.LH
            last_high = sw.price
        else:
            if last_low is not None:
                stype = SwingType.HL if sw.price > last_low else SwingType.LL
            last_low = sw.price

        classified.append(SwingPoint(
            bar_index=sw.bar_index,
            price=sw.price,
            is_high=sw.is_high,
            swing_type=stype,
            timestamp_ns=sw.timestamp_ns,
        ))

    return classified


def swing_structure_score(swings: list[SwingPoint], lookback: int = 4) -> float:
    """Net structural bias from recent swings.

    +1.0 = pure uptrend structure (all HH + HL)
    -1.0 = pure downtrend structure (all LH + LL)
     0.0 = mixed / no structure

    Parameters
    ----------
    swings : classified swings (must have swing_type)
    lookback : how many recent swings to consider
    """
    if not swings:
        return 0.0

    recent = swings[-lookback:]
    score = 0.0
    n = 0
    for sw in recent:
        if sw.swing_type in (SwingType.HH, SwingType.HL):
            score += 1.0
            n += 1
        elif sw.swing_type in (SwingType.LH, SwingType.LL):
            score -= 1.0
            n += 1
    return score / max(n, 1)


# ════════════════════════════════════════════════════════════════════════════
# Trend maturity
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class TrendMaturityResult:
    """Trend maturity assessment."""

    maturity: TrendMaturity
    age_bars: int               # bars since trend start
    momentum_decay: float       # [0,1] — how much momentum has faded
    structure_score: float      # from swing_structure_score
    confidence: float

    def to_dict(self) -> dict:
        return {
            "maturity": self.maturity.name,
            "age_bars": int(self.age_bars),
            "momentum_decay": float(self.momentum_decay),
            "structure_score": float(self.structure_score),
            "confidence": float(self.confidence),
        }


def estimate_trend_maturity(
    close: np.ndarray,
    swings: list[SwingPoint],
    young_threshold: int = 20,
    mature_threshold: int = 80,
) -> TrendMaturityResult:
    """Estimate where the current trend is in its life cycle.

    Logic:
        - age = bars since the trend's originating swing
        - momentum_decay = 1 - (recent slope / peak slope)
        - young:     age < young_threshold AND momentum strong
        - exhausted: momentum_decay high OR structure breaking
        - mature:    everything else

    All causal.
    """
    T = len(close)
    if T < 10 or len(swings) < 2:
        return TrendMaturityResult(
            maturity=TrendMaturity.YOUNG, age_bars=0,
            momentum_decay=0.0, structure_score=0.0, confidence=0.0,
        )

    structure = swing_structure_score(swings, lookback=6)

    # Trend origin: last swing that started the current direction
    origin_idx = swings[-2].bar_index if len(swings) >= 2 else 0
    age = T - 1 - origin_idx

    # Momentum: compare recent slope vs early slope
    half = max(2, age // 2)
    recent_slope = (close[-1] - close[-half]) / max(half, 1)
    early_window = close[origin_idx : origin_idx + half + 1]
    if len(early_window) >= 2:
        early_slope = (early_window[-1] - early_window[0]) / max(len(early_window) - 1, 1)
    else:
        early_slope = recent_slope

    if abs(early_slope) > 1e-12:
        momentum_ratio = recent_slope / early_slope
    else:
        momentum_ratio = 1.0
    momentum_decay = float(np.clip(1.0 - momentum_ratio, 0.0, 1.0))

    # Classify
    if age < young_threshold and momentum_decay < 0.3:
        maturity = TrendMaturity.YOUNG
        confidence = 0.8
    elif momentum_decay > 0.6 or abs(structure) < 0.3:
        maturity = TrendMaturity.EXHAUSTED
        confidence = 0.6 + 0.3 * momentum_decay
    else:
        maturity = TrendMaturity.MATURE
        confidence = 0.7

    return TrendMaturityResult(
        maturity=maturity,
        age_bars=int(age),
        momentum_decay=momentum_decay,
        structure_score=float(structure),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )
