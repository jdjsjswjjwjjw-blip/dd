"""
modules/price_cycle/phase_classifier.py
─────────────────────────────────────────
Wyckoff market-cycle phase classification.

The Wyckoff cycle (Richard Wyckoff, 1931):
    ACCUMULATION → MARKUP → DISTRIBUTION → MARKDOWN → (repeat)

    ACCUMULATION:  range-bound at lows, smart money buying, volume drying
    MARKUP:        uptrend, expanding range, rising volume on up-bars
    DISTRIBUTION:  range-bound at highs, smart money selling
    MARKDOWN:      downtrend, falling prices

This module provides a rules-based classifier (transparent, no training)
PLUS a feature extractor that a learned model can refine.

All causal. No look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import PhaseConfig
from .data_structures import WyckoffPhase


@dataclass
class PhaseAssessment:
    """Phase classification result for a single point in time."""

    phase: WyckoffPhase
    phase_probs: np.ndarray       # (4,) — soft probabilities
    cycle_position: float         # [0, 1] — position within full cycle
    confidence: float
    features: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.name,
            "phase_probs": self.phase_probs.tolist(),
            "cycle_position": float(self.cycle_position),
            "confidence": float(self.confidence),
            "features": dict(self.features),
        }


def _causal_rolling(x: np.ndarray, window: int, fn: str = "mean") -> np.ndarray:
    s = pd.Series(x)
    r = s.rolling(window, min_periods=1)
    if fn == "mean":
        return r.mean().to_numpy()
    if fn == "std":
        return r.std().fillna(0.0).to_numpy()
    if fn == "min":
        return r.min().to_numpy()
    if fn == "max":
        return r.max().to_numpy()
    raise ValueError(fn)


def compute_phase_features(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    config: Optional[PhaseConfig] = None,
) -> dict[str, np.ndarray]:
    """Compute Wyckoff phase features per bar (all causal).

    Returns dict of (T,) arrays:
        trend_slope        : normalized rolling slope of close
        range_ratio        : current range vs historical (range-bound?)
        price_percentile   : where price sits in rolling range [0,1]
        volume_trend       : rising vs falling volume
        volatility_state   : expanding vs contracting
        effort_result      : volume vs price-progress divergence (Wyckoff)
    """
    if config is None:
        config = PhaseConfig()

    W = config.feature_window
    T = len(close)
    eps = 1e-9

    # Trend slope (normalized)
    sma = _causal_rolling(close, W, "mean")
    slope = np.zeros(T)
    for t in range(1, T):
        lookback = min(t, W)
        slope[t] = (close[t] - close[t - lookback]) / max(lookback * sma[t], eps)

    # Range ratio: recent range vs long range
    roll_high = _causal_rolling(high, W, "max")
    roll_low = _causal_rolling(low, W, "min")
    long_high = _causal_rolling(high, W * 3, "max")
    long_low = _causal_rolling(low, W * 3, "min")
    range_ratio = (roll_high - roll_low) / np.maximum(long_high - long_low, eps)

    # Price percentile in rolling range
    price_pct = (close - roll_low) / np.maximum(roll_high - roll_low, eps)
    price_pct = np.clip(price_pct, 0.0, 1.0)

    # Volume trend
    vol_sma_short = _causal_rolling(volume, W // 3 + 1, "mean")
    vol_sma_long = _causal_rolling(volume, W, "mean")
    volume_trend = (vol_sma_short - vol_sma_long) / np.maximum(vol_sma_long, eps)

    # Volatility state (expanding/contracting)
    ret = np.concatenate([[0.0], np.diff(np.log(np.maximum(close, eps)))])
    vol_short = _causal_rolling(np.abs(ret), W // 3 + 1, "mean")
    vol_long = _causal_rolling(np.abs(ret), W, "mean")
    volatility_state = (vol_short - vol_long) / np.maximum(vol_long, eps)

    # Effort vs Result (Wyckoff): high volume + low price progress = absorption
    price_progress = np.abs(slope)
    effort = vol_sma_short / np.maximum(vol_sma_long, eps)
    effort_result = effort - price_progress / np.maximum(price_progress.std() + eps, eps)

    return {
        "trend_slope": np.nan_to_num(slope),
        "range_ratio": np.nan_to_num(range_ratio),
        "price_percentile": np.nan_to_num(price_pct),
        "volume_trend": np.nan_to_num(volume_trend),
        "volatility_state": np.nan_to_num(volatility_state),
        "effort_result": np.nan_to_num(effort_result),
    }


def classify_phase_rulesbased(
    features_at_t: dict[str, float],
) -> PhaseAssessment:
    """Rules-based Wyckoff phase classification (transparent baseline).

    Decision logic:
        MARKUP:        positive slope + price not at extreme low
        MARKDOWN:      negative slope + price not at extreme high
        ACCUMULATION:  flat slope + low price percentile + volume drying
        DISTRIBUTION:  flat slope + high price percentile + volume drying

    Parameters
    ----------
    features_at_t : dict with keys from compute_phase_features (scalar values)

    Returns
    -------
    PhaseAssessment
    """
    slope = features_at_t.get("trend_slope", 0.0)
    price_pct = features_at_t.get("price_percentile", 0.5)
    vol_trend = features_at_t.get("volume_trend", 0.0)
    range_ratio = features_at_t.get("range_ratio", 1.0)

    # ── Normalize slope to a comparable [-1, +1] scale ──────────────────
    # slope is a tiny normalized number (~0.001-0.005 for clear trends).
    # tanh(slope × gain) maps it to a usable scale comparable with price_pct.
    slope_norm = float(np.tanh(slope * 500.0))   # ∈ [-1, +1]
    trend_strength = abs(slope_norm)              # ∈ [0, 1]
    ranging_strength = 1.0 - trend_strength       # ∈ [0, 1]

    # ── Soft scores per phase ────────────────────────────────────────────
    # Trending phases win when trend is strong;
    # ranging phases win when trend is weak (gated by ranging_strength).
    scores = np.zeros(4)

    # MARKUP: positive trend
    scores[WyckoffPhase.MARKUP] = max(0.0, slope_norm) * 2.0
    # MARKDOWN: negative trend
    scores[WyckoffPhase.MARKDOWN] = max(0.0, -slope_norm) * 2.0
    # ACCUMULATION: ranging + low price + volume drying
    scores[WyckoffPhase.ACCUMULATION] = ranging_strength * (
        (1.0 - price_pct) + (0.3 if vol_trend < 0 else 0.0)
    )
    # DISTRIBUTION: ranging + high price + volume drying
    scores[WyckoffPhase.DISTRIBUTION] = ranging_strength * (
        price_pct + (0.3 if vol_trend < 0 else 0.0)
    )

    # Softmax → probabilities (temperature 0.5 sharpens)
    scores = (scores - scores.max()) / 0.5
    exp = np.exp(scores)
    probs = exp / exp.sum()

    phase = WyckoffPhase(int(np.argmax(probs)))
    confidence = float(probs.max())

    # Cycle position: ACCUM(0) → MARKUP(0.25) → DIST(0.5) → MARKDOWN(0.75)
    cycle_map = {
        WyckoffPhase.ACCUMULATION: 0.0,
        WyckoffPhase.MARKUP: 0.25,
        WyckoffPhase.DISTRIBUTION: 0.5,
        WyckoffPhase.MARKDOWN: 0.75,
    }
    cycle_position = cycle_map[phase]

    return PhaseAssessment(
        phase=phase,
        phase_probs=probs,
        cycle_position=cycle_position,
        confidence=confidence,
        features=dict(features_at_t),
    )


def classify_phase_sequence(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    config: Optional[PhaseConfig] = None,
) -> list[PhaseAssessment]:
    """Classify the Wyckoff phase at every bar (causal).

    Returns list of PhaseAssessment, one per bar.
    """
    feats = compute_phase_features(high, low, close, volume, config)
    T = len(close)
    assessments = []
    for t in range(T):
        feats_t = {k: float(v[t]) for k, v in feats.items()}
        assessments.append(classify_phase_rulesbased(feats_t))
    return assessments
