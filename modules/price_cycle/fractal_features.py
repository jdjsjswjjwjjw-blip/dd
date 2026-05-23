"""
modules/price_cycle/fractal_features.py
─────────────────────────────────────────
Multi-timeframe fractal features.

The market exhibits self-similarity across scales (Mandelbrot 1963,
"The Variation of Certain Speculative Prices"). A pattern on the 5min chart
resembles one on the 1h chart. This module quantifies that.

Features:
    - Hurst exponent       : H > 0.5 trending, H < 0.5 mean-reverting, H = 0.5 random
    - Multi-TF alignment   : do 5min/15min/1h agree on direction?
    - Fractal dimension    : roughness of the price path
    - Self-similarity score: correlation of patterns across scales

All causal — no look-ahead.

References:
    - Mandelbrot (1963): self-similarity of price series
    - Hurst (1951): rescaled-range analysis
    - Peters (1994): "Fractal Market Analysis"
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import FractalConfig


# ════════════════════════════════════════════════════════════════════════════
# Hurst exponent
# ════════════════════════════════════════════════════════════════════════════


def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """Estimate the Hurst exponent via the variance-of-lagged-differences method.

    For a series x:
        Var(x_{t+lag} - x_t) ∝ lag^(2H)
    A log-log regression of variance vs lag gives slope = 2H.

    Interpretation:
        H > 0.5  → persistent (trending)
        H = 0.5  → random walk
        H < 0.5  → anti-persistent (mean-reverting)

    Parameters
    ----------
    series : (N,) price or log-price series
    max_lag : maximum lag

    Returns
    -------
    H : float in [0, 1] (clipped)
    """
    series = np.asarray(series, dtype=np.float64)
    n = series.size
    if n < max_lag * 2 or n < 20:
        return 0.5  # insufficient data → assume random walk

    lags = range(2, max_lag)
    tau = []
    valid_lags = []
    for lag in lags:
        diff = series[lag:] - series[:-lag]
        std = np.std(diff)
        if std > 1e-12:
            tau.append(std)
            valid_lags.append(lag)

    if len(valid_lags) < 3:
        return 0.5

    log_lags = np.log(valid_lags)
    log_tau = np.log(tau)
    # slope of log_tau vs log_lags = H
    slope = np.polyfit(log_lags, log_tau, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def rolling_hurst(
    series: np.ndarray, window: int = 100, max_lag: int = 20,
) -> np.ndarray:
    """Causal rolling Hurst exponent.

    Returns (N,) array — Hurst computed on trailing `window` bars.
    """
    series = np.asarray(series, dtype=np.float64)
    n = series.size
    out = np.full(n, 0.5)
    for t in range(n):
        start = max(0, t - window + 1)
        sub = series[start : t + 1]
        if sub.size >= max(20, max_lag * 2):
            out[t] = hurst_exponent(sub, max_lag)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Fractal dimension
# ════════════════════════════════════════════════════════════════════════════


def fractal_dimension(series: np.ndarray) -> float:
    """Estimate fractal dimension via the box-counting / Higuchi-style method.

    D ≈ 2 - H  (for fractional Brownian motion)

    D ≈ 1.5 → random walk
    D → 1.0 → smooth (strong trend)
    D → 2.0 → very rough (choppy)
    """
    H = hurst_exponent(series, max_lag=min(20, len(series) // 4))
    return float(np.clip(2.0 - H, 1.0, 2.0))


# ════════════════════════════════════════════════════════════════════════════
# Multi-timeframe aggregation
# ════════════════════════════════════════════════════════════════════════════


def downsample_ohlc(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray,
    close: np.ndarray, volume: np.ndarray, factor: int,
) -> dict[str, np.ndarray]:
    """Downsample OHLCV bars by an integer factor (e.g. 5min → 15min via factor=3).

    Causal: the last (partial) bucket uses only available bars.
    """
    n = len(close)
    n_buckets = (n + factor - 1) // factor

    ds_open = np.zeros(n_buckets)
    ds_high = np.zeros(n_buckets)
    ds_low = np.zeros(n_buckets)
    ds_close = np.zeros(n_buckets)
    ds_volume = np.zeros(n_buckets)

    for b in range(n_buckets):
        start = b * factor
        end = min(start + factor, n)
        ds_open[b] = open_[start]
        ds_high[b] = high[start:end].max()
        ds_low[b] = low[start:end].min()
        ds_close[b] = close[end - 1]
        ds_volume[b] = volume[start:end].sum()

    return {
        "open": ds_open, "high": ds_high, "low": ds_low,
        "close": ds_close, "volume": ds_volume,
    }


def multi_timeframe_alignment(
    close: np.ndarray,
    timeframe_multipliers: tuple[int, ...] = (1, 4, 16),
    slope_window: int = 10,
) -> dict[str, float]:
    """Measure agreement of trend direction across timeframes.

    Returns
    -------
    dict with:
        alignment_score : [-1, +1] — +1 = all TFs agree bullish
        per_tf_slopes   : list of slopes
        n_aligned       : how many TFs agree with the majority
    """
    slopes = []
    for mult in timeframe_multipliers:
        if mult == 1:
            series = close
        else:
            n = len(close)
            idx = np.arange(0, n, mult)
            series = close[idx]
        if len(series) < 2:
            slopes.append(0.0)
            continue
        w = min(slope_window, len(series))
        recent = series[-w:]
        slope = (recent[-1] - recent[0]) / max(recent[0], 1e-9)
        slopes.append(float(slope))

    signs = [np.sign(s) for s in slopes]
    if not signs:
        return {"alignment_score": 0.0, "per_tf_slopes": [], "n_aligned": 0}

    # Majority direction
    pos = sum(1 for s in signs if s > 0)
    neg = sum(1 for s in signs if s < 0)
    majority = 1 if pos >= neg else -1
    n_aligned = pos if majority > 0 else neg

    alignment_score = majority * (n_aligned / len(signs))

    return {
        "alignment_score": float(alignment_score),
        "per_tf_slopes": slopes,
        "n_aligned": int(n_aligned),
        "n_timeframes": len(timeframe_multipliers),
    }


def compute_fractal_features(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray,
    close: np.ndarray, volume: np.ndarray,
    config: Optional[FractalConfig] = None,
) -> dict[str, np.ndarray]:
    """Compute all fractal features per bar (causal).

    Returns dict of (T,) arrays:
        hurst             : rolling Hurst exponent
        fractal_dim       : 2 - Hurst
        mtf_alignment     : multi-timeframe trend alignment
    """
    if config is None:
        config = FractalConfig()

    log_close = np.log(np.maximum(close, 1e-9))
    T = len(close)

    hurst = rolling_hurst(log_close, config.hurst_window, config.hurst_max_lag)
    fractal_dim = 2.0 - hurst

    # MTF alignment (rolling)
    mtf = np.zeros(T)
    win = config.hurst_window
    for t in range(T):
        start = max(0, t - win + 1)
        sub = close[start : t + 1]
        if sub.size >= 4:
            align = multi_timeframe_alignment(sub, config.timeframe_multipliers)
            mtf[t] = align["alignment_score"]

    return {
        "hurst": hurst,
        "fractal_dim": fractal_dim,
        "mtf_alignment": mtf,
    }
