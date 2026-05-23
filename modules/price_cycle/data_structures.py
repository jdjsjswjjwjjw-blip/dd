"""
modules/price_cycle/data_structures.py
────────────────────────────────────────
Data structures للـ macro-scale price representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════════


class SwingType(IntEnum):
    """Swing classification relative to prior swings."""
    HH = 0   # Higher High  — uptrend continuation
    HL = 1   # Higher Low   — uptrend continuation
    LH = 2   # Lower High   — downtrend / weakening
    LL = 3   # Lower Low    — downtrend continuation
    UNDEFINED = 4


class WyckoffPhase(IntEnum):
    """Wyckoff market cycle phases."""
    ACCUMULATION = 0   # smart money buying, range-bound, low
    MARKUP = 1         # uptrend, price rising
    DISTRIBUTION = 2   # smart money selling, range-bound, high
    MARKDOWN = 3       # downtrend, price falling


class TrendMaturity(IntEnum):
    """Trend life-cycle stage."""
    YOUNG = 0      # fresh trend, strong momentum
    MATURE = 1     # established trend, steady
    EXHAUSTED = 2  # weakening, reversal risk high


# ════════════════════════════════════════════════════════════════════════════
# SwingPoint
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SwingPoint:
    """A detected swing high or low."""

    bar_index: int
    price: float
    is_high: bool             # True = swing high, False = swing low
    swing_type: SwingType = SwingType.UNDEFINED
    timestamp_ns: int = 0

    def to_array(self) -> np.ndarray:
        return np.array([
            float(self.bar_index),
            self.price,
            1.0 if self.is_high else 0.0,
            float(self.swing_type),
        ], dtype=np.float64)


# ════════════════════════════════════════════════════════════════════════════
# BarSequence
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class BarSequence:
    """OHLCV bar sequence — the input للـ Price Cycle Model.

    Stores vectorized arrays + derived features.
    """

    timestamps_ns: np.ndarray   # (T,)
    open: np.ndarray            # (T,)
    high: np.ndarray            # (T,)
    low: np.ndarray             # (T,)
    close: np.ndarray           # (T,)
    volume: np.ndarray          # (T,)

    @property
    def n_bars(self) -> int:
        return self.close.shape[0]

    def __len__(self) -> int:
        return self.n_bars

    def __post_init__(self):
        n = self.close.shape[0]
        for name, arr in [
            ("timestamps_ns", self.timestamps_ns), ("open", self.open),
            ("high", self.high), ("low", self.low), ("volume", self.volume),
        ]:
            if arr.shape[0] != n:
                raise ValueError(f"BarSequence '{name}' length {arr.shape[0]} != {n}")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "BarSequence":
        """Construct from OHLCV DataFrame."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"BarSequence.from_dataframe missing: {missing}")

        ts_col = "ts_event" if "ts_event" in df.columns else None
        if ts_col is not None:
            ts = df[ts_col]
            if pd.api.types.is_datetime64_any_dtype(ts):
                ts_ns = ts.astype("int64").to_numpy()
            else:
                ts_ns = pd.to_datetime(ts).astype("int64").to_numpy()
        else:
            ts_ns = np.arange(len(df), dtype=np.int64)

        return cls(
            timestamps_ns=ts_ns,
            open=df["open"].astype(np.float64).to_numpy(),
            high=df["high"].astype(np.float64).to_numpy(),
            low=df["low"].astype(np.float64).to_numpy(),
            close=df["close"].astype(np.float64).to_numpy(),
            volume=df["volume"].astype(np.float64).to_numpy(),
        )

    def compute_derived_features(self, atr_window: int = 14) -> np.ndarray:
        """Compute the 16-dim per-bar feature matrix.

        Features (all causal — no look-ahead):
            [0] open, [1] high, [2] low, [3] close (normalized to prev close)
            [4] log return
            [5] high-low range (normalized)
            [6] body ratio = |close-open| / range
            [7] upper wick ratio
            [8] lower wick ratio
            [9] log volume
            [10] volume z-score (rolling)
            [11] ATR (rolling)
            [12] return z-score (rolling)
            [13] close position in range = (close-low)/(high-low)
            [14] rolling momentum (close vs N-bar-ago)
            [15] rolling volatility-of-volatility

        Returns
        -------
        features : (T, 16) float32
        """
        T = self.n_bars
        eps = 1e-9
        feats = np.zeros((T, 16), dtype=np.float64)

        prev_close = np.concatenate([[self.close[0]], self.close[:-1]])
        rng = np.maximum(self.high - self.low, eps)

        feats[:, 0] = (self.open - prev_close) / np.maximum(prev_close, eps)
        feats[:, 1] = (self.high - prev_close) / np.maximum(prev_close, eps)
        feats[:, 2] = (self.low - prev_close) / np.maximum(prev_close, eps)
        feats[:, 3] = (self.close - prev_close) / np.maximum(prev_close, eps)
        feats[:, 4] = np.log(np.maximum(self.close, eps) / np.maximum(prev_close, eps))
        feats[:, 5] = rng / np.maximum(prev_close, eps)
        feats[:, 6] = np.abs(self.close - self.open) / rng
        feats[:, 7] = (self.high - np.maximum(self.close, self.open)) / rng
        feats[:, 8] = (np.minimum(self.close, self.open) - self.low) / rng
        feats[:, 9] = np.log1p(self.volume)

        # Rolling features (causal)
        feats[:, 10] = _rolling_zscore(self.volume, atr_window)
        feats[:, 11] = _rolling_atr(self.high, self.low, self.close, atr_window)
        feats[:, 12] = _rolling_zscore(feats[:, 4], atr_window)
        feats[:, 13] = (self.close - self.low) / rng
        # Momentum: close vs atr_window bars ago
        shifted = np.concatenate([
            np.full(atr_window, self.close[0]), self.close[:-atr_window]
        ])
        feats[:, 14] = np.log(np.maximum(self.close, eps) / np.maximum(shifted, eps))
        # Vol-of-vol
        feats[:, 15] = _rolling_std(feats[:, 11], atr_window)

        # Sanitize
        feats = np.where(np.isfinite(feats), feats, 0.0)
        return feats.astype(np.float32)

    def summary(self) -> dict[str, Any]:
        return {
            "n_bars": int(self.n_bars),
            "price_range": [float(self.low.min()), float(self.high.max())],
            "total_volume": float(self.volume.sum()),
            "net_return": float(np.log(self.close[-1] / max(self.close[0], 1e-9))),
        }


# ════════════════════════════════════════════════════════════════════════════
# Rolling helpers (causal — no look-ahead)
# ════════════════════════════════════════════════════════════════════════════


def _rolling_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling z-score."""
    s = pd.Series(x)
    mean = s.rolling(window, min_periods=1).mean()
    std = s.rolling(window, min_periods=1).std().fillna(0.0)
    z = (s - mean) / std.replace(0.0, 1.0)
    return z.fillna(0.0).to_numpy()


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(window, min_periods=1).std().fillna(0.0).to_numpy()


def _rolling_atr(high, low, close, window: int) -> np.ndarray:
    """Causal Average True Range."""
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    return pd.Series(tr).rolling(window, min_periods=1).mean().fillna(0.0).to_numpy()
