"""
modules/deeplob_v7ch.py — Phase 5: DeepLOB من 4 إلى 7 channels.
══════════════════════════════════════════════════════════════════════

الفكرة (من التقرير 4.2 تحسين 3):
  DeepLOB الأصلي: 4 channels (depth + buy_fp + sell_fp + price)
  V19.2 enhanced: 7 channels:
    Ch 0: Depth (L2 orderbook)
    Ch 1: Buy footprint (aggressor flow)
    Ch 2: Sell footprint
    Ch 3: Iceberg footprint        ⭐ جديد
    Ch 4: Wall persistence          ⭐ جديد
    Ch 5: Informed pressure         ⭐ جديد
    Ch 6: Depth imbalance           ⭐ جديد

  CNN يرى "المؤسسات المخفية" - signals iceberg + wall persistence
  مباشرة كـ image channels بدل feature concatenation.

API:
    build_7ch_tensor(df, time_steps=50, price_levels=20) -> np.ndarray
        shape: (n_samples, T, P, 7)

    DeepLOBTensor7Channel: incremental builder بنفس واجهة LOBTensorBuilder

ملاحظة: هذا layer-ahead بسيط - لا يدرّب model فعلياً.
       الـ training يتم في train_v19.py بعد generate الـ tensors.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# الـ channels الإضافية تأتي من V19.2 simulators outputs
EXTRA_CHANNELS = [
    "sim_iceberg_strength",   # ch 3: iceberg footprint
    "sim_wall_persist",        # ch 4: wall persistence
    "sim_informed_prob",       # ch 5: informed pressure
    "sim_depth_imbalance",     # ch 6: depth imbalance (signed)
]

N_TIME_STEPS = 50
N_PRICE_LEVELS = 20
N_CHANNELS_V7 = 7   # 3 (base) + 4 (V19.2 extras)


@dataclass(frozen=True)
class TensorShape:
    time_steps: int = N_TIME_STEPS
    price_levels: int = N_PRICE_LEVELS
    channels: int = N_CHANNELS_V7


def build_7ch_tensor(
    df: pd.DataFrame,
    time_steps: int = N_TIME_STEPS,
    price_levels: int = N_PRICE_LEVELS,
    extra_channels: list[str] = EXTRA_CHANNELS,
    skip_missing: bool = True,
) -> np.ndarray:
    """يبني tensor بـ 7 channels من DataFrame مع MBP-10 + V19.2 sim_* outputs.

    Parameters
    ----------
    df : DataFrame مع:
        - bid_sz_00..bid_sz_09  (10 bid sizes)
        - ask_sz_00..ask_sz_09  (10 ask sizes)
        - extra_channels (sim_* columns)
    time_steps : T - طول النافذة الزمنية
    price_levels : P - 20 (10 bid + 10 ask)
    extra_channels : أسماء الـ 4 V19.2 channels الإضافية
    skip_missing : لو True يملأ missing channels بـ 0

    Returns
    -------
    tensor shape (n_samples, T, P, C=7).
        n_samples = len(df) - T + 1 (sliding window)
    """
    n = len(df)
    if n < time_steps:
        return np.zeros((0, time_steps, price_levels, N_CHANNELS_V7), dtype=np.float32)

    L = price_levels // 2

    # ── Channel 0: depth (bid sizes + ask sizes mirrored) ───────────────
    depth = np.zeros((n, price_levels), dtype=np.float32)
    for i in range(L):
        depth[:, L - 1 - i] = pd.to_numeric(
            df.get(f"bid_sz_{i:02d}", 0), errors="coerce"
        ).fillna(0).to_numpy()
        depth[:, L + i] = pd.to_numeric(
            df.get(f"ask_sz_{i:02d}", 0), errors="coerce"
        ).fillna(0).to_numpy()

    # ── Channel 1: buy footprint ────────────────────────────────────────
    buy_fp = np.zeros((n, price_levels), dtype=np.float32)
    if "buy_volume" in df.columns:
        bv = pd.to_numeric(df["buy_volume"], errors="coerce").fillna(0).to_numpy()
        buy_fp[:, L:] = bv[:, None] / max(L, 1)  # spread evenly
    elif "trade_size" in df.columns and "side" in df.columns:
        ts_vals = pd.to_numeric(df["trade_size"], errors="coerce").fillna(0).to_numpy()
        buy_mask = (df["side"] == "B").to_numpy()
        buy_fp[buy_mask, L:] = ts_vals[buy_mask, None] / max(L, 1)
    # else: zeros

    # ── Channel 2: sell footprint ───────────────────────────────────────
    sell_fp = np.zeros((n, price_levels), dtype=np.float32)
    if "sell_volume" in df.columns:
        sv = pd.to_numeric(df["sell_volume"], errors="coerce").fillna(0).to_numpy()
        sell_fp[:, :L] = sv[:, None] / max(L, 1)
    elif "trade_size" in df.columns and "side" in df.columns:
        ts_vals = pd.to_numeric(df["trade_size"], errors="coerce").fillna(0).to_numpy()
        sell_mask = (df["side"] == "S").to_numpy()
        sell_fp[sell_mask, :L] = ts_vals[sell_mask, None] / max(L, 1)

    # ── Channels 3-6: V19.2 extras (broadcast scalar per row) ───────────
    extras = np.zeros((n, price_levels, len(extra_channels)), dtype=np.float32)
    for k, col in enumerate(extra_channels):
        if col not in df.columns:
            if not skip_missing:
                raise KeyError(f"missing channel column: {col}")
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy().astype(np.float32)
        # broadcast إلى كل levels
        extras[:, :, k] = vals[:, None]

    # ── Stack into (n, P, 7) ─────────────────────────────────────────────
    stacked = np.stack(
        [depth, buy_fp, sell_fp, *(extras[:, :, k] for k in range(len(extra_channels)))],
        axis=-1,
    )  # (n, P, 7)

    # ── Sliding window إلى (n_samples, T, P, 7) ──────────────────────────
    n_samples = n - time_steps + 1
    out = np.zeros((n_samples, time_steps, price_levels, N_CHANNELS_V7), dtype=np.float32)
    for i in range(n_samples):
        out[i] = stacked[i:i + time_steps]
    return out


def estimate_tensor_size_mb(
    n_samples: int,
    time_steps: int = N_TIME_STEPS,
    price_levels: int = N_PRICE_LEVELS,
    channels: int = N_CHANNELS_V7,
    dtype_size: int = 4,
) -> float:
    """تقدير حجم الـ tensor في الـ memory (MB)."""
    bytes_total = n_samples * time_steps * price_levels * channels * dtype_size
    return bytes_total / (1024 ** 2)


def channel_names() -> list[str]:
    """أسماء الـ 7 channels للتشخيص."""
    return [
        "depth",            # 0
        "buy_footprint",    # 1
        "sell_footprint",   # 2
        *EXTRA_CHANNELS,    # 3-6
    ]


__all__ = [
    "EXTRA_CHANNELS",
    "N_TIME_STEPS",
    "N_PRICE_LEVELS",
    "N_CHANNELS_V7",
    "TensorShape",
    "build_7ch_tensor",
    "estimate_tensor_size_mb",
    "channel_names",
]
