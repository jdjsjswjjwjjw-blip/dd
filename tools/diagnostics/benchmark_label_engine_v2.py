"""Benchmark: dynamic_labels (المعطوب) vs label_engine_v2 (Triple Barrier + MFE/MAE).

يحاكي السيناريو في التقرير:
    - سعر يصعد إلى MFE ثم يرجع (mean-revert)
    - dynamic_labels يستخدم future[-1] فقط → NEUTRAL
    - label_engine_v2 يستخدم MFE/MAE → يكتشف LONG

التشغيل:
    python tools/diagnostics/benchmark_label_engine_v2.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from modules.label_engine_v2 import (
    DIR_LONG,
    DIR_NEUTRAL,
    DIR_SHORT,
    TripleBarrierConfig,
    compute_atr,
    label_distribution,
    label_triple_barrier_atr,
)


def legacy_direction_from_future_return(
    future_return: float,
    tick_size: float = 0.0001,
    threshold_ticks: float = 5.0,
) -> int:
    """نسخة طبق الأصل من dynamic_labels.py:705 (المعطوب)."""
    threshold = max(float(tick_size or 0.0), 1e-9) * float(threshold_ticks)
    if future_return > threshold:
        return DIR_LONG
    if future_return < -threshold:
        return DIR_SHORT
    return DIR_NEUTRAL


def legacy_label_forward_scan(
    prices: np.ndarray,
    max_bars_forward: int = 50,
    tick_size: float = 0.0001,
    threshold_ticks: float = 5.0,
) -> np.ndarray:
    """محاكاة الـ legacy: future[-1] - entry فقط."""
    n = len(prices)
    bias = np.full(n, DIR_NEUTRAL, dtype=np.int8)
    for t in range(n - 1):
        entry = prices[t]
        end = min(t + 1 + max_bars_forward, n)
        future = prices[t + 1:end]
        if len(future) == 0:
            continue
        future_return = float(future[-1] - entry)  # 🐛 المشكلة: نقطة النهاية فقط
        bias[t] = legacy_direction_from_future_return(
            future_return, tick_size=tick_size, threshold_ticks=threshold_ticks
        )
    return bias


def make_mean_revert_data(n_samples: int = 12_000, seed: int = 42):
    """يولّد بيانات تشبه السيناريو الموصوف:
       موجات mean-revert صغيرة + قليل من الـ trends الواضحة.

       هذا يحاكي 6B-style intraday data حيث:
       - معظم الوقت السعر يتذبذب ±5p (mean-revert)
       - dynamic_labels يحكم بـ future[-1] على هذا = NEUTRAL غالباً
       - label_engine_v2 يكتشف الحركات الربحية في الوسط
    """
    rng = np.random.RandomState(seed)
    prices = [100.0]
    for _ in range(n_samples - 1):
        # mean-revert random walk بقفزات صغيرة + قليل من trends
        drift = -0.001 * (prices[-1] - 100.0)  # pull-back قوي
        noise = rng.randn() * 0.3  # تذبذب طبيعي
        # 5% من الوقت trend مفاجئ
        if rng.rand() < 0.05:
            noise += rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)
        prices.append(prices[-1] + drift + noise)
    prices = np.array(prices, dtype=np.float64)
    highs = prices + np.abs(rng.randn(n_samples)) * 0.05
    lows = prices - np.abs(rng.randn(n_samples)) * 0.05
    return prices, highs, lows


def print_dist(name: str, bias: np.ndarray) -> None:
    d = label_distribution(bias)
    print(
        f"  {name:30s} → "
        f"NEUTRAL={d['neutral']:6.1%}  "
        f"LONG={d['long']:6.1%}  "
        f"SHORT={d['short']:6.1%}  "
        f"(n={d['n']:,})"
    )


def main():
    print("═" * 78)
    print("Benchmark: dynamic_labels (legacy) vs label_engine_v2 (Triple Barrier)")
    print("═" * 78)

    prices, highs, lows = make_mean_revert_data(n_samples=12_000, seed=42)
    print(f"\n📊 الداتا الصناعية: {len(prices):,} شمعة، mean-revert + trends نادرة")
    print(f"   سعر [min, mean, max]: [{prices.min():.2f}, {prices.mean():.2f}, {prices.max():.2f}]")

    # ── Legacy ────────────────────────────────────────────────────────────────
    print("\n🔴 الـ Legacy (dynamic_labels.py: future[-1] + threshold 5 ticks):")
    legacy_bias = legacy_label_forward_scan(
        prices,
        max_bars_forward=50,
        tick_size=0.01,
        threshold_ticks=5.0,
    )
    print_dist("legacy threshold=5 ticks", legacy_bias)

    legacy_bias_loose = legacy_label_forward_scan(
        prices,
        max_bars_forward=50,
        tick_size=0.01,
        threshold_ticks=1.0,
    )
    print_dist("legacy threshold=1 tick", legacy_bias_loose)

    # ── New v2 ────────────────────────────────────────────────────────────────
    print("\n🟢 الـ Fix الجديد (label_engine_v2: Triple Barrier + MFE/MAE + ATR):")
    atr = compute_atr(highs, lows, prices, window=14)
    print(f"   ATR mean: {atr.mean():.3f}")

    cfg = TripleBarrierConfig(
        tp_atr_mult=2.0,
        sl_atr_mult=2.0,  # symmetric
        mfe_mae_ratio=2.0,
        min_move_atr_mult=0.5,
        horizon_default=20,
    )
    result = label_triple_barrier_atr(prices, atr, config=cfg)
    print_dist("v2 (TB+MFE/MAE, h=20)", result["bias"])

    cfg_aggressive = TripleBarrierConfig(
        tp_atr_mult=1.5,
        sl_atr_mult=1.5,
        mfe_mae_ratio=1.5,
        min_move_atr_mult=0.3,
        horizon_default=20,
    )
    result_agg = label_triple_barrier_atr(prices, atr, config=cfg_aggressive)
    print_dist("v2 (aggressive)", result_agg["bias"])

    # ── Breakdown ─────────────────────────────────────────────────────────────
    print("\n📈 Breakdown للـ v2 (mainстream config):")
    paths = result["path"]
    from modules.label_engine_v2 import (
        PATH_TP_LONG, PATH_TP_SHORT,
        PATH_TIMEOUT_MFE_LONG, PATH_TIMEOUT_MFE_SHORT, PATH_TIMEOUT_NEUTRAL,
    )
    n = len(paths)
    print(f"   PATH_TP_LONG          : {(paths == PATH_TP_LONG).sum():>5,} ({(paths == PATH_TP_LONG).mean():>6.1%})")
    print(f"   PATH_TP_SHORT         : {(paths == PATH_TP_SHORT).sum():>5,} ({(paths == PATH_TP_SHORT).mean():>6.1%})")
    print(f"   PATH_TIMEOUT_MFE_LONG : {(paths == PATH_TIMEOUT_MFE_LONG).sum():>5,} ({(paths == PATH_TIMEOUT_MFE_LONG).mean():>6.1%})")
    print(f"   PATH_TIMEOUT_MFE_SHORT: {(paths == PATH_TIMEOUT_MFE_SHORT).sum():>5,} ({(paths == PATH_TIMEOUT_MFE_SHORT).mean():>6.1%})")
    print(f"   PATH_TIMEOUT_NEUTRAL  : {(paths == PATH_TIMEOUT_NEUTRAL).sum():>5,} ({(paths == PATH_TIMEOUT_NEUTRAL).mean():>6.1%})")

    print("\n💡 الخلاصة:")
    legacy_neutral = label_distribution(legacy_bias)["neutral"]
    v2_neutral = label_distribution(result["bias"])["neutral"]
    print(f"   NEUTRAL %: legacy={legacy_neutral:.1%}  →  v2={v2_neutral:.1%}")
    if legacy_neutral > 0 and v2_neutral > 0:
        reduction = (legacy_neutral - v2_neutral) / legacy_neutral
        print(f"   تقليل NEUTRAL: {reduction:.1%}")

    print("═" * 78)


if __name__ == "__main__":
    main()
