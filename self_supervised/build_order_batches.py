"""
ssl/build_order_batches.py
═══════════════════════════════════════════════════════════
Phase A0: pre-build OrderBatch tensors من raw MBO data.

يحل مشكلة فقدان البيانات في الـ LOB tensor aggregation:
    LOB tensor الحالي (T,P,C) يفقد order IDs و cancellation patterns و
    iceberg signals.

    هذا الـ script يبني (N_bars, T=50, N_orders=200, F=7) من MBO الخام،
    محتفظاً بـ:
        - order_id (لكشف iceberg via repeat detection)
        - action (T=trade, A=add, C=cancel)
        - size, price, side
        - time arrivals دقيقة
        - iceberg signals (repeat counts، level hits)

الـ 7 features لكل order:
    [0] side            (0=bid, 1=ask, 2=neutral)
    [1] action_type     (0=T trade, 1=A add, 2=C cancel, 3=F fill)
    [2] log_size        (log1p of order size)
    [3] price_distance  (signed ticks from mid)
    [4] time_offset_ms  (ms within bar window)
    [5] repeat_count    (iceberg signal: how many times this order_id appeared)
    [6] level_hits      (iceberg signal: how many orders at this price level)

الاستخدام:
    python ssl/build_order_batches.py \\
        --mbo mbo_data.parquet \\
        --features pipeline/day_trading_features.parquet \\
        --output checkpoints/order_batches \\
        --freq 5min --lookback-bars 50 --n-orders 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Action code mapping
ACTION_MAP = {'T': 0, 'A': 1, 'C': 2, 'F': 3, 'M': 1, 'R': 2}
SIDE_MAP = {'A': 1, 'B': 0, 'N': 2}  # A=ask, B=bid, N=neutral


def _build_bar_orders(
    mbo_bar: pd.DataFrame,
    mid_price: float,
    bar_start_ns: int,
    n_orders_max: int = 200,
    tick_size: float = 0.0001,
    bar_duration_ns: int = 900_000_000_000,  # 15min default
) -> tuple[np.ndarray, np.ndarray]:
    """
    يبني OrderBatch لـ bar واحدة من MBO ticks في ذلك الـ window.

    Input:
        mbo_bar: MBO rows داخل bar window
        mid_price: mid price للـ bar
        bar_start_ns: bar start timestamp
        n_orders_max: max orders to keep (latest if more)

    Output:
        orders: (n_orders_max, 7) float32
        mask: (n_orders_max,) bool — True = valid order
    """
    n_actual = len(mbo_bar)
    if n_actual == 0:
        return (
            np.zeros((n_orders_max, 7), dtype=np.float32),
            np.zeros(n_orders_max, dtype=bool),
        )

    # خد آخر n_orders_max لو أكتر
    if n_actual > n_orders_max:
        mbo_bar = mbo_bar.iloc[-n_orders_max:]
        n_actual = n_orders_max

    # Vectorized extraction
    sides = mbo_bar['side'].astype(str).map(SIDE_MAP).fillna(2).to_numpy(dtype=np.float32)
    actions = mbo_bar['action'].astype(str).map(ACTION_MAP).fillna(0).to_numpy(dtype=np.float32)
    sizes = pd.to_numeric(mbo_bar['size'], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
    prices = pd.to_numeric(mbo_bar['price'], errors='coerce').fillna(mid_price).to_numpy(dtype=np.float32)

    # Time offset normalized [0, 1] of bar duration (NOT raw ms — كان يسبب NaN gradients)
    ts_ns = pd.to_datetime(mbo_bar['ts_event']).astype('datetime64[ns]').astype(np.int64).to_numpy()
    offset_ns = (ts_ns - bar_start_ns).astype(np.float64)
    time_offset_norm = np.clip(offset_ns / max(bar_duration_ns, 1), 0.0, 1.0).astype(np.float32)

    # Price distance from mid (in ticks)
    if tick_size > 0:
        price_dist_ticks = ((prices - mid_price) / tick_size).astype(np.float32)
    else:
        price_dist_ticks = np.zeros(n_actual, dtype=np.float32)
    # Clip extreme outliers
    price_dist_ticks = np.clip(price_dist_ticks, -1000, 1000)

    # log_size
    log_size = np.log1p(np.abs(sizes)).astype(np.float32)

    # ── Iceberg signals ──
    # repeat_count: how many times each order_id appears in this bar
    if 'order_id' in mbo_bar.columns:
        oid = mbo_bar['order_id'].astype(str).to_numpy()
        oid_counts = pd.Series(oid).value_counts().to_dict()
        repeat_count = np.array([oid_counts.get(o, 1) for o in oid], dtype=np.float32)
        # Clip extreme
        repeat_count = np.clip(repeat_count, 1, 50)
    else:
        repeat_count = np.ones(n_actual, dtype=np.float32)

    # level_hits: how many orders at the same price level (within 1 tick)
    if tick_size > 0:
        price_buckets = np.round(prices / tick_size).astype(np.int64)
        bucket_counts = pd.Series(price_buckets).value_counts().to_dict()
        level_hits = np.array([bucket_counts.get(b, 1) for b in price_buckets], dtype=np.float32)
        level_hits = np.clip(level_hits, 1, 100)
    else:
        level_hits = np.ones(n_actual, dtype=np.float32)

    # Assemble: (n_actual, 7)
    features = np.zeros((n_orders_max, 7), dtype=np.float32)
    features[:n_actual, 0] = sides
    features[:n_actual, 1] = actions
    features[:n_actual, 2] = log_size
    features[:n_actual, 3] = price_dist_ticks
    features[:n_actual, 4] = time_offset_norm   # ∈ [0, 1] بدل ms
    features[:n_actual, 5] = np.log1p(repeat_count - 1)  # log(repeat_count), 0 for unique
    features[:n_actual, 6] = np.log1p(level_hits - 1)    # log(level_hits)

    mask = np.zeros(n_orders_max, dtype=bool)
    mask[:n_actual] = True

    return features, mask


def build_order_batches(
    mbo_path: str,
    features_parquet: str,
    output_dir: str,
    *,
    freq: str = '5min',
    lookback_bars: int = 50,
    n_orders_max: int = 200,
    tick_size: float = 0.0001,
) -> None:
    """يبني OrderBatch tensors للـ pipeline كاملة."""
    print(f"═══ Building OrderBatches from raw MBO ═══")
    print(f"  MBO: {mbo_path}")
    print(f"  Features: {features_parquet}")
    print(f"  Output: {output_dir}")
    print(f"  lookback_bars={lookback_bars} | n_orders={n_orders_max}")
    print()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load MBO ──
    print(f"📂 Loading MBO data...")
    df_mbo = pd.read_parquet(mbo_path)
    df_mbo['ts_event'] = pd.to_datetime(df_mbo['ts_event'])
    # Normalize timezone (مهم: لو tz mismatch بين MBO و bars، الـ groupby keys لا تطابق)
    if df_mbo['ts_event'].dt.tz is not None:
        df_mbo['ts_event'] = df_mbo['ts_event'].dt.tz_convert('UTC').dt.tz_localize(None)
    df_mbo = df_mbo.sort_values('ts_event').reset_index(drop=True)
    print(f"   {len(df_mbo):,} MBO ticks | tz=naive")

    # ── Load bars features ──
    print(f"📂 Loading bars features...")
    df_bars = pd.read_parquet(features_parquet)
    df_bars['ts_event'] = pd.to_datetime(df_bars['ts_event'])
    # Same normalization
    if df_bars['ts_event'].dt.tz is not None:
        df_bars['ts_event'] = df_bars['ts_event'].dt.tz_convert('UTC').dt.tz_localize(None)
    df_bars = df_bars.sort_values('ts_event').reset_index(drop=True)
    n_bars = len(df_bars)

    # ── Auto-detect freq من bars (override الـ user input لو لا يطابق) ──
    diffs_sec = df_bars['ts_event'].diff().dt.total_seconds().dropna()
    if len(diffs_sec) > 10:
        median_diff = float(diffs_sec.median())
        detected_freq = None
        if median_diff <= 75:
            detected_freq = '1min'
        elif median_diff <= 360:
            detected_freq = '5min'
        elif median_diff <= 1080:
            detected_freq = '15min'
        elif median_diff <= 2400:
            detected_freq = '30min'
        elif median_diff <= 4200:
            detected_freq = '1h'
        if detected_freq and detected_freq != freq:
            print(f"   ⚠️  Detected bar freq = {detected_freq} (overriding --freq {freq})")
            freq = detected_freq
    print(f"   {n_bars:,} bars @ {freq}")

    # ── Pre-compute bar windows ──
    # Each bar's lookback = lookback_bars bars BEFORE bar_ts
    freq_td = pd.Timedelta(freq)
    bar_window_duration = freq_td * lookback_bars

    # ── Allocate output tensors ──
    # Shape: (n_bars, T=lookback_bars, N_orders, 7)
    # But this might be HUGE for large n_bars. Use float32 and disk-friendly format.
    print(f"\n⚙️  Allocating tensors...")
    estimated_size_gb = n_bars * lookback_bars * n_orders_max * 7 * 4 / 1e9
    print(f"   Estimated size: {estimated_size_gb:.2f} GB")

    if estimated_size_gb > 20:
        print(f"   ⚠️  Large size — using chunked mmap output")

    order_features = np.zeros(
        (n_bars, lookback_bars, n_orders_max, 7), dtype=np.float32,
    )
    order_masks = np.zeros(
        (n_bars, lookback_bars, n_orders_max), dtype=bool,
    )

    # ── Build per-bar OrderBatches ──
    print(f"\n🔄 Building OrderBatches for {n_bars:,} bars...")

    # Optimization: pre-group MBO by bar boundaries
    df_mbo['bar_floor'] = df_mbo['ts_event'].dt.floor(freq)

    # Build a dict: bar_floor → DataFrame
    print(f"   Grouping MBO by bar...")
    mbo_by_bar = {k: v for k, v in df_mbo.groupby('bar_floor')}
    print(f"   {len(mbo_by_bar)} unique bar groups")

    log_every = max(n_bars // 20, 100)
    for bi in range(n_bars):
        bar_ts = df_bars['ts_event'].iloc[bi]
        bar_close = float(df_bars['close'].iloc[bi]) if 'close' in df_bars.columns else 0.0

        # For each lookback bar (most recent first)
        for lag in range(lookback_bars):
            src_bar_ts = bar_ts - freq_td * (lookback_bars - 1 - lag)
            src_bar_floor = src_bar_ts.floor(freq)
            mbo_in_bar = mbo_by_bar.get(src_bar_floor)
            if mbo_in_bar is None or len(mbo_in_bar) == 0:
                continue

            bar_start_ns = int(src_bar_floor.value)
            mid_price = float(mbo_in_bar['price'].mean()) if len(mbo_in_bar) > 0 else bar_close

            orders, mask = _build_bar_orders(
                mbo_in_bar, mid_price, bar_start_ns,
                n_orders_max=n_orders_max, tick_size=tick_size,
                bar_duration_ns=int(freq_td.total_seconds() * 1_000_000_000),
            )
            order_features[bi, lag] = orders
            order_masks[bi, lag] = mask

        if (bi + 1) % log_every == 0:
            print(f"   processed {bi+1:,}/{n_bars:,} bars ({100*(bi+1)/n_bars:.1f}%)")

    # ── Save ──
    print(f"\n💾 Saving tensors...")
    features_path = Path(output_dir) / 'order_features.npy'
    masks_path = Path(output_dir) / 'order_masks.npy'
    meta_path = Path(output_dir) / 'order_batches_meta.json'

    np.save(features_path, order_features)
    np.save(masks_path, order_masks)

    meta = {
        'n_bars': int(n_bars),
        'lookback_bars': int(lookback_bars),
        'n_orders_max': int(n_orders_max),
        'features_per_order': 7,
        'feature_names': [
            'side', 'action_type', 'log_size', 'price_dist_ticks',
            'time_offset_ms', 'log_repeat_count', 'log_level_hits',
        ],
        'iceberg_features': {
            'log_repeat_count': 'log(1+order_id repeat count within bar) — iceberg signal',
            'log_level_hits': 'log(1+orders at same price level) — wall/iceberg signal',
        },
        'mbo_source': mbo_path,
        'features_source': features_parquet,
        'freq': freq,
        'tick_size': float(tick_size),
        'sparsity': {
            'mean_orders_per_bar': float(order_masks.sum(axis=-1).mean()),
            'fill_ratio': float(order_masks.mean()),
        },
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"   ✅ order_features: {features_path} ({order_features.nbytes / 1e9:.2f} GB)")
    print(f"   ✅ order_masks: {masks_path} ({order_masks.nbytes / 1e9:.2f} GB)")
    print(f"   ✅ metadata: {meta_path}")
    print()
    print(f"📊 Statistics:")
    print(f"   Mean orders per bar window: {meta['sparsity']['mean_orders_per_bar']:.1f}")
    print(f"   Fill ratio: {meta['sparsity']['fill_ratio']:.1%}")
    print()
    print(f"✅ Done — OrderBatch tensors ready for SSL pretraining")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mbo', required=True, help='Raw MBO parquet path')
    p.add_argument('--features', required=True, help='day_trading_features.parquet')
    p.add_argument('--output', default='checkpoints/order_batches')
    p.add_argument('--freq', default='5min')
    p.add_argument('--lookback-bars', type=int, default=50)
    p.add_argument('--n-orders', type=int, default=200,
                  help='Max orders per bar (latest if more)')
    p.add_argument('--tick-size', type=float, default=0.0001,
                  help='Tick size for price normalization (0.0001 for 6B)')
    args = p.parse_args()

    build_order_batches(
        args.mbo, args.features, args.output,
        freq=args.freq, lookback_bars=args.lookback_bars,
        n_orders_max=args.n_orders, tick_size=args.tick_size,
    )


if __name__ == '__main__':
    main()
