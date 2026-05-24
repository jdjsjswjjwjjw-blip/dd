"""
self_supervised/inspect_data_quality.py
═══════════════════════════════════════════════════════════
يفحص جودة البيانات بدقة على 3 مستويات:

   1. Raw MBO parquet — outlier prices، NaN، duplicates
   2. features.parquet — OHLCV outliers، ATR شاذة، missing values
   3. LOB tensors — NaN/inf، coverage

ينتج تقرير + اختيارياً يحفظ MBO نظيف.

الاستخدام:
   python self_supervised/inspect_data_quality.py \\
       --mbo /workspace/.../mbo.parquet \\
       --features /workspace/.../day_trading_features.parquet \\
       --lob-tensors /workspace/.../lob_tensors.npy \\
       --output-clean /workspace/.../mbo_clean.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def inspect_mbo(mbo_path: str) -> dict:
    """يفحص MBO parquet."""
    print(f"═══ MBO Inspection: {mbo_path} ═══")
    df = pd.read_parquet(mbo_path)
    n = len(df)
    print(f"  Total rows: {n:,}")
    print(f"  Columns: {list(df.columns)[:10]}...")
    print()

    # Price stats
    if 'price' in df.columns:
        prices = pd.to_numeric(df['price'], errors='coerce')
        nan_count = int(prices.isna().sum())
        zero_count = int((prices == 0).sum())
        neg_count = int((prices < 0).sum())
        finite_prices = prices.dropna()
        print(f"  📊 Price statistics:")
        print(f"     NaN: {nan_count:,} ({nan_count/n*100:.2f}%)")
        print(f"     Zero: {zero_count:,}")
        print(f"     Negative: {neg_count:,}")
        if len(finite_prices) > 0:
            print(f"     min: {finite_prices.min():.6f}")
            print(f"     max: {finite_prices.max():.6f}")
            print(f"     median: {finite_prices.median():.6f}")
            print(f"     p99: {finite_prices.quantile(0.99):.6f}")
            print(f"     p01: {finite_prices.quantile(0.01):.6f}")

        # Suspicious outliers (GBPUSD ≈ 1.20-1.40)
        outliers_high = int((finite_prices > 5.0).sum())
        outliers_low = int((finite_prices < 0.5).sum())
        print(f"  🚨 Suspicious outliers:")
        print(f"     price > 5.0:  {outliers_high:,} ({outliers_high/n*100:.3f}%)")
        print(f"     price < 0.5:  {outliers_low:,} ({outliers_low/n*100:.3f}%)")

        if outliers_high + outliers_low > 0:
            print(f"  ⚠️  Total bad rows: {outliers_high + outliers_low + nan_count:,} "
                  f"({(outliers_high + outliers_low + nan_count)/n*100:.3f}%)")

    # Size stats
    if 'size' in df.columns:
        sizes = pd.to_numeric(df['size'], errors='coerce').dropna()
        print()
        print(f"  📊 Size statistics:")
        print(f"     min: {sizes.min():.0f} | max: {sizes.max():.0f}")
        print(f"     median: {sizes.median():.1f}")

    # Timestamp stats
    if 'ts_event' in df.columns:
        ts = pd.to_datetime(df['ts_event'])
        print()
        print(f"  📅 Timestamps:")
        print(f"     range: {ts.min()} → {ts.max()}")
        print(f"     timezone: {ts.dt.tz}")

    return {
        'n_rows': n,
        'outliers_high': outliers_high if 'price' in df.columns else 0,
        'outliers_low': outliers_low if 'price' in df.columns else 0,
        'nan_count': nan_count if 'price' in df.columns else 0,
    }


def inspect_features(features_path: str) -> dict:
    """يفحص day_trading_features.parquet."""
    print(f"\n═══ Features Inspection: {features_path} ═══")
    df = pd.read_parquet(features_path)
    n = len(df)
    print(f"  Total rows: {n:,}")
    print(f"  Columns: {len(df.columns)}")
    print()

    issues = []

    # OHLCV check
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors='coerce')
        nan = int(vals.isna().sum())
        finite = vals.dropna()
        if len(finite) == 0:
            continue
        outliers_h = int((finite > 5.0).sum())
        outliers_l = int((finite < 0.5).sum())
        marker = '🚨' if (outliers_h + outliers_l) > 0 else '✅'
        print(f"  {marker} {col:7s}: min={finite.min():.4f}, max={finite.max():.4f}, "
              f"NaN={nan}, outliers={outliers_h + outliers_l}")
        if outliers_h + outliers_l > 0:
            issues.append(f"{col}: {outliers_h + outliers_l} outliers")

    # ATR check
    print()
    for col in ['atr_14', 'atr']:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(vals) == 0:
            continue
        outliers = int((vals > 0.05).sum())  # > 500 pip ATR (مستحيل)
        marker = '🚨' if outliers > 0 else '✅'
        print(f"  {marker} {col}: min={vals.min():.6f}, max={vals.max():.6f}, "
              f"median={vals.median():.6f}, outliers(>500pip)={outliers}")
        if outliers > 0:
            issues.append(f"{col}: {outliers} outliers > 500 pip")
        break

    # Volume check
    if 'volume' in df.columns:
        v = pd.to_numeric(df['volume'], errors='coerce').dropna()
        zero_vol = int((v == 0).sum())
        print(f"  ✅ volume: min={v.min():.0f}, max={v.max():.0f}, zero={zero_vol}")

    # Labels check
    print()
    if 'bias_label' in df.columns:
        counts = df['bias_label'].value_counts().sort_index().to_dict()
        n_dir = sum(v for k, v in counts.items() if k != 2)
        print(f"  Labels: LONG={counts.get(0, 0)}, SHORT={counts.get(1, 0)}, "
              f"NEUTRAL={counts.get(2, 0)} | directional={n_dir} ({n_dir/n*100:.1f}%)")

    if 'regime_label' in df.columns:
        regs = df['regime_label'].value_counts().to_dict()
        print(f"  Regimes: {regs}")

    # Check for NaN/Inf across all numeric columns
    print()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    nan_cols = []
    inf_cols = []
    for col in numeric_cols:
        vals = df[col].to_numpy()
        n_nan = int(np.isnan(vals).sum())
        n_inf = int(np.isinf(vals).sum())
        if n_nan > 0:
            nan_cols.append((col, n_nan))
        if n_inf > 0:
            inf_cols.append((col, n_inf))
    if nan_cols:
        print(f"  ⚠️  NaN في {len(nan_cols)} columns:")
        for c, cnt in sorted(nan_cols, key=lambda x: -x[1])[:10]:
            print(f"     {c}: {cnt:,} NaN")
    else:
        print(f"  ✅ No NaN في numeric columns")
    if inf_cols:
        print(f"  🚨 Inf في {len(inf_cols)} columns: {inf_cols[:5]}")

    return {'n_rows': n, 'issues': issues, 'nan_cols': len(nan_cols), 'inf_cols': len(inf_cols)}


def inspect_lob_tensors(lob_path: str) -> dict:
    """يفحص LOB tensors."""
    print(f"\n═══ LOB Tensors Inspection: {lob_path} ═══")
    arr = np.load(lob_path, mmap_mode='r')
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Size on disk: {Path(lob_path).stat().st_size / 1e6:.1f} MB")

    # Sample stats (avoid full load)
    sample_idx = np.linspace(0, arr.shape[0] - 1, min(1000, arr.shape[0])).astype(int)
    sample = np.asarray(arr[sample_idx])
    n_nan = int(np.isnan(sample).sum())
    n_inf = int(np.isinf(sample).sum())
    n_total = sample.size
    print(f"  📊 Sample stats (n={len(sample_idx)} bars):")
    print(f"     min: {sample.min():.4f}, max: {sample.max():.4f}")
    print(f"     mean: {sample.mean():.4f}, std: {sample.std():.4f}")
    print(f"     NaN: {n_nan:,} ({n_nan/n_total*100:.3f}%)")
    print(f"     Inf: {n_inf:,}")

    # Per-channel breakdown
    if sample.ndim == 4:
        print(f"  📊 Per-channel:")
        for c in range(sample.shape[3]):
            ch = sample[:, :, :, c]
            print(f"     ch{c}: min={ch.min():.4f}, max={ch.max():.4f}, "
                  f"mean={ch.mean():.4f}, std={ch.std():.4f}")

    return {
        'shape': arr.shape, 'nan_pct': n_nan / n_total * 100, 'inf_count': n_inf,
    }


def clean_mbo(input_path: str, output_path: str, price_min: float, price_max: float):
    """يفلتر MBO من outliers ويحفظه."""
    print(f"\n═══ Cleaning MBO ═══")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Filter: price ∈ [{price_min}, {price_max}]")

    df = pd.read_parquet(input_path)
    n_before = len(df)

    if 'price' in df.columns:
        prices = pd.to_numeric(df['price'], errors='coerce')
        mask = (prices >= price_min) & (prices <= price_max) & prices.notna()
        df = df[mask].copy()
        n_after = len(df)
        n_removed = n_before - n_after
        print(f"  Before: {n_before:,} rows")
        print(f"  After:  {n_after:,} rows")
        print(f"  Removed: {n_removed:,} ({n_removed/n_before*100:.3f}%)")
    else:
        print(f"  ⚠️  no 'price' column, skipping cleanup")

    df.to_parquet(output_path, index=False)
    print(f"  ✅ Saved")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mbo', help='MBO parquet path')
    p.add_argument('--features', help='day_trading_features.parquet path')
    p.add_argument('--lob-tensors', help='lob_tensors.npy path')
    p.add_argument('--output-clean', help='if set، يفلتر MBO outliers ويحفظ')
    p.add_argument('--price-min', type=float, default=0.5)
    p.add_argument('--price-max', type=float, default=5.0)
    args = p.parse_args()

    if args.mbo:
        inspect_mbo(args.mbo)
    if args.features:
        inspect_features(args.features)
    if args.lob_tensors:
        inspect_lob_tensors(args.lob_tensors)

    if args.output_clean and args.mbo:
        clean_mbo(args.mbo, args.output_clean, args.price_min, args.price_max)

    print()
    print("═" * 60)
    print("التوصيات:")
    print("═" * 60)
    print("  لو OHLCV فيها outliers في features.parquet:")
    print("    1. شغّل clean_mbo (--output-clean)")
    print("    2. أعد prepare_day_trading على mbo_clean.parquet")
    print("    3. أعد SSL على features الجديدة")
    print()
    print("  لو LOB tensors فيها NaN كثيرة:")
    print("    أعد prepare_day_trading مع --sanitize-max-bar-return 0.05")


if __name__ == '__main__':
    main()
