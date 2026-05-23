"""
Diagnose MFE/MAE distribution from a preflight parquet to choose
optimal Sprint 19 thresholds (timeout_mfe_min_move_atr, timeout_mfe_mae_ratio).

الاستخدام:
    python tools/diagnostics/mfe_mae_distribution.py \
        --preflight pipeline_test/day_trading_preflight.parquet

OR على output parquet:
    python tools/diagnostics/mfe_mae_distribution.py \
        --data pipeline_test/features/features_v19.2_5min.parquet
"""
from __future__ import annotations

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd


def diagnose(df: pd.DataFrame, *, max_bars: int = 24) -> None:
    """يحسب MFE/MAE لكل event row على نافذة [i+1, i+max_bars]."""
    if 'is_event' in df.columns:
        df = df[df['is_event'].astype(bool)].copy().reset_index(drop=True)
    print(f"Computing MFE/MAE on {len(df):,} event bars (window={max_bars} bars)...")

    close = pd.to_numeric(df['close'], errors='coerce').to_numpy(dtype=np.float64)
    atr = pd.to_numeric(df.get('atr_14', df.get('atr', 0.001)), errors='coerce').to_numpy(dtype=np.float64)

    n = len(close)
    mfe_atr = np.full(n, np.nan)
    mae_atr = np.full(n, np.nan)

    for i in range(n):
        end = min(i + max_bars, n)
        if end <= i + 1 or atr[i] <= 1e-12:
            continue
        win = close[i + 1: end]
        d = win - close[i]
        mfe = max(0.0, float(np.max(d)))
        mae = max(0.0, float(-np.min(d)))
        mfe_atr[i] = mfe / atr[i]
        mae_atr[i] = mae / atr[i]

    mask = ~np.isnan(mfe_atr)
    mfe_v = mfe_atr[mask]
    mae_v = mae_atr[mask]
    print(f"\nMFE/MAE (in ATR units) over {mask.sum():,} valid event windows:")
    print("─" * 60)
    print(f"  MFE percentiles (×ATR):")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"    p{p:>2d}: {np.percentile(mfe_v, p):.3f}")
    print(f"  MAE percentiles (×ATR):")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"    p{p:>2d}: {np.percentile(mae_v, p):.3f}")

    print("\n📊 Sprint 19 Yield Simulation:")
    print("─" * 60)
    print(f"  {'min_move':>10} {'ratio':>8} {'LONG%':>8} {'SHORT%':>8} {'DIR%':>8} {'NEUT%':>8}")
    for min_move in [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        for ratio in [1.2, 1.3, 1.5, 1.8, 2.0]:
            long_n = int(np.sum((mfe_v > ratio * mae_v) & (mfe_v >= min_move)))
            short_n = int(np.sum((mae_v > ratio * mfe_v) & (mae_v >= min_move)))
            dir_pct = (long_n + short_n) / len(mfe_v) * 100
            neut_pct = 100 - dir_pct
            print(f"  {min_move:>10.1f} {ratio:>8.1f} "
                  f"{long_n/len(mfe_v)*100:>7.1f}% {short_n/len(mfe_v)*100:>7.1f}% "
                  f"{dir_pct:>7.1f}% {neut_pct:>7.1f}%")
        print()

    print("\n💡 Recommendations:")
    print("─" * 60)
    # Find thresholds that give ~30-50% directional
    best = None
    for min_move in [0.2, 0.3, 0.4, 0.5, 0.7]:
        for ratio in [1.2, 1.3, 1.5, 1.8, 2.0]:
            long_n = int(np.sum((mfe_v > ratio * mae_v) & (mfe_v >= min_move)))
            short_n = int(np.sum((mae_v > ratio * mfe_v) & (mae_v >= min_move)))
            dir_pct = (long_n + short_n) / len(mfe_v) * 100
            if 30 <= dir_pct <= 50:
                if best is None or abs(dir_pct - 40) < abs(best[2] - 40):
                    best = (min_move, ratio, dir_pct)
    if best:
        mm, rt, pct = best
        print(f"  Recommended: --timeout-mfe-min-move-atr {mm} --timeout-mfe-mae-ratio {rt}")
        print(f"  Expected directional yield: ~{pct:.0f}%")
    else:
        # Find closest to 30%
        candidates = []
        for min_move in [0.2, 0.3, 0.5, 0.7, 1.0]:
            for ratio in [1.2, 1.3, 1.5, 2.0]:
                long_n = int(np.sum((mfe_v > ratio * mae_v) & (mfe_v >= min_move)))
                short_n = int(np.sum((mae_v > ratio * mfe_v) & (mae_v >= min_move)))
                dir_pct = (long_n + short_n) / len(mfe_v) * 100
                candidates.append((dir_pct, min_move, ratio))
        candidates.sort(key=lambda x: abs(x[0] - 30))
        pct, mm, rt = candidates[0]
        print(f"  Closest to 30%% directional: --timeout-mfe-min-move-atr {mm} "
              f"--timeout-mfe-mae-ratio {rt} (yield={pct:.1f}%)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', help='preflight parquet path')
    p.add_argument('--data', help='full features parquet path')
    p.add_argument('--max-bars', type=int, default=24)
    args = p.parse_args()

    path = args.preflight or args.data
    if not path:
        p.error("Must pass --preflight or --data")
    if os.path.isdir(path):
        import glob
        files = sorted(glob.glob(os.path.join(path, '*.parquet')))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(path)

    print(f"Loaded {len(df):,} rows from {path}")
    diagnose(df, max_bars=args.max_bars)
