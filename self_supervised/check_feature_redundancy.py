"""
self_supervised/check_feature_redundancy.py — يحلل correlation
بين الـ context features و LOB tensor channels لكشف redundancy.

الاستخدام:
    python self_supervised/check_feature_redundancy.py \\
        --features pipeline/day_trading_features.parquet \\
        --lob-tensors pipeline/lob_tensors.npy
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--lob-tensors', required=True)
    p.add_argument('--threshold', type=float, default=0.85,
                  help='Correlation threshold للـ redundancy flag')
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    lob = np.load(args.lob_tensors, mmap_mode='r')  # (N, T, P, C)

    # LOB summary stats per bar (current bar = last in T dimension)
    lob_current = np.asarray(lob[:, -1, :, :])  # (N, P, C)
    lob_summaries = {
        'lob_depth_sum': lob_current[:, :, 0].sum(axis=1),
        'lob_buy_sum':   lob_current[:, :, 1].sum(axis=1) if lob.shape[-1] > 1 else None,
        'lob_sell_sum':  lob_current[:, :, 2].sum(axis=1) if lob.shape[-1] > 2 else None,
        'lob_imbalance': (lob_current[:, :, 1].sum(axis=1) - lob_current[:, :, 2].sum(axis=1))
                         if lob.shape[-1] > 2 else None,
    }
    lob_summaries = {k: v for k, v in lob_summaries.items() if v is not None}

    # Numeric context features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f'Analyzing {len(numeric_cols)} numeric features vs {len(lob_summaries)} LOB summaries')
    print()

    high_corr = []
    for ctx_col in numeric_cols:
        ctx_vals = pd.to_numeric(df[ctx_col], errors='coerce').fillna(0).to_numpy()
        if np.std(ctx_vals) < 1e-9:
            continue
        for lob_name, lob_vals in lob_summaries.items():
            if np.std(lob_vals) < 1e-9:
                continue
            corr = np.corrcoef(ctx_vals, lob_vals)[0, 1]
            if abs(corr) >= args.threshold:
                high_corr.append({
                    'context_feature': ctx_col,
                    'lob_summary': lob_name,
                    'correlation': float(corr),
                })

    if high_corr:
        print(f'⚠️  Found {len(high_corr)} high-correlation pairs (|r|>={args.threshold}):')
        high_corr.sort(key=lambda x: -abs(x['correlation']))
        for pair in high_corr[:20]:
            print(f"  {pair['context_feature']:40s} ↔ {pair['lob_summary']:20s}  "
                  f"r={pair['correlation']:+.3f}")
        if len(high_corr) > 20:
            print(f"  ... and {len(high_corr) - 20} more")
    else:
        print(f'✅ No redundancy detected (|r| < {args.threshold} for all pairs)')

    print()
    # Variance check for seasonal features
    seasonal_cols = [c for c in df.columns if c.startswith((
        'is_', 'session_phase', 'dow_', 'dom', 'woy_'
    ))]
    if seasonal_cols:
        print(f'🗓️  Seasonal features variance check:')
        low_var = []
        for c in seasonal_cols:
            vals = pd.to_numeric(df[c], errors='coerce').fillna(0)
            var = float(vals.var())
            n_unique = int(vals.nunique())
            if n_unique <= 1 or var < 1e-6:
                low_var.append((c, n_unique, var))
        if low_var:
            print(f'  ⚠️  {len(low_var)} seasonal features are near-constant:')
            for c, n_u, v in low_var:
                print(f'    {c}: unique={n_u}, var={v:.2e}')
            print(f'  → فكر في حذفها للـ training set دا (تقلل noise)')
        else:
            print(f'  ✅ All {len(seasonal_cols)} seasonal features have variance')


if __name__ == '__main__':
    main()
