"""
self_supervised/analyze_zone_edges.py
═══════════════════════════════════════════════════════════
Zone-level edge discovery — NO TRAINING REQUIRED.

Pure descriptive statistics on the existing directional labels +
forward returns. Slices the directional bars by:

    1. Session       (Asia / London / Overlap / NY / Off-hours)
    2. Hour of day   (UTC, 24 bins)
    3. Day of week   (Mon-Fri)
    4. Regime        (trending / ranging / volatile / low_liquidity)
    5. Session × Regime (cross-tab)

For each cell reports:
    n_directional        : count of LONG+SHORT setups
    long_share / short_share
    accuracy             : fraction where label matched outcome (=1.0 if
                           labels are self-consistent)
    avg_return_pips      : mean signed return per trade (signed by label)
    median_return_pips
    sharpe               : avg / std (non-annualized — comparison across
                           zones at the same bar-duration)
    sharpe_ci_95         : bootstrap percentile CI
    win_rate             : fraction with return > 0 after spread
    edge_score           : avg_return / atr_median (volatility-normalized)

Filter to "promising edges" where:
    n >= 10 (statistical floor)
    CI_lower_bound(sharpe) > 0
    win_rate > 0.52

Output:
    {output}/zone_edges_report.md       — human-readable
    {output}/zone_edges.json            — machine-readable
    {output}/zone_edges_topk.csv        — sorted table of best edges

Usage:
    python self_supervised/analyze_zone_edges.py \\
        --features pipeline_15min_v2/day_trading_features.parquet \\
        --output checkpoints/ssl_15min_v2/zone_edges \\
        --return-col forward_return \\
        --horizon-bars 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# Returns + edge computation
# ════════════════════════════════════════════════════════════════════════════

def _compute_signed_returns(
    df: pd.DataFrame,
    return_col: str = 'forward_return',
    horizon_bars: int = 12,
    spread_pips: float = 2.0,
    tick_size: float = 0.0001,
) -> pd.DataFrame:
    """Build signed return column: positive iff trade in direction wins.

    Tries the following return columns in order:
        1. user-specified --return-col (default: forward_return)
        2. fwd_ret_clean
        3. computes from close prices (close[i+horizon] / close[i] - 1)

    Then signs by bias_label: LONG (0) keeps sign, SHORT (1) flips.
    Subtracts spread cost in pips.
    """
    df = df.copy()

    # Source returns
    src = None
    for c in [return_col, 'fwd_ret_clean', 'forward_return']:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors='coerce')
            if vals.notna().sum() > 100:
                src = c
                break
    if src is None:
        # Build from close
        close = pd.to_numeric(df['close'], errors='coerce').to_numpy()
        n = len(close)
        ret = np.full(n, np.nan)
        ret[: n - horizon_bars] = (
            close[horizon_bars:] / np.maximum(close[: n - horizon_bars], 1e-9) - 1.0
        )
        df['_computed_fwd_ret'] = ret
        src = '_computed_fwd_ret'
    fwd = pd.to_numeric(df[src], errors='coerce')

    bias = df['bias_label'].to_numpy()
    sign = np.where(bias == 0, 1.0, np.where(bias == 1, -1.0, 0.0))
    signed_ret = fwd.to_numpy() * sign       # positive iff label-direction wins

    # Subtract spread (in absolute terms, applied per trade)
    spread_cost = spread_pips * tick_size / np.maximum(
        pd.to_numeric(df['close'], errors='coerce').to_numpy(), 1e-9
    )
    signed_ret_net = signed_ret - spread_cost

    df['signed_return_gross'] = signed_ret
    df['signed_return_net']   = signed_ret_net
    df['signed_return_pips']  = signed_ret_net / tick_size   # in pips
    df['_return_src'] = src
    return df


def _bootstrap_ci(arr: np.ndarray, fn=np.mean, n_boot: int = 2000,
                  ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return float(fn(arr)) if len(arr) else 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot = np.array([fn(rng.choice(arr, len(arr), replace=True)) for _ in range(n_boot)])
    point = float(fn(arr))
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return point, lo, hi


def _zone_metrics(sub: pd.DataFrame, atr_median_overall: float) -> dict:
    """Compute edge metrics for one zone subset."""
    n = len(sub)
    if n == 0:
        return {'n': 0}

    bias = sub['bias_label'].to_numpy()
    n_long = int((bias == 0).sum())
    n_short = int((bias == 1).sum())
    long_share = n_long / max(n, 1)
    short_share = n_short / max(n, 1)

    ret = sub['signed_return_net'].to_numpy()
    pips = sub['signed_return_pips'].to_numpy()
    win_mask = ret > 0
    win_rate = float(win_mask.mean()) if n > 0 else 0.0

    mean_pips, mp_lo, mp_hi = _bootstrap_ci(pips, fn=np.mean)
    median_pips = float(np.nanmedian(pips))

    def _sharpe(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        return float(np.mean(x) / sd) if sd > 0 else 0.0

    sharpe, sh_lo, sh_hi = _bootstrap_ci(pips, fn=_sharpe)

    edge_score = mean_pips / max(atr_median_overall, 1e-9) if atr_median_overall > 0 else 0.0

    # Per-direction win rates
    long_wins = int(((bias == 0) & win_mask).sum())
    short_wins = int(((bias == 1) & win_mask).sum())
    long_wr = long_wins / max(n_long, 1)
    short_wr = short_wins / max(n_short, 1)

    return {
        'n': int(n),
        'n_long': n_long, 'n_short': n_short,
        'long_share': float(long_share), 'short_share': float(short_share),
        'long_win_rate': float(long_wr), 'short_win_rate': float(short_wr),
        'win_rate': float(win_rate),
        'mean_pips': float(mean_pips),
        'mean_pips_ci': [float(mp_lo), float(mp_hi)],
        'median_pips': float(median_pips),
        'sharpe': float(sharpe),
        'sharpe_ci': [float(sh_lo), float(sh_hi)],
        'edge_score': float(edge_score),
    }


# ════════════════════════════════════════════════════════════════════════════
# Slicing
# ════════════════════════════════════════════════════════════════════════════

def _assign_session(df: pd.DataFrame) -> pd.Series:
    """Map each bar to a session label."""
    if all(c in df.columns for c in ('is_london', 'is_ny', 'is_overlap')):
        is_london = df['is_london'].astype(bool).to_numpy()
        is_ny = df['is_ny'].astype(bool).to_numpy()
        is_overlap = df['is_overlap'].astype(bool).to_numpy()
        # Priority: Overlap > London > NY > Asia/Off
        labels = np.full(len(df), 'asia_or_off', dtype=object)
        labels[is_ny] = 'ny'
        labels[is_london] = 'london'
        labels[is_overlap] = 'overlap'
        return pd.Series(labels, index=df.index)
    # Fallback by hour
    hour = pd.to_datetime(df['ts_event']).dt.hour
    out = pd.Series('other', index=df.index)
    out[hour.between(0, 6)] = 'asia'
    out[hour.between(7, 11)] = 'london'
    out[hour.between(12, 15)] = 'overlap'
    out[hour.between(13, 19)] = 'ny'
    return out


def analyze_all_zones(df_dir: pd.DataFrame, atr_median: float) -> dict:
    """Run zone slicing on the directional subset."""
    sessions = _assign_session(df_dir)
    df_dir = df_dir.copy()
    df_dir['_session'] = sessions
    df_dir['_hour'] = pd.to_datetime(df_dir['ts_event']).dt.hour
    df_dir['_dow'] = pd.to_datetime(df_dir['ts_event']).dt.day_name()

    results: dict = {
        'overall': _zone_metrics(df_dir, atr_median),
        'by_session': {},
        'by_hour': {},
        'by_dow': {},
        'by_regime': {},
        'by_session_x_regime': {},
    }

    for s, sub in df_dir.groupby('_session'):
        results['by_session'][s] = _zone_metrics(sub, atr_median)

    for h, sub in df_dir.groupby('_hour'):
        if len(sub) >= 5:
            results['by_hour'][int(h)] = _zone_metrics(sub, atr_median)

    for d, sub in df_dir.groupby('_dow'):
        results['by_dow'][d] = _zone_metrics(sub, atr_median)

    if 'regime_label' in df_dir.columns:
        for r, sub in df_dir.groupby(df_dir['regime_label'].astype(str)):
            results['by_regime'][r] = _zone_metrics(sub, atr_median)
        for (s, r), sub in df_dir.groupby(['_session', df_dir['regime_label'].astype(str)]):
            if len(sub) >= 5:
                results['by_session_x_regime'][f'{s} × {r}'] = _zone_metrics(sub, atr_median)

    return results


# ════════════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════════════

def _format_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    """Build a simple markdown table. cols = [(field, header), ...]."""
    headers = [c[1] for c in cols]
    out = '| ' + ' | '.join(headers) + ' |\n'
    out += '|' + '|'.join(['---'] * len(headers)) + '|\n'
    for r in rows:
        row_vals = []
        for field, _ in cols:
            v = r.get(field, '')
            if isinstance(v, float):
                row_vals.append(f'{v:.3f}')
            elif isinstance(v, int):
                row_vals.append(str(v))
            else:
                row_vals.append(str(v))
        out += '| ' + ' | '.join(row_vals) + ' |\n'
    return out


def _filter_promising(d: dict, min_n: int = 10) -> list[dict]:
    """Return list of zone dicts that look like real edges."""
    promising = []
    for category in ('by_session', 'by_hour', 'by_dow', 'by_regime', 'by_session_x_regime'):
        for name, m in d.get(category, {}).items():
            if m['n'] < min_n:
                continue
            sh_lo, _ = m['sharpe_ci']
            mp_lo, _ = m['mean_pips_ci']
            promising.append({
                'category': category,
                'zone': str(name),
                **m,
                'ci_lower_sharpe': sh_lo,
                'ci_lower_pips': mp_lo,
                'pass': sh_lo > 0 and m['win_rate'] >= 0.52 and m['n'] >= min_n,
            })
    return promising


def build_markdown(results: dict, promising: list[dict], extras: dict) -> str:
    out = ['# Zone Edge Analysis (no-training, descriptive)\n']
    out.append(f"**Data**: {extras['data_source']}")
    out.append(f"**Total directional bars**: {results['overall']['n']}")
    out.append(f"**Return source column**: `{extras['return_col']}`")
    out.append(f"**ATR median (pips)**: {extras['atr_median_pips']:.1f}\n")

    # Overall
    o = results['overall']
    out.append('## Overall')
    out.append(f"- n = {o['n']} | LONG = {o['n_long']} | SHORT = {o['n_short']}")
    out.append(
        f"- win_rate = {o['win_rate']:.3f} | mean = {o['mean_pips']:+.2f} pips "
        f"(CI [{o['mean_pips_ci'][0]:+.2f}, {o['mean_pips_ci'][1]:+.2f}])"
    )
    out.append(
        f"- Sharpe = {o['sharpe']:.3f} (CI [{o['sharpe_ci'][0]:.3f}, {o['sharpe_ci'][1]:.3f}])\n"
    )

    # Categories
    for cat, title in [
        ('by_session', 'By Session'),
        ('by_regime', 'By Regime'),
        ('by_session_x_regime', 'By Session × Regime'),
        ('by_dow', 'By Day-of-Week'),
        ('by_hour', 'By Hour (UTC)'),
    ]:
        if not results[cat]:
            continue
        out.append(f'## {title}')
        rows = []
        for name, m in results[cat].items():
            rows.append({
                'zone': str(name), 'n': m['n'],
                'long_%': f"{m['long_share']*100:.0f}",
                'short_%': f"{m['short_share']*100:.0f}",
                'win_rate': m['win_rate'],
                'mean_pips': m['mean_pips'],
                'sharpe': m['sharpe'],
                'sharpe_lo': m['sharpe_ci'][0],
            })
        rows = sorted(rows, key=lambda r: -r['sharpe'])
        out.append(_format_table(rows, [
            ('zone', 'Zone'), ('n', 'n'),
            ('long_%', 'L%'), ('short_%', 'S%'),
            ('win_rate', 'Win'), ('mean_pips', 'Pips'),
            ('sharpe', 'Sharpe'), ('sharpe_lo', 'Sharpe_lo'),
        ]))
        out.append('')

    # Promising edges (passing filter)
    passing = [p for p in promising if p['pass']]
    out.append('## ⭐ Statistically Promising Edges')
    out.append(f"_Filters: n ≥ 10, win_rate ≥ 0.52, sharpe CI lower bound > 0._\n")
    if not passing:
        out.append('_None found — no zone passes statistical significance on this sample size._\n')
    else:
        passing = sorted(passing, key=lambda p: -p['ci_lower_sharpe'])
        rows = []
        for p in passing:
            rows.append({
                'category': p['category'],
                'zone': p['zone'], 'n': p['n'],
                'win_rate': p['win_rate'],
                'mean_pips': p['mean_pips'],
                'sharpe': p['sharpe'],
                'sharpe_lo': p['ci_lower_sharpe'],
            })
        out.append(_format_table(rows, [
            ('category', 'Category'), ('zone', 'Zone'), ('n', 'n'),
            ('win_rate', 'Win'), ('mean_pips', 'Pips'),
            ('sharpe', 'Sharpe'), ('sharpe_lo', 'Sharpe_lo'),
        ]))
        out.append('')

    # Top 5 anti-edges (consistently losing zones)
    anti = sorted(
        [p for p in promising if p['sharpe'] < 0 and p['n'] >= 10],
        key=lambda p: p['sharpe'],
    )[:5]
    if anti:
        out.append('## ⚠️ Anti-Edges (zones to AVOID)')
        rows = []
        for p in anti:
            rows.append({
                'category': p['category'],
                'zone': p['zone'], 'n': p['n'],
                'win_rate': p['win_rate'],
                'mean_pips': p['mean_pips'],
                'sharpe': p['sharpe'],
            })
        out.append(_format_table(rows, [
            ('category', 'Category'), ('zone', 'Zone'), ('n', 'n'),
            ('win_rate', 'Win'), ('mean_pips', 'Pips'),
            ('sharpe', 'Sharpe'),
        ]))
        out.append('')

    return '\n'.join(out)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--output', default='checkpoints/zone_edges')
    p.add_argument('--return-col', default='forward_return',
                  help='Column to use for trade returns')
    p.add_argument('--horizon-bars', type=int, default=12,
                  help='Used only if return col missing (computes close-to-close)')
    p.add_argument('--spread-pips', type=float, default=2.0)
    p.add_argument('--tick-size', type=float, default=0.0001)
    p.add_argument('--min-n', type=int, default=10,
                  help='Min samples per zone to flag as edge')
    args = p.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"═══ Zone Edge Analysis ═══")
    print(f"Features: {args.features}")
    print(f"Output:   {args.output}\n")

    df = pd.read_parquet(args.features)
    n_total = len(df)
    print(f"Loaded: {n_total:,} bars × {len(df.columns)} columns")

    if 'bias_label' not in df.columns:
        print('❌ No bias_label column — cannot proceed')
        sys.exit(1)

    bias = df['bias_label'].to_numpy()
    directional_mask = (bias == 0) | (bias == 1)
    n_directional = int(directional_mask.sum())
    print(f"Directional bars: {n_directional} ({n_directional/n_total*100:.1f}%)")

    if n_directional < 30:
        print(f'⚠️  Only {n_directional} directional bars — analysis will be noisy')

    # Compute signed returns
    df = _compute_signed_returns(
        df,
        return_col=args.return_col,
        horizon_bars=args.horizon_bars,
        spread_pips=args.spread_pips,
        tick_size=args.tick_size,
    )
    return_src = df['_return_src'].iloc[0] if '_return_src' in df.columns else args.return_col
    print(f"Return source: {return_src}")

    df_dir = df[directional_mask].copy()

    # ATR median (for edge_score normalization)
    if 'atr_14' in df.columns:
        atr_median = float(pd.to_numeric(df['atr_14'], errors='coerce').median())
    else:
        atr_median = 0.0030
    atr_median_pips = atr_median / args.tick_size

    print(f"ATR median: {atr_median_pips:.1f} pips\n")
    print(f"🔍 Slicing by zones...")

    results = analyze_all_zones(df_dir, atr_median)
    promising = _filter_promising(results, min_n=args.min_n)

    # ── Outputs ──
    extras = {
        'data_source': args.features,
        'return_col': return_src,
        'atr_median_pips': atr_median_pips,
        'spread_pips': args.spread_pips,
        'n_total': n_total,
        'n_directional': n_directional,
    }

    md = build_markdown(results, promising, extras)
    md_path = Path(args.output) / 'zone_edges_report.md'
    md_path.write_text(md)
    print(f"✅ Report: {md_path}")

    full = {
        'meta': extras,
        'results': results,
        'promising': promising,
    }
    json_path = Path(args.output) / 'zone_edges.json'
    with open(json_path, 'w') as f:
        json.dump(full, f, indent=2, default=str)
    print(f"✅ JSON: {json_path}")

    # CSV of top edges
    if promising:
        top = sorted(promising, key=lambda x: -x['ci_lower_sharpe'])
        top_df = pd.DataFrame(top)
        csv_path = Path(args.output) / 'zone_edges_topk.csv'
        top_df.to_csv(csv_path, index=False)
        print(f"✅ CSV:  {csv_path}")

    # Summary print
    print()
    print('═' * 60)
    print('Top 10 zones by Sharpe (lower-CI > 0 = statistically positive):')
    print('═' * 60)
    passing = sorted([p for p in promising if p['pass']], key=lambda p: -p['ci_lower_sharpe'])
    if not passing:
        print('  (none passing significance filter)')
    else:
        for p in passing[:10]:
            print(f"  [{p['category']:18s}] {p['zone']:30s} "
                  f"n={p['n']:4d}  win={p['win_rate']:.2f}  "
                  f"pips={p['mean_pips']:+5.2f}  "
                  f"sharpe={p['sharpe']:+.2f} [lo={p['ci_lower_sharpe']:+.2f}]")
    print('═' * 60)


if __name__ == '__main__':
    main()
