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

def _compute_directional_returns(
    df: pd.DataFrame,
    return_col: str = 'forward_return',
    horizon_bars: int = 12,
    spread_pips: float = 2.0,
    tick_size: float = 0.0001,
) -> pd.DataFrame:
    """Build forward return columns for unconditional zone analysis.

    CRITICAL: do NOT sign by bias_label. The bias_label was assigned
    based on the outcome (LONG if MFE>MAE, SHORT otherwise), so signing
    by it would yield artifact "win rates" near 100%. The right question
    is the UNCONDITIONAL forward return at each bar — i.e., "if I trade
    LONG (or SHORT) on every bar in this zone, what's my expectancy?"

    Adds columns:
        forward_ret_raw  : raw forward return (close[i+H]/close[i] - 1)
        long_ret_net     : forward_ret_raw - spread_cost (LONG side)
        short_ret_net    : -forward_ret_raw - spread_cost (SHORT side)
        long_ret_pips    : long_ret_net / tick_size
        short_ret_pips   : short_ret_net / tick_size
    """
    df = df.copy()

    # Pick source for forward return — but only as a sanity check; we
    # ALWAYS recompute from close to avoid label-derived columns.
    close = pd.to_numeric(df['close'], errors='coerce').to_numpy()
    n = len(close)
    ret = np.full(n, np.nan)
    ret[: n - horizon_bars] = (
        close[horizon_bars:] / np.maximum(close[: n - horizon_bars], 1e-9) - 1.0
    )
    df['forward_ret_raw'] = ret
    df['_return_src'] = f'close[i+{horizon_bars}]/close[i] - 1 (causal)'

    # Spread cost (per trade)
    spread_cost = spread_pips * tick_size / np.maximum(close, 1e-9)
    df['long_ret_net']   = ret - spread_cost
    df['short_ret_net']  = -ret - spread_cost
    df['long_ret_pips']  = df['long_ret_net'] / tick_size
    df['short_ret_pips'] = df['short_ret_net'] / tick_size
    return df


# Keep the old function as an alias for backward compatibility (it's
# now considered LEAKY — only useful for sanity-checking the label
# consistency, NOT for edge discovery).
_compute_signed_returns = _compute_directional_returns


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
    """Compute UNCONDITIONAL edge metrics for one zone subset.

    Tests two strategies independently:
      • LONG-every-bar: enter LONG at close[i], exit at close[i+H]
      • SHORT-every-bar: same but short

    For each, reports win-rate, mean pips, Sharpe (bootstrap CI).
    The better of the two indicates whether the zone has directional
    bias; if both have negative Sharpe, the zone is range-bound /
    unprofitable to take a constant direction in.

    Also reports the directional-LABEL distribution (LONG share /
    SHORT share / NEUTRAL share) as a separate signal — but these
    metrics are NOT used to compute the trade-side returns.
    """
    n = len(sub)
    if n == 0:
        return {'n': 0}

    long_pips  = sub['long_ret_pips'].to_numpy()
    short_pips = sub['short_ret_pips'].to_numpy()
    # Filter NaN (last horizon bars don't have forward return)
    long_pips  = long_pips[np.isfinite(long_pips)]
    short_pips = short_pips[np.isfinite(short_pips)]
    n_long_bars  = len(long_pips)
    n_short_bars = len(short_pips)

    def _sharpe_fn(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        return float(np.mean(x) / sd) if sd > 0 else 0.0

    def _summary(arr):
        if len(arr) == 0:
            return {'n': 0, 'win_rate': 0.0, 'mean_pips': 0.0,
                    'mean_pips_ci': [0.0, 0.0], 'median_pips': 0.0,
                    'sharpe': 0.0, 'sharpe_ci': [0.0, 0.0]}
        wr = float((arr > 0).mean())
        mp, mp_lo, mp_hi = _bootstrap_ci(arr, fn=np.mean)
        med = float(np.median(arr))
        sh, sh_lo, sh_hi = _bootstrap_ci(arr, fn=_sharpe_fn)
        return {
            'n': int(len(arr)),
            'win_rate': float(wr),
            'mean_pips': float(mp),
            'mean_pips_ci': [float(mp_lo), float(mp_hi)],
            'median_pips': float(med),
            'sharpe': float(sh),
            'sharpe_ci': [float(sh_lo), float(sh_hi)],
        }

    long_stats  = _summary(long_pips)
    short_stats = _summary(short_pips)

    # Label distribution (informational only — not used in P&L)
    if 'bias_label' in sub.columns:
        bias = sub['bias_label'].to_numpy()
        n_long_lbl  = int((bias == 0).sum())
        n_short_lbl = int((bias == 1).sum())
        n_neut_lbl  = int((bias == 2).sum())
        n_directional = n_long_lbl + n_short_lbl
        long_label_share  = n_long_lbl  / max(n, 1)
        short_label_share = n_short_lbl / max(n, 1)
        directional_rate  = n_directional / max(n, 1)
    else:
        long_label_share = short_label_share = directional_rate = 0.0
        n_directional = 0

    # Edge score: |best side mean| / atr_median
    best_mean = max(long_stats['mean_pips'], short_stats['mean_pips'])
    edge_score = best_mean / max(atr_median_overall / 1e-4, 1e-9) if atr_median_overall > 0 else 0.0

    return {
        'n': int(n),
        'directional_rate': float(directional_rate),
        'long_label_share': float(long_label_share),
        'short_label_share': float(short_label_share),
        'long_side': long_stats,
        'short_side': short_stats,
        'best_side': 'LONG' if long_stats['sharpe'] >= short_stats['sharpe'] else 'SHORT',
        'best_sharpe': float(max(long_stats['sharpe'], short_stats['sharpe'])),
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
    """Surface zones where EITHER side (LONG or SHORT) has a positive edge.

    A zone has a real edge if, taking the better side, the bootstrap CI
    lower bound of Sharpe is > 0 and win rate ≥ 0.52.
    """
    promising = []
    for category in ('by_session', 'by_hour', 'by_dow', 'by_regime', 'by_session_x_regime'):
        for name, m in d.get(category, {}).items():
            if m['n'] < min_n:
                continue
            for side_name in ('LONG', 'SHORT'):
                side_key = 'long_side' if side_name == 'LONG' else 'short_side'
                s = m[side_key]
                if s['n'] < min_n:
                    continue
                sh_lo, _ = s['sharpe_ci']
                mp_lo, _ = s['mean_pips_ci']
                promising.append({
                    'category': category,
                    'zone': str(name),
                    'side': side_name,
                    'n': s['n'],
                    'directional_rate': m.get('directional_rate', 0.0),
                    'win_rate': s['win_rate'],
                    'mean_pips': s['mean_pips'],
                    'mean_pips_ci': s['mean_pips_ci'],
                    'sharpe': s['sharpe'],
                    'sharpe_ci': s['sharpe_ci'],
                    'ci_lower_sharpe': sh_lo,
                    'ci_lower_pips': mp_lo,
                    'pass': sh_lo > 0 and s['win_rate'] >= 0.52 and s['n'] >= min_n,
                })
    return promising


def build_markdown(results: dict, promising: list[dict], extras: dict) -> str:
    out = ['# Zone Edge Analysis — Unconditional (no labels, no training)\n']
    out.append(f"**Data**: {extras['data_source']}")
    out.append(f"**Bars analyzed**: {extras['n_total']} (forward-return horizon: {extras['horizon_bars']} bars)")
    out.append(f"**Directional labels in data**: {extras['n_directional']} (informational only — NOT used in P&L)")
    out.append(f"**Return source**: `{extras['return_col']}`")
    out.append(f"**ATR median (pips)**: {extras['atr_median_pips']:.1f}")
    out.append(f"**Spread cost**: {extras['spread_pips']:.1f} pips per trade\n")

    out.append('> **Methodology**: For each zone, compute the forward return '
               '`close[i+H]/close[i] - 1` on every bar (regardless of label). '
               'Then evaluate two strategies independently: (1) trade LONG '
               'every bar, (2) trade SHORT every bar. A real edge means '
               'one of the sides has Sharpe CI lower bound > 0. The '
               'bias_label column is NOT used to compute returns (it was '
               'derived from outcomes — using it would be data snooping).\n')

    # Overall
    o = results['overall']
    out.append('## Overall')
    out.append(f"- n_bars = {o['n']} | directional labels = {o['directional_rate']*100:.1f}%")
    long_s, short_s = o['long_side'], o['short_side']
    out.append(f"- LONG  : win={long_s['win_rate']:.3f} | mean={long_s['mean_pips']:+.2f} pips "
               f"| Sharpe={long_s['sharpe']:+.3f} CI [{long_s['sharpe_ci'][0]:+.3f}, {long_s['sharpe_ci'][1]:+.3f}]")
    out.append(f"- SHORT : win={short_s['win_rate']:.3f} | mean={short_s['mean_pips']:+.2f} pips "
               f"| Sharpe={short_s['sharpe']:+.3f} CI [{short_s['sharpe_ci'][0]:+.3f}, {short_s['sharpe_ci'][1]:+.3f}]\n")

    # Per category
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
                'long_win': m['long_side']['win_rate'],
                'long_pips': m['long_side']['mean_pips'],
                'long_sharpe': m['long_side']['sharpe'],
                'long_sharpe_lo': m['long_side']['sharpe_ci'][0],
                'short_win': m['short_side']['win_rate'],
                'short_pips': m['short_side']['mean_pips'],
                'short_sharpe': m['short_side']['sharpe'],
                'short_sharpe_lo': m['short_side']['sharpe_ci'][0],
            })
        rows = sorted(rows, key=lambda r: -max(r['long_sharpe_lo'], r['short_sharpe_lo']))
        out.append(_format_table(rows, [
            ('zone', 'Zone'), ('n', 'n'),
            ('long_win', 'L-win'), ('long_pips', 'L-pips'),
            ('long_sharpe', 'L-Sh'), ('long_sharpe_lo', 'L-Sh_lo'),
            ('short_win', 'S-win'), ('short_pips', 'S-pips'),
            ('short_sharpe', 'S-Sh'), ('short_sharpe_lo', 'S-Sh_lo'),
        ]))
        out.append('')

    # Statistically promising
    passing = [p for p in promising if p['pass']]
    out.append('## ⭐ Statistically Promising Edges')
    out.append(f"_Filters: n ≥ 10, win_rate ≥ 0.52, Sharpe CI lower bound > 0_\n")
    if not passing:
        out.append('_No zone passes statistical significance — market appears '
                   'efficient on this sample at the chosen horizon. Try a '
                   'different horizon (--horizon-bars) or larger dataset._\n')
    else:
        passing = sorted(passing, key=lambda p: -p['ci_lower_sharpe'])
        rows = []
        for p in passing:
            rows.append({
                'category': p['category'], 'zone': p['zone'], 'side': p['side'],
                'n': p['n'], 'win_rate': p['win_rate'],
                'mean_pips': p['mean_pips'],
                'sharpe': p['sharpe'], 'sharpe_lo': p['ci_lower_sharpe'],
            })
        out.append(_format_table(rows, [
            ('category', 'Category'), ('zone', 'Zone'), ('side', 'Side'),
            ('n', 'n'), ('win_rate', 'Win'), ('mean_pips', 'Pips'),
            ('sharpe', 'Sharpe'), ('sharpe_lo', 'Sharpe_lo'),
        ]))
        out.append('')

    # Anti-edges
    anti = sorted(
        [p for p in promising if p['sharpe'] < 0],
        key=lambda p: p['sharpe'],
    )[:5]
    if anti:
        out.append('## ⚠️ Anti-Edges (zones to AVOID for this side)')
        rows = []
        for p in anti:
            rows.append({
                'category': p['category'], 'zone': p['zone'], 'side': p['side'],
                'n': p['n'], 'win_rate': p['win_rate'],
                'mean_pips': p['mean_pips'], 'sharpe': p['sharpe'],
            })
        out.append(_format_table(rows, [
            ('category', 'Category'), ('zone', 'Zone'), ('side', 'Side'),
            ('n', 'n'), ('win_rate', 'Win'), ('mean_pips', 'Pips'),
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

    # Compute UNCONDITIONAL forward returns (no label-based signing)
    df = _compute_directional_returns(
        df,
        return_col=args.return_col,
        horizon_bars=args.horizon_bars,
        spread_pips=args.spread_pips,
        tick_size=args.tick_size,
    )
    return_src = df['_return_src'].iloc[0] if '_return_src' in df.columns else args.return_col
    print(f"Return source: {return_src}")

    # Analyze on ALL bars (not just directional) — the strategy is
    # "trade every bar in this zone", not "trade only labeled bars".
    df_for_zones = df.copy()

    # ATR median (for edge_score normalization)
    if 'atr_14' in df.columns:
        atr_median = float(pd.to_numeric(df['atr_14'], errors='coerce').median())
    else:
        atr_median = 0.0030
    atr_median_pips = atr_median / args.tick_size

    print(f"ATR median: {atr_median_pips:.1f} pips\n")
    print(f"🔍 Slicing by zones...")

    results = analyze_all_zones(df_for_zones, atr_median)
    promising = _filter_promising(results, min_n=args.min_n)

    # ── Outputs ──
    extras = {
        'data_source': args.features,
        'return_col': return_src,
        'atr_median_pips': atr_median_pips,
        'spread_pips': args.spread_pips,
        'horizon_bars': args.horizon_bars,
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
    print('═' * 78)
    print('Top 10 zones by Sharpe lower-CI (statistically positive UNCONDITIONAL edge):')
    print('═' * 78)
    passing = sorted([p for p in promising if p['pass']], key=lambda p: -p['ci_lower_sharpe'])
    if not passing:
        print('  (none — market efficient on this sample/horizon)')
    else:
        for p in passing[:10]:
            print(f"  [{p['category']:20s}] {p['zone']:28s} {p['side']:5s} "
                  f"n={p['n']:4d}  win={p['win_rate']:.2f}  "
                  f"pips={p['mean_pips']:+5.2f}  "
                  f"sharpe={p['sharpe']:+.2f} [lo={p['ci_lower_sharpe']:+.2f}]")
    print('═' * 78)


if __name__ == '__main__':
    main()
