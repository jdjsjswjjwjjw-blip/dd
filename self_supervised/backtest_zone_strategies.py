"""
self_supervised/backtest_zone_strategies.py
═══════════════════════════════════════════════════════════
Realistic zone-aware backtest using R-multiple targets + pivot levels
+ bar-walking outcome tracking. NO TRAINING, NO LABEL SNOOPING.

For each bar i, simulates BOTH a LONG and a SHORT trade:

    Entry:  close[i]
    TP:
      • r-multiple mode:  close[i] ± tp_r × ATR[i]
      • pivot mode:       pdh (LONG)  or  pdl (SHORT) — institutional levels
      • wall mode:        nearest large book wall (if available)
    SL:                   close[i] ∓ sl_r × ATR[i]
    Max horizon:          --max-bars (default 24 = 6h on 15min)

Then walks forward through bars [i+1 .. i+max_bars] checking high/low:
    if high[t] ≥ TP_LONG  →  TP hit (LONG win)
    if low[t]  ≤ SL_LONG  →  SL hit (LONG lose)
    if neither            →  timeout, settle at close[i+max_bars]

Reports per-zone:
    n_trades        : number of simulated trades
    win_rate        : fraction of TP hits (timeouts split by sign)
    tp_rate         : fraction reaching TP
    sl_rate         : fraction stopping out
    timeout_rate
    avg_pnl_R       : mean pnl in R-multiples (risk = 1.0 R)
    avg_pnl_pips
    sharpe          : avg_pnl / std_pnl per trade (bootstrap CI)
    expectancy_R    : same as avg_pnl_R, named for emphasis
    profit_factor   : sum(wins) / |sum(losses)|

Filters "promising" edges where:
    n ≥ 30
    Sharpe CI lower bound > 0
    expectancy ≥ 0.1 R
    Profit factor > 1.2

Usage:
    python self_supervised/backtest_zone_strategies.py \\
        --features pipeline_15min_v2/day_trading_features.parquet \\
        --output checkpoints/.../zone_backtest \\
        --target-type r-multiple --target-r 2.0 --stop-r 1.0 \\
        --max-bars 24 --spread-pips 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# Trade simulation
# ════════════════════════════════════════════════════════════════════════════

def simulate_trades(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, atr: np.ndarray,
    tp_px_long: np.ndarray, sl_px_long: np.ndarray,
    tp_px_short: np.ndarray, sl_px_short: np.ndarray,
    max_bars: int, spread_pips: float, tick_size: float,
) -> dict:
    """Vectorized-ish bar-walking trade simulator.

    For each entry bar i, walk forward up to max_bars and find first
    barrier hit. Returns per-bar arrays of:
        long_pnl_R, long_outcome (0=SL, 1=TP, 2=TIMEOUT)
        short_pnl_R, short_outcome
    """
    n = len(close)
    spread_cost_atr = (spread_pips * tick_size) / np.maximum(atr, 1e-9)

    long_pnl_R = np.full(n, np.nan, dtype=np.float64)
    long_out   = np.full(n, -1, dtype=np.int8)
    short_pnl_R = np.full(n, np.nan, dtype=np.float64)
    short_out   = np.full(n, -1, dtype=np.int8)

    for i in range(n - 1):
        if not np.isfinite(atr[i]) or atr[i] <= 0 or not np.isfinite(close[i]):
            continue
        risk_long = close[i] - sl_px_long[i]    # always positive for valid trade
        risk_short = sl_px_short[i] - close[i]  # always positive
        if risk_long <= 0 or risk_short <= 0:
            continue

        end = min(i + 1 + max_bars, n)

        # LONG simulation
        tp_hit = False
        sl_hit = False
        bars_to_exit = 0
        for j in range(i + 1, end):
            if not (np.isfinite(high[j]) and np.isfinite(low[j])):
                continue
            # Check SL first (conservative on same-bar conflict)
            if low[j] <= sl_px_long[i]:
                sl_hit = True
                bars_to_exit = j - i
                break
            if high[j] >= tp_px_long[i]:
                tp_hit = True
                bars_to_exit = j - i
                break
        if sl_hit:
            long_out[i] = 0
            long_pnl_R[i] = -1.0 - spread_cost_atr[i]
        elif tp_hit:
            long_out[i] = 1
            reward = (tp_px_long[i] - close[i]) / risk_long
            long_pnl_R[i] = reward - spread_cost_atr[i]
        elif np.isfinite(close[end - 1]):
            long_out[i] = 2
            settle = close[end - 1]
            long_pnl_R[i] = (settle - close[i]) / risk_long - spread_cost_atr[i]

        # SHORT simulation
        tp_hit = False
        sl_hit = False
        for j in range(i + 1, end):
            if not (np.isfinite(high[j]) and np.isfinite(low[j])):
                continue
            if high[j] >= sl_px_short[i]:
                sl_hit = True
                break
            if low[j] <= tp_px_short[i]:
                tp_hit = True
                break
        if sl_hit:
            short_out[i] = 0
            short_pnl_R[i] = -1.0 - spread_cost_atr[i]
        elif tp_hit:
            short_out[i] = 1
            reward = (close[i] - tp_px_short[i]) / risk_short
            short_pnl_R[i] = reward - spread_cost_atr[i]
        elif np.isfinite(close[end - 1]):
            short_out[i] = 2
            settle = close[end - 1]
            short_pnl_R[i] = (close[i] - settle) / risk_short - spread_cost_atr[i]

    return {
        'long_pnl_R': long_pnl_R, 'long_outcome': long_out,
        'short_pnl_R': short_pnl_R, 'short_outcome': short_out,
    }


# ════════════════════════════════════════════════════════════════════════════
# Target construction
# ════════════════════════════════════════════════════════════════════════════

def build_targets(
    df: pd.DataFrame, target_type: str,
    target_r: float, stop_r: float,
) -> dict[str, np.ndarray]:
    """Compute TP/SL price arrays for both sides."""
    close = pd.to_numeric(df['close'], errors='coerce').to_numpy(dtype=np.float64)
    atr = _get_atr_series(df, verbose=False)

    # Always SL at fixed R (risk anchor)
    sl_px_long = close - stop_r * atr
    sl_px_short = close + stop_r * atr

    if target_type == 'r-multiple':
        tp_px_long = close + target_r * atr
        tp_px_short = close - target_r * atr

    elif target_type == 'pivot':
        # TP = pdh (LONG) or pdl (SHORT). Floor at fixed-R if pdh too close.
        if 'pdh' in df.columns and 'pdl' in df.columns:
            pdh = pd.to_numeric(df['pdh'], errors='coerce').to_numpy()
            pdl = pd.to_numeric(df['pdl'], errors='coerce').to_numpy()
            tp_px_long = np.maximum(pdh, close + target_r * atr)
            tp_px_short = np.minimum(pdl, close - target_r * atr)
            # Replace NaN with R-multiple fallback
            nan_mask = ~np.isfinite(tp_px_long)
            tp_px_long[nan_mask] = close[nan_mask] + target_r * atr[nan_mask]
            nan_mask = ~np.isfinite(tp_px_short)
            tp_px_short[nan_mask] = close[nan_mask] - target_r * atr[nan_mask]
        else:
            print("⚠️  --target-type pivot but no pdh/pdl columns; falling back to r-multiple")
            tp_px_long = close + target_r * atr
            tp_px_short = close - target_r * atr

    elif target_type == 'wall':
        # TP at nearest "wall" — combine bid_wall_strength + distance_to_wall
        # Heuristic: target at close ± (distance_to_wall × atr × strength_factor)
        if all(c in df.columns for c in ('bid_wall_strength', 'ask_wall_strength', 'distance_to_wall')):
            dist = pd.to_numeric(df['distance_to_wall'], errors='coerce').to_numpy()
            ask_w = pd.to_numeric(df['ask_wall_strength'], errors='coerce').fillna(0).to_numpy()
            bid_w = pd.to_numeric(df['bid_wall_strength'], errors='coerce').fillna(0).to_numpy()
            # Floor: at least target_r × ATR (so we don't have absurdly tiny TPs)
            wall_dist_atr = np.maximum(np.abs(dist), target_r * atr)
            # Sign by wall strength (LONG targets ask wall; SHORT targets bid wall)
            tp_px_long = close + wall_dist_atr * (1.0 + 0.3 * np.clip(ask_w, -1, 1))
            tp_px_short = close - wall_dist_atr * (1.0 + 0.3 * np.clip(bid_w, -1, 1))
        else:
            print("⚠️  --target-type wall but missing wall features; falling back to r-multiple")
            tp_px_long = close + target_r * atr
            tp_px_short = close - target_r * atr

    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return {
        'tp_px_long': tp_px_long,
        'sl_px_long': sl_px_long,
        'tp_px_short': tp_px_short,
        'sl_px_short': sl_px_short,
    }


# ════════════════════════════════════════════════════════════════════════════
# Zone assignment + metrics
# ════════════════════════════════════════════════════════════════════════════

def _assign_session(df: pd.DataFrame) -> pd.Series:
    if all(c in df.columns for c in ('is_london', 'is_ny', 'is_overlap')):
        is_london = df['is_london'].astype(bool).to_numpy()
        is_ny = df['is_ny'].astype(bool).to_numpy()
        is_overlap = df['is_overlap'].astype(bool).to_numpy()
        labels = np.full(len(df), 'asia_or_off', dtype=object)
        labels[is_ny] = 'ny'
        labels[is_london] = 'london'
        labels[is_overlap] = 'overlap'
        return pd.Series(labels, index=df.index)
    hour = pd.to_datetime(df['ts_event']).dt.hour
    out = pd.Series('other', index=df.index)
    out[hour.between(0, 6)] = 'asia'
    out[hour.between(7, 11)] = 'london'
    out[hour.between(12, 15)] = 'overlap'
    out[hour.between(13, 19)] = 'ny'
    return out


def _get_atr_series(df: pd.DataFrame, verbose: bool = False) -> np.ndarray:
    """Find ATR column under any of the known names; compute from H/L/C as fallback."""
    for c in ('atr_14', 'atr', 'micro_atr', 'micro_atr_max'):
        if c in df.columns:
            arr = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=np.float64)
            if np.isfinite(arr).sum() > 100:
                if verbose:
                    print(f"  ATR source: {c} (median={np.nanmedian(arr) / 1e-4:.1f} pips)")
                return arr
    if verbose:
        print(f"  ⚠️  no ATR column found; computing from HLC (rolling 14)")
    high = pd.to_numeric(df['high'], errors='coerce')
    low = pd.to_numeric(df['low'], errors='coerce')
    close = pd.to_numeric(df['close'], errors='coerce')
    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=1).mean().to_numpy(dtype=np.float64)
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


def side_metrics(pnl_R: np.ndarray, outcome: np.ndarray, atr_pips: np.ndarray) -> dict:
    """Compute trade statistics for one side."""
    mask = np.isfinite(pnl_R) & (outcome >= 0)
    pnl = pnl_R[mask]
    out = outcome[mask]
    atr_p = atr_pips[mask] if len(atr_pips) == len(pnl_R) else None
    n = len(pnl)
    if n == 0:
        return {'n': 0}

    tp_rate = float((out == 1).mean())
    sl_rate = float((out == 0).mean())
    timeout_rate = float((out == 2).mean())
    win_rate = float((pnl > 0).mean())

    avg_R, avg_R_lo, avg_R_hi = _bootstrap_ci(pnl, fn=np.mean)
    med_R = float(np.median(pnl))

    def _sharpe_fn(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        return float(np.mean(x) / sd) if sd > 0 else 0.0

    sharpe, sh_lo, sh_hi = _bootstrap_ci(pnl, fn=_sharpe_fn)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    # Pips version (uses atr in pips × avg_R)
    if atr_p is not None:
        pnl_pips = pnl * atr_p
        avg_pips, ap_lo, ap_hi = _bootstrap_ci(pnl_pips, fn=np.mean)
    else:
        avg_pips, ap_lo, ap_hi = 0.0, 0.0, 0.0

    return {
        'n': int(n),
        'tp_rate': tp_rate, 'sl_rate': sl_rate, 'timeout_rate': timeout_rate,
        'win_rate': win_rate,
        'avg_pnl_R': float(avg_R),
        'avg_pnl_R_ci': [float(avg_R_lo), float(avg_R_hi)],
        'median_pnl_R': med_R,
        'avg_pnl_pips': float(avg_pips),
        'sharpe': float(sharpe),
        'sharpe_ci': [float(sh_lo), float(sh_hi)],
        'profit_factor': pf if not np.isinf(pf) else 999.0,
    }


def zone_metrics(df: pd.DataFrame, sim: dict) -> dict:
    """Per-zone metrics for both sides."""
    atr_pips = _get_atr_series(df) / 1e-4
    long_m = side_metrics(sim['long_pnl_R'], sim['long_outcome'], atr_pips)
    short_m = side_metrics(sim['short_pnl_R'], sim['short_outcome'], atr_pips)
    return {
        'n_bars': int(len(df)),
        'long': long_m,
        'short': short_m,
        'best_side': 'LONG' if long_m.get('sharpe', -1e9) >= short_m.get('sharpe', -1e9) else 'SHORT',
        'best_sharpe': float(max(long_m.get('sharpe', -1e9), short_m.get('sharpe', -1e9))),
    }


def analyze_all_zones(df: pd.DataFrame, sim: dict) -> dict:
    """Slice by all zone categories."""
    sessions = _assign_session(df)
    df = df.copy()
    df['_session'] = sessions.to_numpy()
    df['_hour'] = pd.to_datetime(df['ts_event']).dt.hour.to_numpy()
    df['_dow'] = pd.to_datetime(df['ts_event']).dt.day_name().to_numpy()

    def _idx_to_slices(values):
        groups = {}
        for v in pd.unique(values):
            idx = np.where(np.asarray(values) == v)[0]
            groups[v] = idx
        return groups

    results = {
        'overall': zone_metrics(df, sim),
        'by_session': {},
        'by_hour': {},
        'by_dow': {},
        'by_regime': {},
        'by_session_x_regime': {},
    }

    def _sub(idx):
        return df.iloc[idx], {
            'long_pnl_R': sim['long_pnl_R'][idx],
            'long_outcome': sim['long_outcome'][idx],
            'short_pnl_R': sim['short_pnl_R'][idx],
            'short_outcome': sim['short_outcome'][idx],
        }

    for s, idx in _idx_to_slices(df['_session']).items():
        results['by_session'][str(s)] = zone_metrics(*_sub(idx))

    for h, idx in _idx_to_slices(df['_hour']).items():
        if len(idx) >= 30:
            results['by_hour'][int(h)] = zone_metrics(*_sub(idx))

    for d, idx in _idx_to_slices(df['_dow']).items():
        results['by_dow'][str(d)] = zone_metrics(*_sub(idx))

    if 'regime_label' in df.columns:
        regime = df['regime_label'].astype(str).to_numpy()
        for r, idx in _idx_to_slices(regime).items():
            results['by_regime'][str(r)] = zone_metrics(*_sub(idx))
        # Cross-tab
        combo = np.char.add(np.char.add(np.asarray(df['_session'], dtype=str), ' × '), regime.astype(str))
        for k, idx in _idx_to_slices(combo).items():
            if len(idx) >= 30:
                results['by_session_x_regime'][str(k)] = zone_metrics(*_sub(idx))

    return results


def filter_promising(results: dict, min_n: int = 30,
                     min_sharpe_lo: float = 0.0,
                     min_expectancy_R: float = 0.05,
                     min_pf: float = 1.2) -> list[dict]:
    promising = []
    for cat in ('by_session', 'by_hour', 'by_dow', 'by_regime', 'by_session_x_regime'):
        for name, m in results.get(cat, {}).items():
            for side in ('long', 'short'):
                s = m.get(side, {})
                if s.get('n', 0) < min_n:
                    continue
                sh_lo = s['sharpe_ci'][0]
                exp_R = s['avg_pnl_R']
                pf = s['profit_factor']
                if sh_lo > min_sharpe_lo and exp_R >= min_expectancy_R and pf >= min_pf:
                    promising.append({
                        'category': cat, 'zone': str(name), 'side': side.upper(),
                        **s,
                        'sharpe_lo': sh_lo,
                    })
    return sorted(promising, key=lambda p: -p['sharpe_lo'])


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--output', default='checkpoints/zone_backtest')
    p.add_argument('--target-type', choices=['r-multiple', 'pivot', 'wall'],
                  default='r-multiple')
    p.add_argument('--target-r', type=float, default=2.0,
                  help='TP at target_r × ATR for r-multiple mode')
    p.add_argument('--stop-r', type=float, default=1.0,
                  help='SL at stop_r × ATR')
    p.add_argument('--max-bars', type=int, default=24,
                  help='Max bars to hold before timeout')
    p.add_argument('--spread-pips', type=float, default=2.0)
    p.add_argument('--tick-size', type=float, default=0.0001)
    p.add_argument('--min-n', type=int, default=30)
    args = p.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"═══ Zone-aware Strategy Backtest ═══")
    print(f"Features:    {args.features}")
    print(f"Target type: {args.target_type}  (TP={args.target_r}R, SL={args.stop_r}R)")
    print(f"Max bars:    {args.max_bars}  (= {args.max_bars * 15} min hold horizon at 15min)")
    print(f"Spread cost: {args.spread_pips} pips per trade\n")

    df = pd.read_parquet(args.features)
    print(f"Loaded: {len(df):,} bars × {len(df.columns)} cols\n")

    targets = build_targets(df, args.target_type, args.target_r, args.stop_r)

    close = pd.to_numeric(df['close'], errors='coerce').to_numpy()
    high = pd.to_numeric(df['high'], errors='coerce').to_numpy()
    low = pd.to_numeric(df['low'], errors='coerce').to_numpy()
    atr = _get_atr_series(df, verbose=True)

    print(f"🎯 Simulating {len(df):,} LONG + SHORT trades...")
    sim = simulate_trades(
        close, high, low, atr,
        targets['tp_px_long'], targets['sl_px_long'],
        targets['tp_px_short'], targets['sl_px_short'],
        args.max_bars, args.spread_pips, args.tick_size,
    )

    # Overall stats first
    atr_pips = atr / args.tick_size
    overall_long = side_metrics(sim['long_pnl_R'], sim['long_outcome'], atr_pips)
    overall_short = side_metrics(sim['short_pnl_R'], sim['short_outcome'], atr_pips)
    print()
    print(f"  ── Overall ──")
    print(f"  LONG : n={overall_long['n']:5d}  tp%={overall_long['tp_rate']:.2f} "
          f"sl%={overall_long['sl_rate']:.2f}  win={overall_long['win_rate']:.3f}  "
          f"avg={overall_long['avg_pnl_R']:+.3f}R "
          f"sharpe={overall_long['sharpe']:+.2f} [lo={overall_long['sharpe_ci'][0]:+.2f}]  "
          f"PF={overall_long['profit_factor']:.2f}")
    print(f"  SHORT: n={overall_short['n']:5d}  tp%={overall_short['tp_rate']:.2f} "
          f"sl%={overall_short['sl_rate']:.2f}  win={overall_short['win_rate']:.3f}  "
          f"avg={overall_short['avg_pnl_R']:+.3f}R "
          f"sharpe={overall_short['sharpe']:+.2f} [lo={overall_short['sharpe_ci'][0]:+.2f}]  "
          f"PF={overall_short['profit_factor']:.2f}")
    print()

    print(f"🔍 Slicing by zones...")
    results = analyze_all_zones(df, sim)
    promising = filter_promising(results, min_n=args.min_n)

    # Write outputs
    full = {
        'config': vars(args),
        'overall': {'long': overall_long, 'short': overall_short},
        'results': results,
        'promising': promising,
    }
    json_path = Path(args.output) / 'zone_backtest.json'
    with open(json_path, 'w') as f:
        json.dump(full, f, indent=2, default=str)
    print(f"✅ JSON: {json_path}")

    # Print top zones
    print()
    print('═' * 90)
    print(f'Promising zones (Sharpe-CI > 0, expectancy ≥ 0.05R, PF ≥ 1.2, n ≥ {args.min_n}):')
    print('═' * 90)
    if not promising:
        print('  (none — try different --target-r, --stop-r, or --max-bars)')
    else:
        for p in promising[:15]:
            print(f"  [{p['category']:20s}] {p['zone']:28s} {p['side']:5s} "
                  f"n={p['n']:4d}  win={p['win_rate']:.2f}  "
                  f"avg={p['avg_pnl_R']:+.2f}R  "
                  f"sharpe={p['sharpe']:+.2f} [lo={p['sharpe_lo']:+.2f}]  "
                  f"PF={p['profit_factor']:.2f}")
    print('═' * 90)


if __name__ == '__main__':
    main()
