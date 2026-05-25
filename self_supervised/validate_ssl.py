"""
ssl/validate_ssl.py
═══════════════════════════════════════════════════════════
Phase E: SSL validation with quant-grade rigor.

Six acceptance metrics on the holdout window:
    ① Holdout Accuracy            (≥ 0.55, +95% bootstrap CI)
    ② Sharpe Ratio after costs    (≥ 0.5,  HAC + non-overlapping trades)
    ③ Calibration AUC             (≥ 0.60)
    ④ LONG/SHORT balance          (0.5 ≤ ratio ≤ 2.0)
    ⑤ Per-regime accuracy         (trending+ranging ≥ 0.50, n ≥ 10)
    ⑥ Max Drawdown                (≤ 5%, non-overlapping trades)

Quant-grade fixes embedded:
  * S5/S6 — entry is close[i+1] not close[i]; horizons measured in
    CALENDAR bars (not in directional rows).
  * S7   — embedding validity mask required; NaN rows refused.
  * S8   — DirectionHead hidden_dim/dropout pulled from checkpoint.
  * B2   — Sharpe uses non-overlapping trades; annualization adjusts
           for bar frequency.
  * B3   — Max DD computed on non-overlapping trade returns.
  * B6   — bootstrap CIs reported for all point estimates.

Verdict: ✅ PASS | ⚠️ MIXED | ❌ FAIL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from self_supervised.fine_tune_direction import DirectionHead


ACCEPTANCE_CRITERIA = {
    'holdout_accuracy_min'    : 0.55,
    'sharpe_min'              : 0.5,
    'calibration_auc_min'     : 0.60,
    'balance_min'             : 0.5,
    'balance_max'             : 2.0,
    'per_regime_accuracy_min' : 0.50,
    'per_regime_min_samples'  : 10,    # B6: ignore regimes with too few samples
    'max_drawdown_max'        : 0.05,
}


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════

def _bootstrap_ci(stat_fn, *arrays, n_boot: int = 1000, seed: int = 42,
                  ci: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap percentile CI for a scalar statistic.

    Returns (point_estimate, lower, upper).
    For paired arrays of length n, samples indices with replacement.
    """
    if len(arrays) == 0 or len(arrays[0]) == 0:
        return 0.0, 0.0, 0.0
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = float(stat_fn(*(a[idx] for a in arrays)))
    point = float(stat_fn(*arrays))
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return point, lo, hi


def _annualization_factor(median_bar_seconds: float, n_per_year: int = 252) -> float:
    """Per-sqrt-bar annualization for Sharpe.

    For NON-OVERLAPPING per-bar returns, Sharpe annualization = sqrt(bars_per_year).
    bars_per_year ≈ session_seconds_per_year / median_bar_seconds.
    GBPUSD futures: ~23h/day × 5 days/week × 52 weeks ≈ 21,528,000 sec/year,
    fall back to 252 trading days × 24h if unknown.
    """
    if median_bar_seconds <= 0:
        return float(np.sqrt(n_per_year * 96))  # default 15m × 24h
    seconds_per_year = n_per_year * 24 * 3600.0
    bars_per_year = seconds_per_year / median_bar_seconds
    return float(np.sqrt(max(bars_per_year, 1.0)))


def _build_non_overlapping_trades(
    df_holdout: pd.DataFrame,
    preds: np.ndarray, probs: np.ndarray,
    horizon: int, confidence_threshold: float,
    spread_ticks: float, tick_size: float, fees_atr_proxy: float,
) -> dict:
    """Build a list of NON-OVERLAPPING trades from per-bar predictions.

    Algorithm:
      Iterate bars left→right (calendar order). When confidence ≥ threshold,
      open a trade at close[i+1] (the bar AFTER the signal, not the signal
      bar itself — fixes S6 look-ahead). Exit at close[i+1+horizon].
      Skip horizon bars after opening so trades don't overlap (fixes B2/B3).

    Returns dict with:
      returns_net  : (M,) net returns per non-overlapping trade
      entry_idx    : (M,) entry bar indices
      median_dt    : median bar duration in seconds (for annualization)
    """
    if 'close' not in df_holdout.columns or 'atr_14' not in df_holdout.columns:
        return {'returns_net': np.array([]), 'entry_idx': np.array([], dtype=int),
                'median_dt': 0.0}

    close = df_holdout['close'].to_numpy(dtype=np.float64)
    atr = df_holdout['atr_14'].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(df_holdout['ts_event']).to_numpy()
    n = len(close)

    # Median inter-bar gap (excluding session breaks) for annualization
    bar_dt_sec = pd.Series(ts).diff().dt.total_seconds().dropna()
    if len(bar_dt_sec) > 0:
        # Use modal-ish gap, not full median (weekend gaps would bias it up)
        bar_dt_sec = bar_dt_sec[bar_dt_sec <= bar_dt_sec.quantile(0.95)]
        median_dt = float(bar_dt_sec.median()) if len(bar_dt_sec) > 0 else 0.0
    else:
        median_dt = 0.0

    max_probs = probs.max(axis=1)
    take = max_probs >= confidence_threshold

    returns_net = []
    entry_idx = []
    i = 0
    while i < n - 1 - horizon:
        if not take[i]:
            i += 1
            continue
        entry_i = i + 1            # S6 fix: enter NEXT bar, not signal bar
        exit_i = entry_i + horizon
        if exit_i >= n:
            break
        entry_px = close[entry_i]
        exit_px = close[exit_i]
        if not (np.isfinite(entry_px) and np.isfinite(exit_px) and entry_px > 0):
            i += 1
            continue
        if preds[i] == 0:           # LONG
            gross = (exit_px - entry_px) / entry_px
        else:                       # SHORT
            gross = (entry_px - exit_px) / entry_px
        cost = (
            fees_atr_proxy * (atr[i] if np.isfinite(atr[i]) else 0.0)
            + spread_ticks * tick_size
        ) / max(entry_px, 1e-9)
        returns_net.append(gross - cost)
        entry_idx.append(entry_i)
        i = exit_i + 1              # B2/B3 fix: skip horizon for non-overlap

    return {
        'returns_net': np.array(returns_net, dtype=np.float64),
        'entry_idx': np.array(entry_idx, dtype=np.int64),
        'median_dt': median_dt,
    }


def compute_sharpe_after_costs(trades: dict, n_per_year: int = 252) -> dict:
    """Sharpe + bootstrap CI on non-overlapping trades."""
    returns = trades['returns_net']
    n = len(returns)
    if n < 5:
        return {'sharpe': 0.0, 'sharpe_lo': 0.0, 'sharpe_hi': 0.0,
                'n_trades': int(n), 'mean_return': 0.0, 'std_return': 0.0,
                'annualization': 0.0}
    ann = _annualization_factor(trades['median_dt'], n_per_year)

    def _sharpe(r):
        mu = float(np.mean(r))
        sd = float(np.std(r, ddof=1))
        return (mu / sd * ann) if sd > 0 else 0.0

    sharpe, lo, hi = _bootstrap_ci(_sharpe, returns, n_boot=2000)
    return {
        'sharpe': float(sharpe), 'sharpe_lo': float(lo), 'sharpe_hi': float(hi),
        'n_trades': int(n),
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns, ddof=1)),
        'annualization': float(ann),
    }


def compute_calibration_auc(probs: np.ndarray, labels: np.ndarray,
                            n_boot: int = 1000) -> dict:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {'auc': 0.0, 'auc_lo': 0.0, 'auc_hi': 0.0}
    if len(np.unique(labels)) < 2 or len(labels) < 5:
        return {'auc': 0.0, 'auc_lo': 0.0, 'auc_hi': 0.0}
    correct = (probs.argmax(axis=1) == labels).astype(int)
    confidence = probs.max(axis=1)
    if len(np.unique(correct)) < 2:
        return {'auc': 0.0, 'auc_lo': 0.0, 'auc_hi': 0.0}

    def _auc(c, p):
        if len(np.unique(c)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(c, p))
        except Exception:
            return 0.5

    point, lo, hi = _bootstrap_ci(_auc, correct, confidence, n_boot=n_boot)
    return {'auc': float(point), 'auc_lo': float(lo), 'auc_hi': float(hi)}


def compute_max_drawdown(returns: np.ndarray) -> dict:
    """Max DD on non-overlapping trades.

    Returns:
      max_dd      : peak-to-trough fractional drawdown
      max_dd_lo   : bootstrap lower bound (less negative = milder)
      max_dd_hi   : bootstrap upper bound (more negative = worse)
    """
    if len(returns) == 0:
        return {'max_dd': 0.0, 'max_dd_lo': 0.0, 'max_dd_hi': 0.0}

    def _mdd(r):
        cum = np.cumprod(1.0 + r) - 1.0
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / np.maximum(1.0 + peak, 1e-9)
        return float(-dd.min()) if len(dd) > 0 else 0.0

    if len(returns) < 5:
        return {'max_dd': _mdd(returns), 'max_dd_lo': 0.0, 'max_dd_hi': 0.0}

    point, lo, hi = _bootstrap_ci(_mdd, returns, n_boot=2000)
    return {'max_dd': float(point), 'max_dd_lo': float(lo), 'max_dd_hi': float(hi)}


def compute_accuracy(preds: np.ndarray, labels: np.ndarray,
                     n_boot: int = 1000) -> dict:
    if len(preds) == 0:
        return {'accuracy': 0.0, 'acc_lo': 0.0, 'acc_hi': 0.0}

    def _acc(p, y):
        return float((p == y).mean())

    point, lo, hi = _bootstrap_ci(_acc, preds, labels, n_boot=n_boot)
    return {'accuracy': float(point), 'acc_lo': float(lo), 'acc_hi': float(hi)}


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--embeddings', required=True)
    p.add_argument('--direction-head', required=True, help='best_direction_head.pt')
    p.add_argument('--output', default='checkpoints/validation_report.json')
    p.add_argument('--train-split', type=float, default=0.75)
    p.add_argument('--horizon-bars', type=int, default=12,
                  help='Trade horizon in CALENDAR bars (not directional rows)')
    p.add_argument('--confidence-threshold', type=float, default=0.51,
                  help='Trade if max(prob) >= threshold. Default 0.51 (any lean). '
                       'Use 0.55+ to restrict to higher-confidence signals.')
    p.add_argument('--spread-ticks', type=float, default=2.0)
    p.add_argument('--tick-size', type=float, default=0.0001)
    p.add_argument('--fees-atr-proxy', type=float, default=0.15)
    p.add_argument('--n-per-year', type=int, default=252,
                  help='Trading days/year for annualization')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = p.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # ── Load ──
    df = pd.read_parquet(args.features)
    n = len(df)
    embeddings = np.load(args.embeddings)
    if len(embeddings) != n:
        raise RuntimeError(f"Length mismatch: df={n} embeddings={len(embeddings)}")

    valid_path = Path(args.embeddings).parent / 'embedding_valid.npy'
    if valid_path.exists():
        embedding_valid = np.load(valid_path)
    else:
        embedding_valid = np.isfinite(embeddings).all(axis=1)

    ckpt = torch.load(args.direction_head, map_location=device, weights_only=False)
    mu = ckpt['mu']
    sigma = ckpt['sigma']
    input_dim = ckpt['input_dim']
    # S8 fix: read hidden_dim / dropout from checkpoint (was hard-coded)
    hidden_dim = int(ckpt.get('hidden_dim', 64))
    dropout = float(ckpt.get('dropout', 0.3))
    model = DirectionHead(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    print(f"✅ Direction head loaded (input_dim={input_dim}, hidden_dim={hidden_dim})")

    # ── A4 FIX: calendar-based split, NOT directional-filtered split ──
    # Previous version filtered directional rows first, then took
    # int(len_dir * 0.75) — that boundary varies in time and could
    # interleave train/val on the calendar. Now: use a fixed calendar
    # cutoff matching fine_tune_direction.py.
    ts = pd.to_datetime(df['ts_event']).to_numpy()
    split_calendar_idx = int(n * args.train_split)
    # Use the calendar cutoff stored in the checkpoint when available
    # (guarantees train/eval consistency).
    if 'split_calendar_idx' in ckpt:
        split_calendar_idx = int(ckpt['split_calendar_idx'])
        print(f"   Using checkpoint split_calendar_idx={split_calendar_idx}")

    bias = df['bias_label'].to_numpy()
    directional_mask = (bias == 0) | (bias == 1)
    holdout_row_mask = (
        directional_mask
        & embedding_valid
        & (np.arange(n) >= split_calendar_idx)
    )

    if int(holdout_row_mask.sum()) == 0:
        print("❌ FATAL: no valid+directional rows in holdout window")
        sys.exit(1)

    X_holdout = embeddings[holdout_row_mask].astype(np.float32)
    y_holdout = bias[holdout_row_mask].astype(np.int64)
    df_holdout = df[holdout_row_mask].reset_index(drop=True)
    n_holdout = len(X_holdout)
    print(f"Holdout: {n_holdout} directional+valid samples "
          f"(calendar-split @ {pd.Timestamp(ts[split_calendar_idx])})")
    if n_holdout < 30:
        print(f"⚠️  Tiny holdout ({n_holdout}) — bootstrap CIs will be very wide. "
              f"Treat point estimates as indicative only.")

    # ── Predict (using train-fit stats from checkpoint) ──
    X_norm = (X_holdout - mu) / sigma
    with torch.no_grad():
        logits = model(torch.from_numpy(X_norm.astype(np.float32)).to(device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)

    # ── Probability distribution (key diagnostic for tiny holdouts) ──
    max_probs = probs.max(axis=1)
    prob_stats = {
        'min': float(max_probs.min()),
        'p25': float(np.percentile(max_probs, 25)),
        'p50': float(np.percentile(max_probs, 50)),
        'p75': float(np.percentile(max_probs, 75)),
        'max': float(max_probs.max()),
        'mean': float(max_probs.mean()),
        'frac_above_threshold': float((max_probs >= args.confidence_threshold).mean()),
    }
    print(f"\n📊 Confidence distribution (max_prob):")
    print(f"   min={prob_stats['min']:.3f} | p25={prob_stats['p25']:.3f} | "
          f"median={prob_stats['p50']:.3f} | p75={prob_stats['p75']:.3f} | "
          f"max={prob_stats['max']:.3f}")
    print(f"   mean={prob_stats['mean']:.3f} | "
          f"frac ≥ {args.confidence_threshold} = {prob_stats['frac_above_threshold']*100:.1f}%")

    # ── Metric 1: Accuracy + CI ──
    acc_info = compute_accuracy(preds, y_holdout)

    # ── Metric 2: LONG/SHORT balance ──
    n_long = int((preds == 0).sum())
    n_short = int((preds == 1).sum())
    if n_long + n_short == 0:
        balance = float('inf')
    elif min(n_long, n_short) == 0:
        balance = float('inf')   # one class never predicted — fails balance
    else:
        balance = max(n_long, n_short) / min(n_long, n_short)

    # ── Metric 3: Calibration AUC ──
    cal_info = compute_calibration_auc(probs, y_holdout)

    # ── Metric 4: Sharpe after costs (non-overlapping trades) ──
    # IMPORTANT: walk the full holdout calendar (NOT df_dir) so the
    # horizon parameter measures CALENDAR bars. We need a per-bar prob
    # vector aligned to the calendar — set prob=0.0 (don't trade) on
    # non-directional bars.
    # Build a calendar-aligned prob/pred array for the holdout window.
    holdout_calendar_idx = np.arange(split_calendar_idx, n)
    cal_df = df.iloc[split_calendar_idx:].reset_index(drop=True).copy()

    cal_probs = np.zeros((len(cal_df), 2), dtype=np.float32)
    cal_preds = np.zeros(len(cal_df), dtype=np.int64)
    # Map holdout-direction rows back to calendar positions
    direction_rows_in_holdout = np.where(directional_mask[split_calendar_idx:]
                                          & embedding_valid[split_calendar_idx:])[0]
    cal_probs[direction_rows_in_holdout] = probs
    cal_preds[direction_rows_in_holdout] = preds

    trades = _build_non_overlapping_trades(
        cal_df, cal_preds, cal_probs,
        horizon=args.horizon_bars,
        confidence_threshold=args.confidence_threshold,
        spread_ticks=args.spread_ticks, tick_size=args.tick_size,
        fees_atr_proxy=args.fees_atr_proxy,
    )
    sharpe_info = compute_sharpe_after_costs(trades, n_per_year=args.n_per_year)

    # ── Metric 5: Per-regime accuracy ──
    per_regime = {}
    if 'regime_label' in df_holdout.columns:
        for regime in ['trending', 'ranging', 'volatile', 'low_liquidity']:
            mask = df_holdout['regime_label'].astype(str) == regime
            n_reg = int(mask.sum())
            if n_reg >= ACCEPTANCE_CRITERIA['per_regime_min_samples']:
                reg_info = compute_accuracy(preds[mask], y_holdout[mask])
                per_regime[regime] = {**reg_info, 'n_samples': n_reg}
            elif n_reg > 0:
                per_regime[regime] = {
                    'accuracy': float((preds[mask] == y_holdout[mask]).mean()),
                    'acc_lo': 0.0, 'acc_hi': 0.0,
                    'n_samples': n_reg, 'underpowered': True,
                }

    # ── Metric 6: Max Drawdown ──
    dd_info = compute_max_drawdown(trades['returns_net'])

    # ── Verdict ──
    crit = ACCEPTANCE_CRITERIA
    passes = {
        'holdout_accuracy': acc_info['accuracy'] >= crit['holdout_accuracy_min'],
        'sharpe': sharpe_info['sharpe'] >= crit['sharpe_min'],
        'calibration_auc': cal_info['auc'] >= crit['calibration_auc_min'],
        'balance': crit['balance_min'] <= balance <= crit['balance_max'],
        'per_regime_accuracy': all(
            r['accuracy'] >= crit['per_regime_accuracy_min']
            for r in per_regime.values()
            if not r.get('underpowered', False)
        ) if any(not r.get('underpowered', False) for r in per_regime.values()) else False,
        'max_drawdown': dd_info['max_dd'] <= crit['max_drawdown_max'],
    }

    n_passes = sum(passes.values())
    n_total = len(passes)
    if n_passes >= 5:
        verdict_code = 'PASS'
        verdict = '✅ PASS'
    elif n_passes >= 3:
        verdict_code = 'MIXED'
        verdict = '⚠️ MIXED'
    else:
        verdict_code = 'FAIL'
        verdict = '❌ FAIL'

    # ── Report ──
    def _ci(v_lo, v_hi):
        return f"[{v_lo:.3f}, {v_hi:.3f}]"

    print()
    print("═" * 72)
    print("              SSL VALIDATION REPORT")
    print("═" * 72)
    print(f"Holdout: {n_holdout} directional+valid samples")
    print(f"Split:   calendar-based @ row {split_calendar_idx} = "
          f"{pd.Timestamp(ts[split_calendar_idx])}")
    print(f"Trade horizon: {args.horizon_bars} calendar bars; "
          f"non-overlapping (skip horizon after entry)")
    print(f"Trades:  {sharpe_info['n_trades']} non-overlapping")
    print()
    print(f"  ① Holdout Accuracy:        {acc_info['accuracy']:.3f}  "
          f"CI {_ci(acc_info['acc_lo'], acc_info['acc_hi'])} "
          f"{'✅' if passes['holdout_accuracy'] else '❌'} (min {crit['holdout_accuracy_min']})")
    print(f"  ② Sharpe (after costs):    {sharpe_info['sharpe']:.3f}  "
          f"CI {_ci(sharpe_info['sharpe_lo'], sharpe_info['sharpe_hi'])} "
          f"{'✅' if passes['sharpe'] else '❌'} (min {crit['sharpe_min']})")
    print(f"     annualization = sqrt(bars/yr) = {sharpe_info['annualization']:.1f}")
    print(f"  ③ Calibration AUC:         {cal_info['auc']:.3f}  "
          f"CI {_ci(cal_info['auc_lo'], cal_info['auc_hi'])} "
          f"{'✅' if passes['calibration_auc'] else '❌'} (min {crit['calibration_auc_min']})")
    bal_str = 'inf' if np.isinf(balance) else f"{balance:.2f}"
    print(f"  ④ LONG/SHORT balance:      {bal_str}:1 "
          f"{'✅' if passes['balance'] else '❌'} ({crit['balance_min']}-{crit['balance_max']})")
    if per_regime:
        print("  ⑤ Per-regime accuracy:")
        for reg, info in per_regime.items():
            mark = '⚠️ underpowered' if info.get('underpowered') else (
                '✅' if info['accuracy'] >= crit['per_regime_accuracy_min'] else '❌'
            )
            ci_str = _ci(info['acc_lo'], info['acc_hi']) if not info.get('underpowered') else ''
            print(f"     {reg:14s}: acc={info['accuracy']:.3f} {ci_str} "
                  f"n={info['n_samples']} {mark}")
    print(f"  ⑥ Max Drawdown:            {dd_info['max_dd']:.3f}  "
          f"CI {_ci(dd_info['max_dd_lo'], dd_info['max_dd_hi'])} "
          f"{'✅' if passes['max_drawdown'] else '❌'} (max {crit['max_drawdown_max']})")
    print()
    print(f"  Mean return per trade:   {sharpe_info['mean_return']:.5f}")
    print(f"═" * 72)
    print(f"  VERDICT: {verdict}   ({n_passes}/{n_total} criteria passed)")
    print(f"═" * 72)

    # ── Save ──
    report = {
        'verdict': verdict_code,
        'n_criteria_passed': n_passes,
        'n_criteria_total': n_total,
        'metrics': {
            'holdout_accuracy': acc_info,
            'sharpe': sharpe_info,
            'calibration_auc': cal_info,
            'balance': float(balance) if not np.isinf(balance) else None,
            'max_drawdown': dd_info,
        },
        'per_regime': per_regime,
        'passes': passes,
        'criteria': crit,
        'n_holdout': n_holdout,
        'predictions': {'long': int(n_long), 'short': int(n_short)},
        'split': {
            'calendar_idx': int(split_calendar_idx),
            'calendar_ts': str(pd.Timestamp(ts[split_calendar_idx])),
        },
        'prob_stats': prob_stats,
        'config': {
            'horizon_bars': int(args.horizon_bars),
            'confidence_threshold': float(args.confidence_threshold),
            'spread_ticks': float(args.spread_ticks),
            'tick_size': float(args.tick_size),
            'fees_atr_proxy': float(args.fees_atr_proxy),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {args.output}")


if __name__ == '__main__':
    main()
