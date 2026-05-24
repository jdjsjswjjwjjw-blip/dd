"""
ssl/validate_ssl.py
═══════════════════════════════════════════════════════════
Phase C: Validation شامل مع acceptance criteria.

يحسب الـ 6 metrics الحاسمة لقرار "هل نعتمد على SSL":
    ① Holdout Accuracy (> 55%)
    ② Sharpe Ratio after costs (> 0.5)
    ③ Calibration AUC (> 0.60)
    ④ LONG/SHORT balance (0.5-2.0)
    ⑤ Per-regime accuracy (trending + ranging > 50%)
    ⑥ Max Drawdown (< 5%)

ينتج verdict نهائي: ✅ PASS | ⚠️ MIXED | ❌ FAIL
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


# Acceptance thresholds (matches our agreed criteria)
ACCEPTANCE_CRITERIA = {
    'holdout_accuracy_min'    : 0.55,
    'sharpe_min'              : 0.5,
    'calibration_auc_min'     : 0.60,
    'balance_min'             : 0.5,
    'balance_max'             : 2.0,
    'per_regime_accuracy_min' : 0.50,
    'max_drawdown_max'        : 0.05,
}


def compute_sharpe_after_costs(
    df_holdout: pd.DataFrame, preds: np.ndarray, probs: np.ndarray,
    spread_ticks: float = 2.0, tick_size: float = 0.0001,
    fees_atr_proxy: float = 0.15, confidence_threshold: float = 0.55,
) -> dict:
    """يحسب Sharpe ratio بعد الـ costs بناءً على الـ predictions."""
    if 'close' not in df_holdout.columns or 'atr_14' not in df_holdout.columns:
        return {'sharpe': 0.0, 'n_trades': 0, 'mean_return': 0.0, 'std_return': 0.0}

    close = df_holdout['close'].to_numpy()
    atr = df_holdout['atr_14'].to_numpy()
    n = len(close)

    # Only trade when confidence > threshold
    max_probs = probs.max(axis=1)
    trade_mask = max_probs >= confidence_threshold

    # Forward returns over 12 bars
    horizon = 12
    returns = np.zeros(n)
    for i in range(n - horizon):
        if not trade_mask[i]:
            continue
        future_close = close[i + horizon]
        if preds[i] == 0:  # LONG
            ret = (future_close - close[i]) / max(close[i], 1e-9)
        else:  # SHORT
            ret = (close[i] - future_close) / max(close[i], 1e-9)
        # Subtract costs
        cost = (fees_atr_proxy * atr[i] + spread_ticks * tick_size) / max(close[i], 1e-9)
        ret -= cost
        returns[i] = ret

    trade_returns = returns[trade_mask & (np.arange(n) < n - horizon)]
    if len(trade_returns) < 2:
        return {'sharpe': 0.0, 'n_trades': 0, 'mean_return': 0.0, 'std_return': 0.0}

    mean_ret = float(np.mean(trade_returns))
    std_ret = float(np.std(trade_returns))
    sharpe = mean_ret / (std_ret + 1e-9) * np.sqrt(252 * 24 * 4)  # annualize (15min bars)
    return {
        'sharpe': float(sharpe),
        'n_trades': int(trade_mask.sum()),
        'mean_return': mean_ret,
        'std_return': std_ret,
    }


def compute_calibration_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Calibration AUC: هل max_prob ~= P(correct)؟"""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return 0.0
    if len(np.unique(labels)) < 2:
        return 0.0
    correct = (probs.argmax(axis=1) == labels).astype(int)
    confidence = probs.max(axis=1)
    try:
        return float(roc_auc_score(correct, confidence))
    except Exception:
        return 0.0


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown من cumulative returns."""
    if len(returns) == 0:
        return 0.0
    cum = np.cumprod(1.0 + returns) - 1.0
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.maximum(1.0 + peak, 1e-9)
    return float(-dd.min()) if len(dd) > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--embeddings', required=True)
    p.add_argument('--direction-head', required=True, help='best_direction_head.pt')
    p.add_argument('--output', default='checkpoints/validation_report.json')
    p.add_argument('--train-split', type=float, default=0.75)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = p.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Load
    df = pd.read_parquet(args.features)
    n = len(df)
    embeddings = np.load(args.embeddings)
    ckpt = torch.load(args.direction_head, map_location=device, weights_only=False)
    mu = ckpt['mu']
    sigma = ckpt['sigma']
    input_dim = ckpt['input_dim']

    model = DirectionHead(input_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Filter directional + apply same time split
    bias = df['bias_label'].to_numpy()
    directional_mask = (bias == 0) | (bias == 1)
    X_all = embeddings[directional_mask]
    y_all = bias[directional_mask]
    df_dir = df[directional_mask].reset_index(drop=True)
    split_idx = int(len(X_all) * args.train_split)
    X_holdout = X_all[split_idx:]
    y_holdout = y_all[split_idx:]
    df_holdout = df_dir.iloc[split_idx:].reset_index(drop=True)
    n_holdout = len(X_holdout)
    print(f"Holdout size: {n_holdout}")
    if n_holdout < 10:
        print(f"⚠️  Very few holdout samples — results unreliable")

    # Normalize + predict
    X_holdout_norm = (X_holdout - mu) / sigma
    X_tensor = torch.from_numpy(X_holdout_norm.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)

    # ── Metric 1: Holdout Accuracy ──
    accuracy = float((preds == y_holdout).mean()) if n_holdout > 0 else 0.0

    # ── Metric 2: LONG/SHORT balance ──
    n_long_pred = int((preds == 0).sum())
    n_short_pred = int((preds == 1).sum())
    balance = max(n_long_pred, n_short_pred) / max(min(n_long_pred, n_short_pred), 1)

    # ── Metric 3: Calibration AUC ──
    calibration_auc = compute_calibration_auc(probs, y_holdout)

    # ── Metric 4: Sharpe after costs ──
    sharpe_info = compute_sharpe_after_costs(df_holdout, preds, probs)

    # ── Metric 5: Per-regime accuracy ──
    per_regime = {}
    if 'regime_label' in df_holdout.columns:
        for regime in ['trending', 'ranging', 'volatile', 'low_liquidity']:
            mask = df_holdout['regime_label'].astype(str) == regime
            if mask.sum() > 0:
                regime_acc = float((preds[mask] == y_holdout[mask]).mean())
                per_regime[regime] = {
                    'accuracy': regime_acc,
                    'n_samples': int(mask.sum()),
                }

    # ── Metric 6: Max Drawdown ──
    # Reconstruct returns from sharpe_info computation
    horizon = 12
    if 'close' in df_holdout.columns:
        close = df_holdout['close'].to_numpy()
        returns_traded = []
        for i in range(len(close) - horizon):
            if probs[i].max() >= 0.55:
                future_close = close[i + horizon]
                if preds[i] == 0:
                    ret = (future_close - close[i]) / max(close[i], 1e-9)
                else:
                    ret = (close[i] - future_close) / max(close[i], 1e-9)
                returns_traded.append(ret)
        max_dd = compute_max_drawdown(np.array(returns_traded))
    else:
        max_dd = 0.0

    # ── Verdict ──
    crit = ACCEPTANCE_CRITERIA
    passes = {
        'holdout_accuracy': accuracy >= crit['holdout_accuracy_min'],
        'sharpe': sharpe_info['sharpe'] >= crit['sharpe_min'],
        'calibration_auc': calibration_auc >= crit['calibration_auc_min'],
        'balance': crit['balance_min'] <= balance <= crit['balance_max'],
        'per_regime_accuracy': all(
            r.get('accuracy', 0) >= crit['per_regime_accuracy_min']
            for r in per_regime.values() if r.get('n_samples', 0) >= 5
        ) if per_regime else False,
        'max_drawdown': max_dd <= crit['max_drawdown_max'],
    }

    n_passes = sum(passes.values())
    n_total = len(passes)
    if n_passes >= 5:
        verdict = '✅ PASS — نوسّع SSL'
        verdict_code = 'PASS'
    elif n_passes >= 3:
        verdict = '⚠️ MIXED — يحتاج tuning'
        verdict_code = 'MIXED'
    else:
        verdict = '❌ FAIL — نراجع approach'
        verdict_code = 'FAIL'

    # ── Print Report ──
    print()
    print("═" * 72)
    print("              SSL VALIDATION REPORT")
    print("═" * 72)
    print(f"Holdout size: {n_holdout} directional samples")
    print(f"Split: time-based ({args.train_split*100:.0f}% train / {(1-args.train_split)*100:.0f}% holdout)")
    print()
    print(f"  ① Holdout Accuracy:        {accuracy:.3f}   "
          f"{'✅' if passes['holdout_accuracy'] else '❌'} (min {crit['holdout_accuracy_min']})")
    print(f"  ② Sharpe (after costs):    {sharpe_info['sharpe']:.3f}   "
          f"{'✅' if passes['sharpe'] else '❌'} (min {crit['sharpe_min']})")
    print(f"  ③ Calibration AUC:         {calibration_auc:.3f}   "
          f"{'✅' if passes['calibration_auc'] else '❌'} (min {crit['calibration_auc_min']})")
    print(f"  ④ LONG/SHORT balance:      {balance:.2f}:1  "
          f"{'✅' if passes['balance'] else '❌'} ({crit['balance_min']}-{crit['balance_max']})")
    if per_regime:
        for reg, info in per_regime.items():
            mark = '✅' if info['accuracy'] >= crit['per_regime_accuracy_min'] and info['n_samples'] >= 5 else ('⚠️' if info['n_samples'] < 5 else '❌')
            print(f"     {reg:14s}: acc={info['accuracy']:.3f} (n={info['n_samples']}) {mark}")
    print(f"  ⑥ Max Drawdown:            {max_dd:.3f}   "
          f"{'✅' if passes['max_drawdown'] else '❌'} (max {crit['max_drawdown_max']})")
    print()
    print(f"  Trade count (validated): {sharpe_info['n_trades']}")
    print(f"  Mean return per trade:   {sharpe_info['mean_return']:.5f}")
    print()
    print(f"═" * 72)
    print(f"  VERDICT: {verdict}   ({n_passes}/{n_total} criteria passed)")
    print(f"═" * 72)

    # Save report
    report = {
        'verdict': verdict_code,
        'n_criteria_passed': n_passes,
        'n_criteria_total': n_total,
        'metrics': {
            'holdout_accuracy': accuracy,
            'sharpe': sharpe_info['sharpe'],
            'calibration_auc': calibration_auc,
            'balance': balance,
            'max_drawdown': max_dd,
            'n_trades': sharpe_info['n_trades'],
            'mean_return': sharpe_info['mean_return'],
            'std_return': sharpe_info['std_return'],
        },
        'per_regime': per_regime,
        'passes': passes,
        'criteria': crit,
        'n_holdout': n_holdout,
        'predictions': {
            'long': int(n_long_pred),
            'short': int(n_short_pred),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {args.output}")


if __name__ == '__main__':
    main()
