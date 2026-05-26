"""
self_supervised/integrate_with_day_trade.py
═══════════════════════════════════════════════════════════════════
يدمج predictions الـ SSL regression head مع day_trade events
عشان نشوف:
  (1) هل الـ SSL signal بيتفق مع direction اللي day_trade بيتنبأ بيه؟
  (2) لو اتفقا، الـ hit rate بيتحسن قد إيه؟
  (3) إيه الـ events اللي SSL بيقول "خلّيك بعيد"؟

الـ output: events_with_ssl.parquet — كل event عليه:
  ssl_score_h6, ssl_score_h24       (signed magnitude predictions)
  ssl_prob_up_h6, ssl_prob_up_h24
  ssl_mag_h6, ssl_mag_h24
  ssl_rank_h24                      (rolling quintile)
  ssl_agree_h6, ssl_agree_h24       (sign match مع event_direction)

ويطبع تقرير hit rate مقارنة:
  ① base day_trade hit rate (no filter)
  ② SSL-direction-agree filter (h=6)
  ③ SSL-direction-agree filter (h=24)
  ④ SSL high-conviction filter (top-quintile magnitude + agreement)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _rolling_rank(s: pd.Series, window: int = 200) -> pd.Series:
    """Rank of each value within its trailing window. Returns ∈ [0,1]."""
    return s.rolling(window=window, min_periods=20).apply(
        lambda x: (x.iloc[-1] >= x).mean(), raw=False,
    )


def _hit_rate_at_horizon(
    df: pd.DataFrame, h: int, event_dir_col: str = 'event_direction',
) -> tuple[float, int]:
    """Hit rate = fraction of events where event_direction agrees with
    sign(target_ret_h). Returns (hit_rate, n_events)."""
    ret_col = f'target_ret_{h}'
    valid_col = f'target_valid_{h}'
    if ret_col not in df.columns or valid_col not in df.columns:
        return float('nan'), 0
    msk = (
        (df[event_dir_col] != 0)
        & df[valid_col].fillna(False)
        & df[ret_col].notna()
    )
    sub = df.loc[msk]
    if len(sub) == 0:
        return float('nan'), 0
    correct = (np.sign(sub[ret_col]) == sub[event_dir_col]).mean()
    return float(correct), int(len(sub))


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True,
                   help='combined_6m/day_trading_features.parquet')
    p.add_argument('--ssl-predictions', required=True,
                   help='regression/predictions.parquet')
    p.add_argument('--output', required=True,
                   help='Output: events_with_ssl.parquet')
    p.add_argument('--holdout-start', default='',
                   help='ISO timestamp; defaults to dataset_slice==holdout if column exists')
    p.add_argument('--rank-window', type=int, default=200,
                   help='Window for rolling SSL rank computation')
    p.add_argument('--top-quintile', type=float, default=0.80,
                   help='Rank threshold for "high conviction" filter')
    args = p.parse_args()

    print("═══ Day-Trade × SSL Integration ═══")
    print()
    feat = pd.read_parquet(args.features)
    feat['ts_event'] = pd.to_datetime(feat['ts_event'])
    ssl = pd.read_parquet(args.ssl_predictions)
    ssl['ts_event'] = pd.to_datetime(ssl['ts_event'])
    print(f"Features: {len(feat):,} rows | SSL preds: {len(ssl):,} rows")

    # ── Merge on ts_event ──
    keep_ssl = ['ts_event', 'emb_valid']
    for h in (6, 24):
        keep_ssl += [f'prob_up_{h}', f'mag_{h}', f'signed_score_{h}',
                     f'target_ret_{h}', f'target_valid_{h}']
    ssl_keep = ssl[[c for c in keep_ssl if c in ssl.columns]].copy()
    merged = feat.merge(ssl_keep, on='ts_event', how='left', suffixes=('', '_ssl'))
    if len(merged) != len(feat):
        print(f"⚠️  Merge changed row count: {len(feat)} → {len(merged)}")
    print(f"After merge: {len(merged):,} rows")
    print()

    # ── Compute rolling ranks ──
    print("Computing rolling SSL ranks (window={})...".format(args.rank_window))
    for h in (6, 24):
        col = f'signed_score_{h}'
        if col in merged.columns:
            merged[f'ssl_rank_{h}'] = _rolling_rank(merged[col], args.rank_window)

    # ── Agreement flags ──
    for h in (6, 24):
        sc = f'signed_score_{h}'
        if sc in merged.columns:
            merged[f'ssl_agree_h{h}'] = (
                (np.sign(merged[sc]) == merged['event_direction'])
                & (merged['event_direction'] != 0)
                & merged[sc].notna()
            )

    # ── Holdout slice for honest evaluation ──
    if args.holdout_start:
        holdout_mask = merged['ts_event'] >= pd.Timestamp(args.holdout_start)
        print(f"Holdout: ts >= {args.holdout_start}  → {int(holdout_mask.sum()):,} rows")
    elif 'dataset_slice' in merged.columns:
        holdout_mask = merged['dataset_slice'] == 'holdout'
        print(f"Holdout: dataset_slice=='holdout'  → {int(holdout_mask.sum()):,} rows")
    else:
        # Fall back to 75% calendar
        cutoff = int(len(merged) * 0.75)
        holdout_mask = pd.Series([False] * len(merged), index=merged.index)
        holdout_mask.iloc[cutoff:] = True
        print(f"Holdout: last 25% rows  → {int(holdout_mask.sum()):,} rows")
    print()

    # ── Save enriched parquet ──
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"💾 {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print()

    # ════════════════════════════════════════════════════════════════
    # REPORT — comparison of base vs SSL-filtered hit rates
    # ════════════════════════════════════════════════════════════════
    print("═" * 70)
    print("        DAY-TRADE × SSL INTEGRATION REPORT (HOLDOUT)")
    print("═" * 70)
    print()

    holdout = merged.loc[holdout_mask & (merged['event_direction'] != 0)].copy()
    print(f"Holdout directional bars: {len(holdout):,}")

    # Events-only subset (event_flag == 1)
    events_h = holdout.loc[holdout['event_flag'] == 1].copy()
    print(f"Holdout day_trade events: {len(events_h):,}")
    if len(events_h) == 0:
        print("⚠️  No events in holdout — try broader holdout range.")
        return

    # SSL coverage on events
    cov_ssl = events_h['emb_valid'].fillna(False).mean()
    print(f"  SSL embedding coverage: {cov_ssl*100:.1f}% of events")
    print()

    # ── Stratum 1: base (no SSL filter) ──
    print("─── Hit rate at horizon 6 (90 min target) ───")
    base_hit_6, n_base = _hit_rate_at_horizon(events_h, 6)
    print(f"  [BASE] all events:                       hit={base_hit_6:.3f}  n={n_base}")

    for h in (6, 24):
        agree_col = f'ssl_agree_h{h}'
        if agree_col not in events_h.columns:
            continue
        agreed = events_h.loc[events_h[agree_col].fillna(False)].copy()
        disagreed = events_h.loc[
            events_h['emb_valid'].fillna(False)
            & (events_h[agree_col].fillna(False) == False)
            & (events_h['event_direction'] != 0)
        ].copy()
        hit_agree, n_agree = _hit_rate_at_horizon(agreed, 6)
        hit_disagree, n_dis = _hit_rate_at_horizon(disagreed, 6)
        print(f"  [SSL h={h} AGREE]   filter:                hit={hit_agree:.3f}  n={n_agree}  "
              f"({_safe_div(n_agree, n_base)*100:.0f}% retained)")
        print(f"  [SSL h={h} DISAGREE] would-skip:           hit={hit_disagree:.3f}  n={n_dis}")

    # ── Stratum 2: high conviction (top-quintile rank + agree) ──
    print()
    print(f"─── High-conviction filter (rank ≥ {args.top_quintile}) ───")
    for h in (6, 24):
        rank_col = f'ssl_rank_{h}'
        sc_col = f'signed_score_{h}'
        if rank_col not in events_h.columns:
            continue
        # Long: high score & event_dir==1; Short: low score & event_dir==-1
        long_hc = events_h.loc[
            (events_h[rank_col] >= args.top_quintile)
            & (events_h['event_direction'] == 1)
            & events_h['emb_valid'].fillna(False)
        ]
        short_hc = events_h.loc[
            (events_h[rank_col] <= (1.0 - args.top_quintile))
            & (events_h['event_direction'] == -1)
            & events_h['emb_valid'].fillna(False)
        ]
        hc = pd.concat([long_hc, short_hc])
        hit_hc, n_hc = _hit_rate_at_horizon(hc, 6)
        print(f"  [SSL h={h} HIGH-CONV] long top {(1-args.top_quintile)*100:.0f}%, "
              f"short bot {(1-args.top_quintile)*100:.0f}%: "
              f"hit={hit_hc:.3f}  n={n_hc}  ({_safe_div(n_hc, n_base)*100:.0f}% retained)")

    # ── Stratum 3: by SSL high-magnitude regardless of agreement ──
    print()
    print("─── Signal-quality breakdown (day_trade's own tiers) ───")
    if 'signal_quality' in events_h.columns:
        for tier in sorted(events_h['signal_quality'].dropna().unique()):
            sub = events_h.loc[events_h['signal_quality'] == tier]
            hit_t, n_t = _hit_rate_at_horizon(sub, 6)
            print(f"  signal_quality={int(tier)}: hit={hit_t:.3f}  n={n_t}")

    # ── Stratum 4: path_outcome decode if available ──
    print()
    print("─── path_outcome on holdout events ───")
    if 'path_outcome' in events_h.columns:
        po = events_h['path_outcome'].value_counts().sort_index()
        print(f"  raw distribution: {po.to_dict()}")

    # ── Stratum 5: outputs ──
    metrics = {
        'holdout_directional_bars': int(len(holdout)),
        'holdout_events': int(len(events_h)),
        'ssl_coverage': float(cov_ssl),
        'base_hit_rate_h6': float(base_hit_6),
        'base_n': int(n_base),
    }
    for h in (6, 24):
        agree_col = f'ssl_agree_h{h}'
        if agree_col not in events_h.columns:
            continue
        agreed = events_h.loc[events_h[agree_col].fillna(False)].copy()
        h_a, n_a = _hit_rate_at_horizon(agreed, 6)
        metrics[f'ssl_h{h}_agree_hit'] = h_a
        metrics[f'ssl_h{h}_agree_n'] = n_a
        metrics[f'ssl_h{h}_lift'] = (h_a - base_hit_6) if not np.isnan(h_a) else None

    report_path = out_path.parent / 'integration_report.json'
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    print()
    print(f"💾 {report_path}")
    print()
    print("═" * 70)


if __name__ == '__main__':
    main()
