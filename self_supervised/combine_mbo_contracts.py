"""
self_supervised/combine_mbo_contracts.py
═══════════════════════════════════════════════════════════
يدمج N من MBO contract files في continuous series باستخدام
back-adjustment (Panama method).

ليه دا مهم: 6B futures بتنتهي كل 3 شهور (H/M/U/Z). لو دمجت
contracts كأنها متصلة بدون adjustment، هتلاقي price gap عند
كل rollover (الـ FX futures عادة فيها front-month premium بسبب
interest rate differential).

ال back-adjustment الـ standard:
  1. خد آخر سعر للـ old contract وأول سعر للـ new contract عند
     نفس الـ ts (أو أقرب overlap).
  2. offset = new_price - old_price
  3. اضف الـ offset لكل الأسعار التاريخية في الـ old contract.
  4. الـ new contract يفضل بـ real prices.
  5. log returns حقيقية في كل contract؛ الـ return عبر الـ roll
     بقت 0 (لأن الـ adjustment ألغى الـ artificial gap).

طريقة التركيز للسبب: log(p_new / p_old_adj) ≈ 0 حول الـ roll.

الاستخدام:
    python self_supervised/combine_mbo_contracts.py \\
        --inputs glbx-6BZ4.parquet glbx-6BH5.parquet glbx-6BM5.parquet \\
        --output glbx-6B-2024-12-to-2025-06.parquet \\
        --roll-policy volume      # auto-detect roll by daily volume crossover
        # OR --roll-policy date --roll-dates 2025-03-14 2025-06-13
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TRADE_ACTIONS = {'T', 'F', 'TRADE', 'EXECUTE', 'E', '0'}


def _load_mbo(path: str, tag: str) -> pd.DataFrame:
    """Load one MBO parquet, tag with contract id, normalize tz."""
    print(f"  📂 {path} → {tag}")
    df = pd.read_parquet(path)
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    if df['ts_event'].dt.tz is not None:
        df['ts_event'] = df['ts_event'].dt.tz_convert('UTC').dt.tz_localize(None)
    df['_contract'] = tag
    df = df.sort_values('ts_event').reset_index(drop=True)
    print(f"     {len(df):,} ticks | {df['ts_event'].iloc[0]} → {df['ts_event'].iloc[-1]}")
    return df


def _detect_roll_by_volume(
    contracts: list[pd.DataFrame], freq: str = '1D',
) -> list[pd.Timestamp]:
    """For each pair of consecutive contracts, find the day where new contract's
    daily trade volume first exceeds old contract's.

    Returns list of roll timestamps (one per transition).
    """
    rolls = []
    for i in range(len(contracts) - 1):
        old, new = contracts[i], contracts[i + 1]
        # Daily trade volume per contract
        def _daily_vol(df):
            is_trade = df['action'].astype(str).str.upper().isin(TRADE_ACTIONS) \
                if 'action' in df.columns else pd.Series(True, index=df.index)
            sz = pd.to_numeric(df['size'], errors='coerce').fillna(0.0).where(is_trade, 0.0)
            return sz.groupby(df['ts_event'].dt.floor(freq)).sum()

        v_old = _daily_vol(old)
        v_new = _daily_vol(new)
        # Find overlap window
        overlap_start = max(v_old.index.min(), v_new.index.min())
        overlap_end = min(v_old.index.max(), v_new.index.max())
        if overlap_start > overlap_end:
            # No overlap — fall back to midpoint between contracts
            roll_ts = old['ts_event'].iloc[-1] + (
                (new['ts_event'].iloc[0] - old['ts_event'].iloc[-1]) / 2
            )
            print(f"  ⚠️  No overlap between {old['_contract'].iloc[0]} and "
                  f"{new['_contract'].iloc[0]} — using midpoint {roll_ts}")
        else:
            overlap_days = pd.date_range(overlap_start, overlap_end, freq=freq)
            roll_day = None
            for d in overlap_days:
                if v_new.get(d, 0.0) > v_old.get(d, 0.0):
                    roll_day = d
                    break
            if roll_day is None:
                roll_day = overlap_end
                print(f"  ⚠️  Volume never crossed; rolling at last overlap day "
                      f"{roll_day}")
            roll_ts = roll_day + pd.Timedelta(hours=17)  # end-of-session roll
            print(f"  ✅ Roll detected: {old['_contract'].iloc[0]} → "
                  f"{new['_contract'].iloc[0]} @ {roll_ts}")
        rolls.append(roll_ts)
    return rolls


def _compute_offset(
    old_df: pd.DataFrame, new_df: pd.DataFrame, roll_ts: pd.Timestamp,
    window_minutes: int = 30,
) -> float:
    """Compute back-adjustment offset using prices in a ±window around roll_ts.

    offset = median(new_price_in_window) - median(old_price_in_window)
    Apply to old contract: old_adjusted = old + offset
    Such that median price is continuous across the roll.
    """
    win = pd.Timedelta(minutes=window_minutes)
    is_trade_old = (
        old_df['action'].astype(str).str.upper().isin(TRADE_ACTIONS)
        if 'action' in old_df.columns else pd.Series(True, index=old_df.index)
    )
    is_trade_new = (
        new_df['action'].astype(str).str.upper().isin(TRADE_ACTIONS)
        if 'action' in new_df.columns else pd.Series(True, index=new_df.index)
    )

    old_win = old_df.loc[
        is_trade_old
        & (old_df['ts_event'] >= roll_ts - win)
        & (old_df['ts_event'] <= roll_ts + win)
    ]
    new_win = new_df.loc[
        is_trade_new
        & (new_df['ts_event'] >= roll_ts - win)
        & (new_df['ts_event'] <= roll_ts + win)
    ]

    if len(old_win) == 0 or len(new_win) == 0:
        # Fallback: nearest-tick
        old_px = float(pd.to_numeric(old_df.loc[is_trade_old, 'price'], errors='coerce').iloc[-1])
        new_px = float(pd.to_numeric(new_df.loc[is_trade_new, 'price'], errors='coerce').iloc[0])
        offset = new_px - old_px
        print(f"     fallback: last_old={old_px:.5f}, first_new={new_px:.5f}, "
              f"offset={offset:+.5f}")
        return offset

    old_med = float(pd.to_numeric(old_win['price'], errors='coerce').median())
    new_med = float(pd.to_numeric(new_win['price'], errors='coerce').median())
    offset = new_med - old_med
    print(f"     window stats: old_med={old_med:.5f} (n={len(old_win)}), "
          f"new_med={new_med:.5f} (n={len(new_win)}), offset={offset:+.5f} "
          f"({offset/0.0001:+.1f} ticks)")
    return offset


def combine_contracts(
    input_paths: list[str],
    output_path: str,
    roll_policy: str = 'volume',
    roll_dates: list[str] | None = None,
    window_minutes: int = 30,
) -> dict:
    """Combine N contract MBO files into a single back-adjusted continuous series.

    Output:
      • Concatenated, sorted by ts_event
      • Old-contract prices back-adjusted so the series is continuous
      • Adds 'is_roll' column (1 for the bar covering each roll, else 0)
      • Drops any post-roll old-contract ticks (no double-counting)
    """
    print(f"═══ Combining {len(input_paths)} MBO contracts ═══")

    contracts = []
    for i, path in enumerate(input_paths):
        tag = Path(path).stem.split('_')[-1] if '_' in Path(path).stem else f"contract_{i}"
        contracts.append(_load_mbo(path, tag))

    # Sort contracts by their first ts
    contracts.sort(key=lambda d: d['ts_event'].iloc[0])

    # Determine roll timestamps
    if roll_policy == 'volume':
        rolls = _detect_roll_by_volume(contracts)
    elif roll_policy == 'date':
        if not roll_dates or len(roll_dates) != len(contracts) - 1:
            raise ValueError(
                f"--roll-policy date requires exactly {len(contracts) - 1} --roll-dates"
            )
        rolls = [pd.to_datetime(d) for d in roll_dates]
    else:
        raise ValueError(f"Unknown roll_policy: {roll_policy}")

    # Compute back-adjustments. Walk from newest → oldest, accumulating offsets.
    # final_offset[i] = sum of offsets needed to bring contract i in line with
    # the newest contract.
    cumulative_offsets = [0.0] * len(contracts)
    for i in range(len(contracts) - 2, -1, -1):
        offset = _compute_offset(
            contracts[i], contracts[i + 1], rolls[i],
            window_minutes=window_minutes,
        )
        # Newest contract's "stack offset" is cumulative_offsets[i+1]
        # The offset we just computed brings contract i to contract i+1's frame
        # Then we add contract i+1's stack offset to bring it to newest
        cumulative_offsets[i] = offset + cumulative_offsets[i + 1]

    print()
    print("  Cumulative back-adjustments (additive to price):")
    for i, c in enumerate(contracts):
        print(f"    {c['_contract'].iloc[0]:15s}: {cumulative_offsets[i]:+.5f} "
              f"({cumulative_offsets[i]/0.0001:+.1f} ticks)")

    # Apply offsets and filter to non-overlapping windows
    pieces = []
    for i, c in enumerate(contracts):
        # Window: [-inf, roll[i]] for i=0..N-2; [roll[N-2], +inf] for last
        lo = rolls[i - 1] if i > 0 else pd.Timestamp.min
        hi = rolls[i] if i < len(contracts) - 1 else pd.Timestamp.max
        piece = c.loc[(c['ts_event'] >= lo) & (c['ts_event'] < hi)].copy()
        # Apply back-adjustment to price
        piece['price'] = pd.to_numeric(piece['price'], errors='coerce') + cumulative_offsets[i]
        # Tag roll boundary (first bar of new contract piece, except first piece)
        piece['is_roll'] = 0
        if i > 0 and len(piece) > 0:
            piece.loc[piece.index[0], 'is_roll'] = 1
        pieces.append(piece)
        print(f"  {c['_contract'].iloc[0]:15s}: window [{lo}, {hi}) → "
              f"{len(piece):,} ticks (adjusted)")

    combined = pd.concat(pieces, ignore_index=True).sort_values('ts_event').reset_index(drop=True)
    # Drop helper column
    combined = combined.drop(columns=['_contract'])

    # Sanity: continuity check at roll boundaries
    roll_idx = combined.index[combined['is_roll'] == 1].tolist()
    print()
    print(f"  Continuity check at {len(roll_idx)} roll(s):")
    for ri in roll_idx:
        if ri > 0:
            p_before = float(combined['price'].iloc[ri - 1])
            p_after = float(combined['price'].iloc[ri])
            gap_ticks = (p_after - p_before) / 0.0001
            mark = '✅' if abs(gap_ticks) < 50 else '⚠️'
            print(f"    {mark} {combined['ts_event'].iloc[ri]}: "
                  f"prev={p_before:.5f}, next={p_after:.5f}, "
                  f"gap={gap_ticks:+.1f} ticks")

    print()
    print(f"  💾 Writing {len(combined):,} ticks → {output_path}")
    combined.to_parquet(output_path, index=False)
    size_gb = Path(output_path).stat().st_size / 1e9
    print(f"     File size: {size_gb:.2f} GB")

    # Metadata sidecar
    meta = {
        'inputs': list(input_paths),
        'roll_policy': roll_policy,
        'roll_timestamps': [str(r) for r in rolls],
        'cumulative_offsets': [float(o) for o in cumulative_offsets],
        'n_ticks': int(len(combined)),
        'ts_range': [str(combined['ts_event'].iloc[0]), str(combined['ts_event'].iloc[-1])],
    }
    meta_path = Path(output_path).with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"     Meta: {meta_path}")

    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True, nargs='+',
                  help='List of MBO contract parquet files (any order)')
    p.add_argument('--output', required=True, help='Combined continuous parquet')
    p.add_argument('--roll-policy', default='volume', choices=['volume', 'date'],
                  help='How to detect roll points: by volume crossover or fixed dates')
    p.add_argument('--roll-dates', nargs='*', default=None,
                  help='Required if --roll-policy date: N-1 dates for N contracts')
    p.add_argument('--window-minutes', type=int, default=30,
                  help='Window around roll for computing offset (default 30 min)')
    args = p.parse_args()

    if len(args.inputs) < 2:
        print("❌ Need at least 2 input contracts to combine")
        sys.exit(1)

    combine_contracts(
        args.inputs, args.output,
        roll_policy=args.roll_policy,
        roll_dates=args.roll_dates,
        window_minutes=args.window_minutes,
    )


if __name__ == '__main__':
    main()
