"""
self_supervised/combine_quarter_artifacts.py
═══════════════════════════════════════════════════════════════════
دمج كل الـ SSL artifacts بين الربعين في خطوة واحدة:
  • bars parquet            (مع carry-only price offset + is_roll marker)
  • lob_tensors npy         (np.concatenate axis=0 — bar-aligned)
  • order_features npy      (np.concatenate axis=0 — bar-aligned)
  • order_masks npy         (np.concatenate axis=0 — bar-aligned)

الفلسفة:
─────────
كل من lob و order_features متماثلين مع الـ bars (entry لكل bar).
بنبني عمل prepare_day_trading + build_order_batches لكل ربع منفصل،
وبعدين بندمج النواتج على مستوى الـ bar — مش على مستوى MBO الخام.

ده بيتجنب:
  ‣ تحميل MBO files العملاقة في الـ RAM
  ‣ double-sort على ملايين الـ ticks
  ‣ تشويه back-adjustment للـ gap الكبير بين الربعين (~25-30 يوم)
  ‣ مشاكل alignment بين bar indices والـ npy arrays

بيضمن:
  ‣ carry-only price adjustment صح (40 pips افتراضي للـ 6B)
  ‣ is_roll=1 + is_session_break=1 عند كل حد ربع
  ‣ alignment محفوظ بين bars/lob/orders
  ‣ خاصية الـ segment-aware filter في data_loader شغّالة

الاستخدام:
    python self_supervised/combine_quarter_artifacts.py \\
        --quarter-dirs /workspace/clean/q1_6BH5 /workspace/clean/q2_6BM5 \\
        --output-dir /workspace/clean/combined_6m \\
        --bars-name day_trading_features.parquet \\
        --lob-name lob_tensors.npy \\
        --order-features-name order_features.npy \\
        --order-masks-name order_masks.npy \\
        --carry-pips-per-quarter 40

كل quarter-dir لازم يحتوي على:
    <bars-name>          (parquet)
    <lob-name>           (npy: shape (N_bars, T, P, C))
    <order-features-name> (npy: shape (N_bars, T, N_orders, F))
    <order-masks-name>    (npy: shape (N_bars, T, N_orders))
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# 6B (GBP/USD futures) tick size
TICK_SIZE = 0.0001

# Price columns that may need offset adjustment in bar parquet files
PRICE_COLS = [
    'open', 'high', 'low', 'close',  # OHLC
    'vwap',                          # volume-weighted avg price
    'price',                         # generic price
]
# Plus any rolling pivot / level columns that are absolute prices
PRICE_PREFIXES = (
    'bid_px_', 'ask_px_',            # book levels (if carried over)
    'pivot_', 'resistance_', 'support_',
    'session_high', 'session_low',
    'london_high', 'london_low',
    'rolling_vwap_',
)


def _is_price_col(name: str) -> bool:
    if name in PRICE_COLS:
        return True
    return any(name.startswith(p) for p in PRICE_PREFIXES)


def _midprice(df: pd.DataFrame) -> pd.Series:
    """Mid price from bid_px_00/ask_px_00, fall back to close, then price."""
    if 'bid_px_00' in df.columns and 'ask_px_00' in df.columns:
        bid = pd.to_numeric(df['bid_px_00'], errors='coerce')
        ask = pd.to_numeric(df['ask_px_00'], errors='coerce')
        mid = (bid + ask) / 2
        mid = mid.where(mid > 0)
        if mid.notna().sum() > 0:
            return mid
    for fallback in ('close', 'price'):
        if fallback in df.columns:
            s = pd.to_numeric(df[fallback], errors='coerce').where(lambda x: x > 0)
            if s.notna().sum() > 0:
                return s
    raise ValueError("No price column found (need bid_px_00+ask_px_00, close, or price)")


def _load_quarter(qdir: Path, names: dict) -> dict:
    """Load all 4 artifacts for one quarter and validate alignment."""
    out = {}
    bars_path = qdir / names['bars']
    if not bars_path.exists():
        raise FileNotFoundError(f"bars not found: {bars_path}")
    bars = pd.read_parquet(bars_path)
    if 'ts_event' in bars.columns:
        bars['ts_event'] = pd.to_datetime(bars['ts_event'])
        if bars['ts_event'].dt.tz is not None:
            bars['ts_event'] = bars['ts_event'].dt.tz_convert('UTC').dt.tz_localize(None)
    n_bars = len(bars)
    print(f"  📊 bars: {n_bars:,} rows | {bars['ts_event'].iloc[0]} → {bars['ts_event'].iloc[-1]}")
    out['bars'] = bars

    for key, fname in [('lob', names['lob']),
                        ('order_features', names['order_features']),
                        ('order_masks', names['order_masks'])]:
        if fname is None:
            print(f"  ⏭️  {key}: skipped (no name provided)")
            out[key] = None
            continue
        p = qdir / fname
        if not p.exists():
            print(f"  ⚠️  {key}: NOT FOUND at {p} — will skip this artifact")
            out[key] = None
            continue
        arr = np.load(p)
        print(f"  📦 {key}: shape={arr.shape} dtype={arr.dtype} ({arr.nbytes / 1e9:.2f} GB)")
        if arr.shape[0] != n_bars:
            raise ValueError(
                f"{key} first dim ({arr.shape[0]}) != bars count ({n_bars}) in {qdir} — "
                f"misaligned artifacts! Re-run prepare_day_trading + build_order_batches "
                f"with the same lookback/freq settings."
            )
        out[key] = arr
    return out


def combine_artifacts(
    quarter_dirs: list[Path],
    output_dir: Path,
    names: dict,
    carry_pips_per_quarter: float = 40.0,
    auto_carry: bool = False,
) -> dict:
    """Combine bars + lob + orders across N quarters with carry-only adjust."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── 1. Load all quarters ──────────────────────────────────────────
    print(f"═══ Combining {len(quarter_dirs)} quarter artifact dirs ═══")
    print()
    quarters = []
    for i, qd in enumerate(quarter_dirs):
        print(f"  ─── Quarter {i+1}/{len(quarter_dirs)}: {qd} ───")
        quarters.append(_load_quarter(qd, names))
        print()

    # ─── 2. Sort by ts_min (chronological) ─────────────────────────────
    order = sorted(range(len(quarters)),
                   key=lambda i: quarters[i]['bars']['ts_event'].iloc[0])
    quarters = [quarters[i] for i in order]
    quarter_dirs = [quarter_dirs[i] for i in order]

    # ─── 3. Compute cumulative carry offsets ───────────────────────────
    print(f"  ─── Computing boundary offsets ───")
    cumulative_offsets = [0.0] * len(quarters)
    boundary_info = []
    boundary_minutes_window = 60

    for i in range(len(quarters) - 2, -1, -1):
        old_bars = quarters[i]['bars']
        new_bars = quarters[i + 1]['bars']

        # Sample boundary midprices (last hour of old, first hour of new)
        old_end_ts = old_bars['ts_event'].iloc[-1] - pd.Timedelta(minutes=boundary_minutes_window)
        new_start_ts = new_bars['ts_event'].iloc[0] + pd.Timedelta(minutes=boundary_minutes_window)

        old_window = old_bars.loc[old_bars['ts_event'] >= old_end_ts]
        new_window = new_bars.loc[new_bars['ts_event'] <= new_start_ts]

        old_mid_med = float(_midprice(old_window).median())
        new_mid_med = float(_midprice(new_window).median())
        gap_days = (new_bars['ts_event'].iloc[0] - old_bars['ts_event'].iloc[-1]).days
        raw_gap = new_mid_med - old_mid_med
        raw_gap_pips = raw_gap / TICK_SIZE

        if auto_carry:
            carry = raw_gap
            carry_pips = raw_gap_pips
            method = 'auto-panama (FULL gap — UNSAFE for >7d gap)'
        else:
            sign = 1.0 if raw_gap >= 0 else -1.0
            carry_pips = sign * carry_pips_per_quarter
            carry = carry_pips * TICK_SIZE
            method = f'carry-only ({carry_pips:+.0f} pips)'

        cumulative_offsets[i] = carry + cumulative_offsets[i + 1]

        print(f"    boundary {i}→{i+1}: gap={gap_days}d")
        print(f"      old_mid={old_mid_med:.5f}  new_mid={new_mid_med:.5f}")
        print(f"      raw_gap={raw_gap_pips:+.1f} pips (carry + drift)")
        print(f"      applied: {method}")
        print(f"      cum offset for Q{i+1}: {cumulative_offsets[i]/TICK_SIZE:+.1f} pips")

        boundary_info.append({
            'old_dir': str(quarter_dirs[i]),
            'new_dir': str(quarter_dirs[i + 1]),
            'gap_days': gap_days,
            'raw_gap_pips': raw_gap_pips,
            'applied_carry_pips': carry_pips,
            'roll_ts': str(new_bars['ts_event'].iloc[0]),
        })
    print()

    # ─── 4. Apply offsets to bars + mark is_roll ───────────────────────
    print(f"  ─── Applying offsets to bar prices ───")
    bar_pieces = []
    for i, q in enumerate(quarters):
        bars = q['bars'].copy()
        offset = cumulative_offsets[i]

        if abs(offset) > 0:
            adjusted_cols = []
            for c in bars.columns:
                if _is_price_col(c):
                    bars[c] = pd.to_numeric(bars[c], errors='coerce') + offset
                    adjusted_cols.append(c)
            print(f"    Q{i+1}: offset {offset/TICK_SIZE:+.1f} pips on {len(adjusted_cols)} cols "
                  f"({', '.join(adjusted_cols[:5])}{'...' if len(adjusted_cols) > 5 else ''})")
        else:
            print(f"    Q{i+1}: offset 0 (newest contract, no adjustment)")

        # Mark roll at first bar of every quarter after the first
        bars['is_roll'] = 0
        if 'is_session_break' not in bars.columns:
            bars['is_session_break'] = 0
        if i > 0 and len(bars) > 0:
            first_idx = bars.index[0]
            bars.loc[first_idx, 'is_roll'] = 1
            # Force session break — so segment-aware filter blocks lookback across gap
            bars.loc[first_idx, 'is_session_break'] = 1
        bars['_quarter'] = i
        bar_pieces.append(bars)
    print()

    # ─── 5. Concatenate everything (bars + npy artifacts) ──────────────
    print(f"  ─── Concatenating artifacts ───")
    combined_bars = pd.concat(bar_pieces, ignore_index=True)
    print(f"    bars combined: {len(combined_bars):,} rows")

    # Sanity check chronological order
    ts = combined_bars['ts_event'].values
    if not (ts[1:] >= ts[:-1]).all():
        print(f"    ⚠️  ts not monotonic post-concat — sorting once")
        combined_bars = combined_bars.sort_values('ts_event').reset_index(drop=True)
    else:
        print(f"    ✅ ts monotonic (no sort needed)")

    # Concat npy artifacts
    npy_combined = {}
    for key in ('lob', 'order_features', 'order_masks'):
        arrs = [q[key] for q in quarters if q[key] is not None]
        if len(arrs) != len(quarters):
            print(f"    ⚠️  {key}: skipped (not all quarters have it)")
            continue
        if len(arrs) == 0:
            continue
        # Validate shapes match except first dim
        ref_shape = arrs[0].shape[1:]
        for j, a in enumerate(arrs):
            if a.shape[1:] != ref_shape:
                raise ValueError(
                    f"{key} shape mismatch: Q1 has {ref_shape}, Q{j+1} has {a.shape[1:]} — "
                    f"different lookback/n_orders settings between quarters!"
                )
        combined = np.concatenate(arrs, axis=0)
        print(f"    {key}: combined shape={combined.shape} ({combined.nbytes / 1e9:.2f} GB)")
        npy_combined[key] = combined

    # If we did a post-sort on bars, the npy arrays would be misaligned —
    # so we must re-order them too. Build a permutation from the original
    # row order. The order before sort was [Q1_bars, Q2_bars, ...] aka
    # the same order as the npy concat. If ts was monotonic we're fine.
    # If not, we need a permutation index.
    if not (ts[1:] >= ts[:-1]).all():
        # We have to redo: re-sort bar_pieces alignment carefully
        raise RuntimeError(
            "Bars required re-sorting post-concat, but npy arrays cannot be "
            "trivially permuted. This means quarter file timestamps overlap. "
            "Please check the input quarter dirs for chronological cleanliness."
        )

    # ─── 6. Continuity check ───────────────────────────────────────────
    print()
    print(f"  ─── Continuity check at roll boundaries ───")
    roll_idx = combined_bars.index[combined_bars['is_roll'] == 1].tolist()
    for ri in roll_idx:
        if ri > 0:
            c_before = float(combined_bars['close'].iloc[ri - 1]) if 'close' in combined_bars else None
            c_after = float(combined_bars['close'].iloc[ri]) if 'close' in combined_bars else None
            t_before = combined_bars['ts_event'].iloc[ri - 1]
            t_after = combined_bars['ts_event'].iloc[ri]
            if c_before is not None and c_after is not None:
                gap_pips = (c_after - c_before) / TICK_SIZE
                mark = '✅' if abs(gap_pips) < 500 else '⚠️'
                print(f"    {mark} idx={ri}: {t_before} → {t_after}")
                print(f"        close: {c_before:.5f} → {c_after:.5f} (gap {gap_pips:+.1f} pips)")

    # ─── 7. Write outputs ───────────────────────────────────────────────
    print()
    print(f"  ─── Writing combined artifacts → {output_dir} ───")
    out_bars = output_dir / names['bars']
    combined_bars.to_parquet(out_bars, index=False)
    print(f"    💾 {out_bars.name}: {out_bars.stat().st_size / 1e9:.2f} GB")

    for key, fname_key in [('lob', 'lob'),
                            ('order_features', 'order_features'),
                            ('order_masks', 'order_masks')]:
        if key in npy_combined:
            out_npy = output_dir / names[fname_key]
            np.save(out_npy, npy_combined[key])
            print(f"    💾 {out_npy.name}: {out_npy.stat().st_size / 1e9:.2f} GB")

    # Metadata
    meta = {
        'inputs': [str(q) for q in quarter_dirs],
        'method': 'auto-panama' if auto_carry else 'carry-only',
        'carry_pips_per_quarter': carry_pips_per_quarter,
        'cumulative_offsets_pips': [o / TICK_SIZE for o in cumulative_offsets],
        'cumulative_offsets': [float(o) for o in cumulative_offsets],
        'boundaries': boundary_info,
        'n_bars_per_quarter': [len(q['bars']) for q in quarters],
        'n_bars_combined': int(len(combined_bars)),
        'ts_range': [str(combined_bars['ts_event'].iloc[0]),
                     str(combined_bars['ts_event'].iloc[-1])],
        'roll_indices': roll_idx,
    }
    meta_path = output_dir / 'combine_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"    💾 {meta_path.name}")

    print()
    print(f"  ⚠️  NOTE: only carry was applied to old-contract prices.")
    print(f"     The remaining gap at each roll boundary is REAL market drift")
    print(f"     during the gap days — NOT a price error. The is_session_break=1")
    print(f"     marker at boundary tells the SSL data_loader to block lookbacks")
    print(f"     from crossing the gap (via segment-aware filter).")

    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--quarter-dirs', required=True, nargs='+',
                   help='Quarter dirs each containing bars+lob+order artifacts')
    p.add_argument('--output-dir', required=True, help='Combined output dir')
    p.add_argument('--bars-name', default='day_trading_features.parquet',
                   help='Bar parquet filename inside each quarter dir')
    p.add_argument('--lob-name', default='lob_tensors.npy',
                   help='LOB tensor filename (set to NONE to skip)')
    p.add_argument('--order-features-name', default='order_features.npy',
                   help='Order features filename (set to NONE to skip)')
    p.add_argument('--order-masks-name', default='order_masks.npy',
                   help='Order masks filename (set to NONE to skip)')
    p.add_argument('--carry-pips-per-quarter', type=float, default=40.0,
                   help='Contract carry per quarter in pips (6B: ~30-60)')
    p.add_argument('--auto-carry', action='store_true',
                   help='UNSAFE: use full price gap (includes market drift)')
    args = p.parse_args()

    if len(args.quarter_dirs) < 2:
        print("❌ Need at least 2 quarter dirs to combine")
        sys.exit(1)

    names = {
        'bars': args.bars_name,
        'lob': None if args.lob_name.upper() == 'NONE' else args.lob_name,
        'order_features': None if args.order_features_name.upper() == 'NONE' else args.order_features_name,
        'order_masks': None if args.order_masks_name.upper() == 'NONE' else args.order_masks_name,
    }

    combine_artifacts(
        [Path(q) for q in args.quarter_dirs],
        Path(args.output_dir),
        names,
        carry_pips_per_quarter=args.carry_pips_per_quarter,
        auto_carry=args.auto_carry,
    )


if __name__ == '__main__':
    main()
