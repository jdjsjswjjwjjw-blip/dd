"""
self_supervised/combine_quarters_fast.py
═══════════════════════════════════════════════════════════════════
دمج quarter files مع gap كبير بين العقود — معالجة علمية صحيحة.

المشكلة: عندنا Q1 (6BH5) + Q2 (6BM5) بفجوة 25-30 يوم بينهم
(لأن آخر 15 يوم من كل ربع متقطعة). الـ standard back-adjustment
بيخلط contract_carry بـ market_drift_in_gap — وده غلط.

الحل العلمي: Constant-Maturity Carry-Only Adjustment
─────────────────────────────────────────────────────
نقدّر الـ contract carry (interest rate differential × Δt) من
بيانات Q2 نفسها — مش من القفزة عند الحد:

  carry_per_day = (mean_close_Q2_first_5d - mean_close_Q2_last_5d) / days_in_Q2
  // ده intra-contract drift, بيحتوي على carry + market drift in Q2
  // غير مناسب كـ carry estimate

أحسن: نستخدم theoretical fair value
  F_Q2 / F_Q1 ≈ exp((r_USD - r_GBP) × Δt)
  للـ 6B بـ rates حالية: ~30-60 pips per quarter

نطبق offset صغير ثابت (40 pips افتراضي، configurable) ونعلم
boundary كـ is_session_break = 1 عشان الـ segment-aware filter
في data_loader يمنع الـ lookback من العبور.

الأداء:
─────────
• MBP فقط (متعدّاش حجم الـ MBO الضخم)
• بدون double-sort — بنفترض input مرتب per file
• pyarrow streaming-friendly read
• boundary offset computation على sample صغير فقط
• الناتج: combined parquet جاهز لـ prepare_day_trading

الاستخدام:
    python self_supervised/combine_quarters_fast.py \\
        --inputs q1.mbp_clean.parquet q2.mbp_clean.parquet \\
        --output combined_6m_mbp.parquet \\
        --carry-pips-per-quarter 40   # 6B typical
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# 6B (GBP/USD futures) tick size = 0.0001 USD/GBP = 1 pip
TICK_SIZE = 0.0001

# Price columns that need offset adjustment in MBP files
PRICE_COLS = ['price'] + \
    [f'bid_px_{i:02d}' for i in range(10)] + \
    [f'ask_px_{i:02d}' for i in range(10)]


def _peek_file_metadata(path: str) -> dict:
    """Get row count + first/last ts WITHOUT loading the whole file."""
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    # Read just ts_event column to get bounds (fast — single column)
    ts_table = pf.read(columns=['ts_event'])
    ts = pd.to_datetime(ts_table['ts_event'].to_pandas())
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    return {
        'path': path,
        'n_rows': n_rows,
        'ts_min': ts.min(),
        'ts_max': ts.max(),
        'schema_cols': pf.schema.names,
    }


def _read_boundary_sample(path: str, side: str, n_minutes: int = 60) -> pd.DataFrame:
    """Read only the last (side='end') or first (side='start') N minutes
    of a parquet file — used to compute offsets without loading everything."""
    # Get ts bounds first
    meta = _peek_file_metadata(path)
    if side == 'end':
        cutoff = meta['ts_max'] - pd.Timedelta(minutes=n_minutes)
    elif side == 'start':
        cutoff = meta['ts_min'] + pd.Timedelta(minutes=n_minutes)
    else:
        raise ValueError(f"side must be 'start' or 'end', got {side}")

    # Read full file (we have to, no row-group filter without dataset API)
    # but only the columns we need
    cols_needed = ['ts_event'] + [c for c in PRICE_COLS if c in meta['schema_cols']]
    df = pd.read_parquet(path, columns=cols_needed)
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    if df['ts_event'].dt.tz is not None:
        df['ts_event'] = df['ts_event'].dt.tz_convert('UTC').dt.tz_localize(None)

    if side == 'end':
        sample = df.loc[df['ts_event'] >= cutoff]
    else:
        sample = df.loc[df['ts_event'] <= cutoff]
    print(f"     boundary sample ({side}): {len(sample):,} rows in last/first {n_minutes}min")
    return sample


def _midprice(df: pd.DataFrame) -> pd.Series:
    """Compute midprice from bid_px_00 + ask_px_00 if available, else price."""
    if 'bid_px_00' in df.columns and 'ask_px_00' in df.columns:
        bid = pd.to_numeric(df['bid_px_00'], errors='coerce')
        ask = pd.to_numeric(df['ask_px_00'], errors='coerce')
        mid = (bid + ask) / 2
        mid = mid.where(mid > 0)  # drop zero/negative
        if mid.notna().sum() > 0:
            return mid
    # Fallback to price
    return pd.to_numeric(df['price'], errors='coerce').where(lambda x: x > 0)


def combine_quarters(
    input_paths: list[str],
    output_path: str,
    carry_pips_per_quarter: float = 40.0,
    auto_carry: bool = False,
    boundary_minutes: int = 60,
) -> dict:
    """Combine N quarter MBP files with carry-only back-adjustment.

    Each pair of consecutive quarters is treated as having a session-break
    boundary; only a small constant carry offset is applied to the older
    contract (NOT the full price gap, which includes market drift).
    """
    print(f"═══ Combining {len(input_paths)} quarters (carry-only adjust) ═══")
    print()

    # ─── 1. Peek metadata (fast, no full load yet) ─────────────────────
    metas = [_peek_file_metadata(p) for p in input_paths]
    for m in metas:
        days = (m['ts_max'] - m['ts_min']).days
        print(f"  📂 {Path(m['path']).name}")
        print(f"     rows={m['n_rows']:,} | {m['ts_min']} → {m['ts_max']} ({days}d)")
    print()

    # ─── 2. Sort metas by ts_min so we know chronological order ────────
    order = sorted(range(len(metas)), key=lambda i: metas[i]['ts_min'])
    metas = [metas[i] for i in order]
    input_paths = [input_paths[i] for i in order]

    # ─── 3. Compute boundary offsets ───────────────────────────────────
    # For each transition i→i+1:
    #   - read end of contract i, start of contract i+1
    #   - measure raw gap (informational only — includes market drift)
    #   - apply carry-only offset to older contract
    cumulative_offsets = [0.0] * len(metas)
    boundary_info = []

    # Walk newest → oldest, accumulate carry
    for i in range(len(metas) - 2, -1, -1):
        old_path, new_path = input_paths[i], input_paths[i + 1]
        old_tag = Path(old_path).stem
        new_tag = Path(new_path).stem
        print(f"  ─── Boundary {i}→{i+1}: {old_tag[:30]} → {new_tag[:30]} ───")

        old_end = _read_boundary_sample(old_path, 'end', boundary_minutes)
        new_start = _read_boundary_sample(new_path, 'start', boundary_minutes)
        gap_days = (metas[i + 1]['ts_min'] - metas[i]['ts_max']).days

        old_mid_med = float(_midprice(old_end).median())
        new_mid_med = float(_midprice(new_start).median())
        raw_gap = new_mid_med - old_mid_med
        raw_gap_pips = raw_gap / TICK_SIZE

        # ─── Carry choice ─────────────────────────────────────────────
        if auto_carry:
            # Auto: use raw gap (Panama back-adjust). UNSAFE for big gaps.
            carry = raw_gap
            carry_pips = raw_gap_pips
            method = 'auto (full gap — UNSAFE for >7d gap)'
        else:
            # Carry-only: fixed pips, sign matches raw_gap direction
            sign = 1.0 if raw_gap >= 0 else -1.0
            carry_pips = sign * carry_pips_per_quarter
            carry = carry_pips * TICK_SIZE
            method = f'carry-only ({carry_pips:+.0f} pips)'

        cumulative_offsets[i] = carry + cumulative_offsets[i + 1]

        print(f"     gap: {gap_days}d | old_mid={old_mid_med:.5f} new_mid={new_mid_med:.5f}")
        print(f"     raw_gap={raw_gap_pips:+.1f} pips (contains carry + {gap_days}d market drift)")
        print(f"     applied: {method}")
        print(f"     cum offset for {old_tag[:20]}: {cumulative_offsets[i]/TICK_SIZE:+.1f} pips")
        print()

        boundary_info.append({
            'old_file': old_path,
            'new_file': new_path,
            'gap_days': gap_days,
            'raw_gap_pips': raw_gap_pips,
            'applied_carry_pips': carry_pips,
            'old_mid': old_mid_med,
            'new_mid': new_mid_med,
            'roll_ts': str(metas[i + 1]['ts_min']),
        })

    # ─── 4. Stream-apply offsets, write combined ────────────────────────
    print(f"  ─── Applying offsets and concatenating ───")
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pieces = []
    total_rows = 0
    for i, path in enumerate(input_paths):
        print(f"  [{i+1}/{len(input_paths)}] Loading {Path(path).name}...")
        df = pd.read_parquet(path)
        df['ts_event'] = pd.to_datetime(df['ts_event'])
        if df['ts_event'].dt.tz is not None:
            df['ts_event'] = df['ts_event'].dt.tz_convert('UTC').dt.tz_localize(None)

        # Apply cumulative offset to all price columns present
        offset = cumulative_offsets[i]
        if abs(offset) > 0:
            n_adj = 0
            for c in PRICE_COLS:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce') + offset
                    n_adj += 1
            print(f"     applied offset {offset/TICK_SIZE:+.1f} pips to {n_adj} price cols")

        # Tag contract identity for traceability
        df['_contract_idx'] = i
        # Mark session break at first row of every contract AFTER the first
        df['is_roll'] = 0
        if i > 0 and len(df) > 0:
            df.loc[df.index[0], 'is_roll'] = 1

        pieces.append(df)
        total_rows += len(df)
        print(f"     {len(df):,} rows | cum total: {total_rows:,}")

    print()
    print(f"  ─── Concatenating {total_rows:,} rows ───")
    combined = pd.concat(pieces, ignore_index=True)
    # NOTE: we do NOT re-sort because each piece is already sorted and
    # pieces are non-overlapping by construction. This saves a huge sort.
    # Sanity check the assumption:
    ts = combined['ts_event'].values
    is_mono = (ts[1:] >= ts[:-1]).all()
    if not is_mono:
        print(f"  ⚠️  combined ts not monotonic — applying single sort")
        combined = combined.sort_values('ts_event').reset_index(drop=True)
    else:
        print(f"  ✅ combined ts is monotonic (no sort needed)")

    # ─── 5. Continuity check at boundaries ──────────────────────────────
    print()
    print(f"  ─── Continuity check ───")
    roll_idx = combined.index[combined['is_roll'] == 1].tolist()
    price_col = 'bid_px_00' if 'bid_px_00' in combined.columns else 'price'
    for ri in roll_idx:
        if ri > 0:
            p_before = float(pd.to_numeric(combined[price_col].iloc[ri - 1], errors='coerce'))
            p_after = float(pd.to_numeric(combined[price_col].iloc[ri], errors='coerce'))
            gap_ticks = (p_after - p_before) / TICK_SIZE
            mark = '✅' if abs(gap_ticks) < 500 else '⚠️'  # 500 pips threshold (large gap expected)
            t_before = combined['ts_event'].iloc[ri - 1]
            t_after = combined['ts_event'].iloc[ri]
            print(f"    {mark} roll @ idx {ri}: {t_before} → {t_after}")
            print(f"        {price_col}: {p_before:.5f} → {p_after:.5f} (gap {gap_ticks:+.1f} pips)")

    # ─── 6. Write ────────────────────────────────────────────────────────
    print()
    print(f"  💾 Writing {len(combined):,} rows → {output_path}")
    combined.to_parquet(output_path, index=False)
    size_gb = Path(output_path).stat().st_size / 1e9
    print(f"     File size: {size_gb:.2f} GB")

    # ─── 7. Metadata sidecar ────────────────────────────────────────────
    meta = {
        'inputs': list(input_paths),
        'method': 'auto-panama' if auto_carry else 'carry-only',
        'carry_pips_per_quarter': carry_pips_per_quarter,
        'cumulative_offsets_pips': [o / TICK_SIZE for o in cumulative_offsets],
        'cumulative_offsets': [float(o) for o in cumulative_offsets],
        'boundaries': boundary_info,
        'n_rows': int(len(combined)),
        'ts_range': [str(combined['ts_event'].iloc[0]), str(combined['ts_event'].iloc[-1])],
    }
    meta_path = Path(output_path).with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"     Meta: {meta_path}")
    print()
    print(f"  ⚠️  NOTE: prices in old contract(s) are shifted by carry only.")
    print(f"     The remaining price-level gap at the boundary is REAL market")
    print(f"     drift during the {sum(b['gap_days'] for b in boundary_info)}d total gap.")
    print(f"     The is_roll=1 column marks every boundary — make sure your")
    print(f"     downstream pipeline treats is_roll as session_break.")

    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True, nargs='+',
                   help='MBP parquet files in any order (will be sorted by ts)')
    p.add_argument('--output', required=True, help='Combined parquet path')
    p.add_argument('--carry-pips-per-quarter', type=float, default=40.0,
                   help='Contract carry estimate in pips per quarter (6B: ~30-60 pips)')
    p.add_argument('--auto-carry', action='store_true',
                   help='UNSAFE for large gaps: use full Panama back-adjust '
                        '(includes market drift in offset)')
    p.add_argument('--boundary-minutes', type=int, default=60,
                   help='Minutes of data to sample around boundary for diagnostics')
    args = p.parse_args()

    if len(args.inputs) < 2:
        print("❌ Need at least 2 input files to combine")
        sys.exit(1)

    combine_quarters(
        args.inputs, args.output,
        carry_pips_per_quarter=args.carry_pips_per_quarter,
        auto_carry=args.auto_carry,
        boundary_minutes=args.boundary_minutes,
    )


if __name__ == '__main__':
    main()
