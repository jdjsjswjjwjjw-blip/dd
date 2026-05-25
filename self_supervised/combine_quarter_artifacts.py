"""
self_supervised/combine_quarter_artifacts.py
═══════════════════════════════════════════════════════════════════
دمج كل الـ SSL artifacts بين الربعين في خطوة واحدة:
  • bars parquet            (price offset + is_roll + is_session_break marker)
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
  ‣ مشاكل alignment بين bar indices والـ npy arrays

الـ overlap modes (CRITICAL لمعالجة rollover):
──────────────────────────────────────────────
  auto    — افتراضي. كشف الـ overlap تلقائياً:
            • لو فيه overlap (داتا الربعين فيها أيام مشتركة):
              ▸ يحدد roll day = أول يوم volume_Q2 > volume_Q1
              ▸ يشيل Q1 الميت (bars بعد roll_day، low volume)
              ▸ يشيل Q2 الرفيع (bars قبل roll_day، thin)
              ▸ يحسب real offset من ±24h window حوالين roll_day
              ▸ continuity gap بعد adjust = حركة سوق حقيقية فقط (تقريباً صفر)
            • لو مفيش overlap (الداتا متقطعة):
              ▸ fallback لـ carry-only (40 pips افتراضي للـ 6B)
              ▸ continuity gap = real market drift (لا يمكن استرجاعه)

  overlap — يطلب overlap (يطفي error لو مش موجود). الأفضل لو الداتا كاملة.
  carry   — يفرض carry-only حتى لو فيه overlap.

ليه overlap mode أفضل:
─────────────────────────
بدون overlap = آخر 15 يوم من العقد بـ low liquidity (prices noisy/wrong).
مع overlap = الـ pipeline يستبدل تلقائياً للعقد التاني عند نقطة الـ volume
crossover، وبيمسح الـ dying tail. ده بالظبط اللي بتعمله الـ exchange
official continuous contract methodology.

الاستخدام:
    # داتا كاملة بـ overlap (مفضّل):
    python self_supervised/combine_quarter_artifacts.py \\
        --quarter-dirs /workspace/clean/q1_6BH5 /workspace/clean/q2_6BM5 \\
        --output-dir /workspace/clean/combined_6m \\
        --overlap-mode auto

    # داتا متقطعة (آخر 15 يوم محذوفة):
    python self_supervised/combine_quarter_artifacts.py \\
        --quarter-dirs /workspace/clean/q1_6BH5 /workspace/clean/q2_6BM5 \\
        --output-dir /workspace/clean/combined_6m \\
        --overlap-mode carry \\
        --carry-pips-per-quarter 40

كل quarter-dir لازم يحتوي على:
    <bars-name>          (parquet — لازم يحتوي على volume column)
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


def _detect_roll_day_by_volume(
    old_bars: pd.DataFrame, new_bars: pd.DataFrame,
) -> tuple[pd.Timestamp | None, pd.DataFrame, pd.DataFrame]:
    """For overlapping quarters: find the day where new contract's daily
    volume first exceeds old contract's daily volume. Returns (roll_ts,
    overlap_old_daily_vol, overlap_new_daily_vol) for diagnostics.

    Returns (None, ...) if no overlap or no crossover found.
    """
    if 'volume' not in old_bars.columns or 'volume' not in new_bars.columns:
        return None, None, None

    old_day = old_bars['ts_event'].dt.floor('1D')
    new_day = new_bars['ts_event'].dt.floor('1D')

    overlap_start = max(old_day.min(), new_day.min())
    overlap_end = min(old_day.max(), new_day.max())
    if overlap_start > overlap_end:
        # No calendar overlap between quarters
        return None, None, None

    # Daily volume per contract within overlap
    old_in = old_bars.loc[(old_day >= overlap_start) & (old_day <= overlap_end)]
    new_in = new_bars.loc[(new_day >= overlap_start) & (new_day <= overlap_end)]
    if len(old_in) == 0 or len(new_in) == 0:
        return None, None, None

    v_old = old_in.groupby(old_in['ts_event'].dt.floor('1D'))['volume'].sum()
    v_new = new_in.groupby(new_in['ts_event'].dt.floor('1D'))['volume'].sum()

    # Find first day where new > old
    common_days = sorted(set(v_old.index) & set(v_new.index))
    if not common_days:
        return None, v_old, v_new
    for d in common_days:
        if float(v_new.get(d, 0)) > float(v_old.get(d, 0)):
            return d, v_old, v_new
    # Crossover never happened in overlap — use last common day
    return common_days[-1], v_old, v_new


def _compute_overlap_offset(
    old_bars: pd.DataFrame, new_bars: pd.DataFrame, roll_day: pd.Timestamp,
    window_hours: int = 24,
) -> float:
    """Compute REAL contract carry from overlap window around roll_day.
    Uses median mid prices in [roll_day - 24h, roll_day + 24h].
    """
    win = pd.Timedelta(hours=window_hours)
    lo, hi = roll_day - win, roll_day + win
    old_win = old_bars.loc[(old_bars['ts_event'] >= lo) & (old_bars['ts_event'] <= hi)]
    new_win = new_bars.loc[(new_bars['ts_event'] >= lo) & (new_bars['ts_event'] <= hi)]
    if len(old_win) == 0 or len(new_win) == 0:
        return float('nan')
    return float(_midprice(new_win).median() - _midprice(old_win).median())


def combine_artifacts(
    quarter_dirs: list[Path],
    output_dir: Path,
    names: dict,
    carry_pips_per_quarter: float = 40.0,
    auto_carry: bool = False,
    overlap_mode: str = 'auto',
) -> dict:
    """Combine bars + lob + orders across N quarters.

    overlap_mode:
      'auto'    — detect overlap; if present, use volume-crossover roll
                  + real offset + drop dying/thin wings.
                  If no overlap, fall back to carry-only.
      'overlap' — REQUIRE overlap; error if none.
      'carry'   — force carry-only regardless of overlap.
    """
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

    # ─── 3. Compute cumulative offsets + per-quarter keep-masks ─────────
    print(f"  ─── Computing boundary offsets (mode={overlap_mode}) ───")
    cumulative_offsets = [0.0] * len(quarters)
    boundary_info = []
    boundary_minutes_window = 60

    # keep_masks[i] = boolean mask over quarter i's bars (True = keep)
    # Default: keep everything (overwritten if overlap-based trimming runs)
    keep_masks = [np.ones(len(q['bars']), dtype=bool) for q in quarters]
    # roll_indices[i] = ts where Q[i] ends and Q[i+1] begins (after trim)
    roll_timestamps = [None] * (len(quarters) - 1)

    for i in range(len(quarters) - 2, -1, -1):
        old_bars = quarters[i]['bars']
        new_bars = quarters[i + 1]['bars']

        # Try overlap-based detection if requested
        roll_day = None
        v_old_d = v_new_d = None
        if overlap_mode in ('auto', 'overlap'):
            roll_day, v_old_d, v_new_d = _detect_roll_day_by_volume(old_bars, new_bars)

        used_method = None
        if roll_day is not None:
            # ─── OVERLAP MODE: trim dying/thin wings + real offset ───────
            print(f"    boundary {i}→{i+1}: OVERLAP detected")
            print(f"      common days: {len(set(v_old_d.index) & set(v_new_d.index))}")
            print(f"      roll day (volume crossover): {roll_day}")

            # Real offset from ±24h window where both contracts active
            real_offset = _compute_overlap_offset(
                old_bars, new_bars, roll_day, window_hours=24,
            )
            if np.isnan(real_offset):
                print(f"      ⚠️  offset window empty — falling back to carry-only")
                roll_day = None  # trigger carry-only branch below
            else:
                offset = real_offset
                offset_pips = offset / TICK_SIZE
                print(f"      real offset (from overlap): {offset_pips:+.1f} pips")

                # Trim:
                #   old: drop bars STRICTLY AFTER roll_day (dying contract)
                #   new: drop bars STRICTLY BEFORE roll_day (thin contract)
                # Roll_day itself goes to new (front-month perspective at roll)
                old_keep = old_bars['ts_event'] < roll_day
                new_keep = new_bars['ts_event'] >= roll_day
                n_old_dropped = (~old_keep).sum()
                n_new_dropped = (~new_keep).sum()
                print(f"      trimmed: Q{i+1} dying tail = {n_old_dropped:,} bars dropped")
                print(f"      trimmed: Q{i+2} thin head  = {n_new_dropped:,} bars dropped")

                keep_masks[i] &= old_keep.values
                keep_masks[i + 1] &= new_keep.values
                cumulative_offsets[i] = offset + cumulative_offsets[i + 1]
                roll_timestamps[i] = roll_day
                used_method = 'overlap-volume-crossover'

                boundary_info.append({
                    'old_dir': str(quarter_dirs[i]),
                    'new_dir': str(quarter_dirs[i + 1]),
                    'mode': used_method,
                    'roll_day': str(roll_day),
                    'real_offset_pips': offset_pips,
                    'applied_offset_pips': offset_pips,
                    'old_dying_dropped': int(n_old_dropped),
                    'new_thin_dropped': int(n_new_dropped),
                    'overlap_days': len(set(v_old_d.index) & set(v_new_d.index)),
                })

        if roll_day is None:
            # ─── FALLBACK: no overlap → carry-only ───────────────────────
            if overlap_mode == 'overlap':
                raise RuntimeError(
                    f"--overlap-mode=overlap requires overlap between Q{i+1} and Q{i+2}, "
                    f"but none was found. Use 'auto' or 'carry' instead."
                )
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
                used_method = 'auto-panama-FULL-GAP-UNSAFE'
            else:
                sign = 1.0 if raw_gap >= 0 else -1.0
                carry_pips = sign * carry_pips_per_quarter
                carry = carry_pips * TICK_SIZE
                used_method = f'carry-only({carry_pips_per_quarter:.0f}pips)'

            cumulative_offsets[i] = carry + cumulative_offsets[i + 1]
            roll_timestamps[i] = new_bars['ts_event'].iloc[0]

            print(f"    boundary {i}→{i+1}: NO overlap (gap={gap_days}d)")
            print(f"      raw_gap={raw_gap_pips:+.1f} pips (includes drift)")
            print(f"      applied: {used_method} → {carry_pips:+.1f} pips")

            boundary_info.append({
                'old_dir': str(quarter_dirs[i]),
                'new_dir': str(quarter_dirs[i + 1]),
                'mode': used_method,
                'gap_days': gap_days,
                'raw_gap_pips': raw_gap_pips,
                'applied_offset_pips': carry_pips,
            })

        print(f"      cum offset for Q{i+1}: {cumulative_offsets[i]/TICK_SIZE:+.1f} pips")
    print()

    # ─── 4. Apply keep_masks (trim wings) + offsets + is_roll mark ─────
    print(f"  ─── Applying keep-masks + offsets to bar prices ───")
    bar_pieces = []
    trimmed_npy = {key: [] for key in ('lob', 'order_features', 'order_masks')}

    for i, q in enumerate(quarters):
        mask = keep_masks[i]
        n_total = len(q['bars'])
        n_kept = int(mask.sum())
        if n_kept < n_total:
            print(f"    Q{i+1}: trimmed {n_total - n_kept:,} bars ({n_kept:,} kept)")
        bars = q['bars'].loc[mask].copy().reset_index(drop=True)

        # Apply trim to npy too (alignment preserved by construction)
        for key in trimmed_npy.keys():
            if q.get(key) is not None:
                trimmed_npy[key].append(q[key][mask])

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

    # Concat trimmed npy artifacts (already aligned to keep_masks)
    npy_combined = {}
    for key in ('lob', 'order_features', 'order_masks'):
        arrs = trimmed_npy[key]
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
        # Sanity: combined npy first-dim must equal combined_bars row count
        if combined.shape[0] != len(combined_bars):
            raise RuntimeError(
                f"{key} combined first dim ({combined.shape[0]}) != combined_bars "
                f"length ({len(combined_bars)}) — alignment broken!"
            )
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
        'overlap_mode': overlap_mode,
        'auto_carry': auto_carry,
        'carry_pips_per_quarter': carry_pips_per_quarter,
        'cumulative_offsets_pips': [o / TICK_SIZE for o in cumulative_offsets],
        'cumulative_offsets': [float(o) for o in cumulative_offsets],
        'boundaries': boundary_info,
        'n_bars_per_quarter_input': [len(q['bars']) for q in quarters],
        'n_bars_per_quarter_kept': [int(m.sum()) for m in keep_masks],
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
    overlap_used = any(b.get('mode') == 'overlap-volume-crossover' for b in boundary_info)
    if overlap_used:
        print(f"  ✅ Used overlap-based volume crossover: REAL contract offsets,")
        print(f"     dying-tail Q-N and thin-head Q-(N+1) bars trimmed automatically.")
        print(f"     Continuity gap at boundary should be tiny (true market move only).")
    else:
        print(f"  ⚠️  No overlap found — fell back to carry-only adjustment.")
        print(f"     Remaining gap at boundary is REAL market drift during the cut.")
    print(f"     is_session_break=1 + is_roll=1 markers prevent lookbacks crossing.")

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
                   help='Contract carry per quarter in pips (6B: ~30-60). '
                        'Used only when no overlap exists between quarters.')
    p.add_argument('--auto-carry', action='store_true',
                   help='UNSAFE: use full price gap as offset (includes market drift). '
                        'Only valid in carry mode (no overlap).')
    p.add_argument('--overlap-mode', default='auto',
                   choices=['auto', 'overlap', 'carry'],
                   help='auto: detect overlap, use volume crossover if present, '
                        'else fall back to carry-only. '
                        'overlap: REQUIRE overlap (error if none — best for FULL data). '
                        'carry: force carry-only regardless of overlap.')
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
        overlap_mode=args.overlap_mode,
    )


if __name__ == '__main__':
    main()
