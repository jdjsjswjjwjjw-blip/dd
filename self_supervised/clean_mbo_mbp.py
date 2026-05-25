"""
self_supervised/clean_mbo_mbp.py
═══════════════════════════════════════════════════════════
Comprehensive cleaner for MBO and MBP parquet files.

Handles 11 categories of data corruption:

  1. NaN/Inf in critical columns (ts_event, price, action)
  2. Out-of-band prices (instrument-specific bounds)
  3. Non-positive sizes (except for cancels)
  4. Invalid action codes (outside known vocabulary)
  5. Invalid side codes
  6. Duplicate composite keys
  7. Non-monotonic ts_event
  8. Mixed symbols (when single-instrument expected)
  9. Crossed books (bid > ask) — MBP only
  10. Wrong-ordered book levels — MBP only
  11. Time discontinuities (flagged not dropped)

Output:
  • Cleaned parquet at --output
  • Sidecar .audit.json with per-step removal counts + before/after stats
  • Exit code 0 on success, 1 if too much data removed (>5% by default)

Usage:
  python self_supervised/clean_mbo_mbp.py \\
      --input  glbx-mdp3-20250401-20250620.mbo_6BM5.parquet \\
      --output glbx-mdp3-20250401-20250620.mbo_6BM5_clean.parquet \\
      --kind mbo \\
      --price-min 0.5 --price-max 5.0

  python self_supervised/clean_mbo_mbp.py \\
      --input  glbx-mdp3-20250401-20250620.mbp.parquet \\
      --output glbx-mdp3-20250401-20250620.mbp_clean.parquet \\
      --kind mbp \\
      --price-min 0.5 --price-max 5.0
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Known vocabulary (Databento + common feed conventions)
KNOWN_ACTIONS = {'T', 'A', 'C', 'M', 'F', 'R', 'B', 'N', 'X'}
KNOWN_SIDES   = {'A', 'B', 'N', ''}
CANCEL_LIKE_ACTIONS = {'C', 'R', 'X'}   # zero/negative size is OK for these


# ════════════════════════════════════════════════════════════════════════════
# Audit
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CleanAudit:
    """Tracks per-step removal counts and before/after stats."""
    input_path: str
    output_path: str
    kind: str
    n_input: int
    removed_per_step: dict[str, int] = field(default_factory=dict)
    flags_added: dict[str, int] = field(default_factory=dict)
    n_output: int = 0
    pre_stats: dict = field(default_factory=dict)
    post_stats: dict = field(default_factory=dict)

    def add_removal(self, step: str, n: int):
        if n > 0:
            self.removed_per_step[step] = n
            print(f"  − {step:35s}: removed {n:>8,}")

    def add_flag(self, step: str, n: int):
        self.flags_added[step] = n
        print(f"  ⚑ {step:35s}: flagged {n:>8,}")

    def to_dict(self) -> dict:
        return {
            'input': self.input_path,
            'output': self.output_path,
            'kind': self.kind,
            'n_input': self.n_input,
            'n_output': self.n_output,
            'n_removed_total': self.n_input - self.n_output,
            'removed_pct': (self.n_input - self.n_output) / max(self.n_input, 1) * 100,
            'removed_per_step': self.removed_per_step,
            'flags_added': self.flags_added,
            'pre_stats': self.pre_stats,
            'post_stats': self.post_stats,
        }


def _basic_stats(df: pd.DataFrame) -> dict:
    """Compact stats snapshot for audit."""
    out: dict = {'rows': int(len(df))}
    if 'price' in df.columns:
        p = pd.to_numeric(df['price'], errors='coerce')
        out['price'] = {
            'min': float(p.min()) if len(p) else None,
            'max': float(p.max()) if len(p) else None,
            'median': float(p.median()) if len(p) else None,
            'nan': int(p.isna().sum()),
        }
    if 'ts_event' in df.columns:
        ts = pd.to_datetime(df['ts_event'], errors='coerce')
        out['ts_event'] = {
            'first': str(ts.min()), 'last': str(ts.max()),
            'is_monotonic': bool(ts.is_monotonic_increasing) if len(ts) else None,
        }
    if 'size' in df.columns:
        s = pd.to_numeric(df['size'], errors='coerce')
        out['size'] = {
            'min': float(s.min()) if len(s) else None,
            'max': float(s.max()) if len(s) else None,
            'median': float(s.median()) if len(s) else None,
        }
    if 'action' in df.columns:
        out['action_dist'] = (
            df['action'].astype(str).str.upper().value_counts().head(10).to_dict()
        )
    return out


# ════════════════════════════════════════════════════════════════════════════
# Cleaning steps (shared)
# ════════════════════════════════════════════════════════════════════════════

def _drop_nan_critical(df: pd.DataFrame, audit: CleanAudit,
                       critical_cols: list[str]) -> pd.DataFrame:
    """Drop rows with NaN/Inf in any critical column."""
    n_before = len(df)
    for c in critical_cols:
        if c not in df.columns:
            continue
        if c == 'ts_event':
            df = df[pd.to_datetime(df[c], errors='coerce').notna()]
        elif pd.api.types.is_numeric_dtype(df[c]):
            df = df[np.isfinite(df[c])]
        else:
            df = df[df[c].notna()]
    audit.add_removal('NaN/Inf in critical cols', n_before - len(df))
    return df


def _filter_price_range(df: pd.DataFrame, audit: CleanAudit,
                        price_min: float, price_max: float,
                        price_col: str = 'price') -> pd.DataFrame:
    if price_col not in df.columns:
        return df
    n_before = len(df)
    p = pd.to_numeric(df[price_col], errors='coerce')
    df = df[p.between(price_min, price_max, inclusive='both')]
    audit.add_removal(f'price not in [{price_min}, {price_max}]', n_before - len(df))
    return df


def _drop_invalid_actions(df: pd.DataFrame, audit: CleanAudit) -> pd.DataFrame:
    if 'action' not in df.columns:
        return df
    n_before = len(df)
    act = df['action'].astype(str).str.upper().str.strip()
    df = df[act.isin(KNOWN_ACTIONS)]
    audit.add_removal('action not in vocabulary', n_before - len(df))
    return df


def _drop_invalid_sizes(df: pd.DataFrame, audit: CleanAudit) -> pd.DataFrame:
    """Drop non-positive sizes — except for cancel-like actions where 0 is OK."""
    if 'size' not in df.columns:
        return df
    n_before = len(df)
    sz = pd.to_numeric(df['size'], errors='coerce')
    act = df.get('action', pd.Series('', index=df.index)).astype(str).str.upper()
    is_cancel = act.isin(CANCEL_LIKE_ACTIONS)
    # Keep: positive size, OR cancel-like with size >= 0
    keep = (sz > 0) | (is_cancel & (sz >= 0))
    df = df[keep.fillna(False)]
    audit.add_removal('non-positive size (non-cancel)', n_before - len(df))
    return df


def _drop_invalid_sides(df: pd.DataFrame, audit: CleanAudit) -> pd.DataFrame:
    if 'side' not in df.columns:
        return df
    n_before = len(df)
    side = df['side'].astype(str).str.upper().str.strip()
    df = df[side.isin(KNOWN_SIDES)]
    audit.add_removal('side not in vocabulary', n_before - len(df))
    return df


def _ensure_sorted_ts(df: pd.DataFrame, audit: CleanAudit) -> pd.DataFrame:
    """Sort by ts_event (stable). Audit but don't drop."""
    if 'ts_event' not in df.columns:
        return df
    ts = pd.to_datetime(df['ts_event'])
    if not ts.is_monotonic_increasing:
        print(f"  ↻ ts_event not monotonic; sorting stably")
        df = df.assign(ts_event=ts).sort_values('ts_event', kind='mergesort').reset_index(drop=True)
    return df


def _dedupe_composite_keys(df: pd.DataFrame, audit: CleanAudit,
                           keys: list[str]) -> pd.DataFrame:
    keys_present = [k for k in keys if k in df.columns]
    if len(keys_present) < 2:
        return df
    n_before = len(df)
    df = df.drop_duplicates(subset=keys_present, keep='first')
    audit.add_removal(f'duplicate ({"+".join(keys_present)})', n_before - len(df))
    return df


def _validate_single_symbol(df: pd.DataFrame, audit: CleanAudit) -> pd.DataFrame:
    for sym_col in ('symbol', 'instrument_id', 'raw_symbol'):
        if sym_col in df.columns:
            n_unique = df[sym_col].nunique()
            if n_unique > 1:
                most_common = df[sym_col].value_counts().idxmax()
                n_before = len(df)
                df = df[df[sym_col] == most_common]
                audit.add_removal(
                    f'symbols other than {most_common!r} ({sym_col})',
                    n_before - len(df),
                )
            break
    return df


def _flag_feed_gaps(df: pd.DataFrame, audit: CleanAudit,
                    gap_threshold_sec: float) -> pd.DataFrame:
    """Add is_feed_gap = 1 for the first tick after a > threshold gap."""
    if 'ts_event' not in df.columns or len(df) < 2:
        return df
    ts = pd.to_datetime(df['ts_event'])
    dt = ts.diff().dt.total_seconds()
    is_gap = (dt > gap_threshold_sec).fillna(False).astype(np.int8)
    df = df.copy()
    df['is_feed_gap'] = is_gap.to_numpy()
    audit.add_flag(f'feed gap > {gap_threshold_sec}s', int(is_gap.sum()))
    return df


# ════════════════════════════════════════════════════════════════════════════
# MBP-specific
# ════════════════════════════════════════════════════════════════════════════

def _flag_crossed_books(df: pd.DataFrame, audit: CleanAudit,
                        drop_above_ticks: float = 100.0,
                        tick_size: float = 0.0001) -> pd.DataFrame:
    """Flag crossed books (bid > ask) as is_crossed_book. Drop ONLY
    massively-crossed snapshots (likely feed corruption).

    Mild crossing is common during fast moves (stale snapshots) — we keep
    those and let downstream code decide.
    """
    if 'bid_px_00' not in df.columns or 'ask_px_00' not in df.columns:
        return df
    df = df.copy()
    bid = pd.to_numeric(df['bid_px_00'], errors='coerce')
    ask = pd.to_numeric(df['ask_px_00'], errors='coerce')
    cross_ticks = ((bid - ask) / tick_size).where(bid.notna() & ask.notna(), 0.0)
    is_crossed = (cross_ticks > 0).astype(np.int8)
    df['is_crossed_book'] = is_crossed.to_numpy()
    audit.add_flag('crossed book (bid > ask)', int(is_crossed.sum()))

    # Drop only if cross is huge (feed glitch territory)
    n_before = len(df)
    keep = (cross_ticks <= drop_above_ticks).fillna(True)
    df = df[keep]
    audit.add_removal(
        f'severely-crossed book (>{drop_above_ticks} ticks)',
        n_before - len(df),
    )
    return df


def _flag_wrong_ordered_levels(df: pd.DataFrame, audit: CleanAudit,
                               max_depth: int = 10) -> pd.DataFrame:
    """Flag rows with disordered levels (bid not descending, ask not ascending)
    as is_level_disordered. Don't drop — these often appear in transient
    snapshots between book updates.
    """
    df = df.copy()
    bad = pd.Series(False, index=df.index)

    bid_cols = [f'bid_px_{i:02d}' for i in range(max_depth) if f'bid_px_{i:02d}' in df.columns]
    if len(bid_cols) >= 2:
        bid_arr = df[bid_cols].apply(pd.to_numeric, errors='coerce').to_numpy()
        for i in range(len(bid_cols) - 1):
            bad |= pd.Series(
                (bid_arr[:, i] < bid_arr[:, i + 1])
                & np.isfinite(bid_arr[:, i]) & np.isfinite(bid_arr[:, i + 1]),
                index=df.index,
            )

    ask_cols = [f'ask_px_{i:02d}' for i in range(max_depth) if f'ask_px_{i:02d}' in df.columns]
    if len(ask_cols) >= 2:
        ask_arr = df[ask_cols].apply(pd.to_numeric, errors='coerce').to_numpy()
        for i in range(len(ask_cols) - 1):
            bad |= pd.Series(
                (ask_arr[:, i] > ask_arr[:, i + 1])
                & np.isfinite(ask_arr[:, i]) & np.isfinite(ask_arr[:, i + 1]),
                index=df.index,
            )

    df['is_level_disordered'] = bad.astype(np.int8).to_numpy()
    audit.add_flag('book levels disordered', int(bad.sum()))
    return df


# ════════════════════════════════════════════════════════════════════════════
# Top-level cleaners
# ════════════════════════════════════════════════════════════════════════════

def clean_mbo(df: pd.DataFrame, audit: CleanAudit,
              price_min: float, price_max: float,
              gap_threshold_sec: float) -> pd.DataFrame:
    """MBO cleaning pipeline."""
    df = _drop_nan_critical(df, audit, ['ts_event', 'price', 'action'])
    df = _filter_price_range(df, audit, price_min, price_max)
    df = _drop_invalid_actions(df, audit)
    df = _drop_invalid_sides(df, audit)
    df = _drop_invalid_sizes(df, audit)
    df = _validate_single_symbol(df, audit)
    df = _ensure_sorted_ts(df, audit)
    df = _dedupe_composite_keys(df, audit, ['ts_event', 'order_id', 'action'])
    df = _flag_feed_gaps(df, audit, gap_threshold_sec)
    return df


def clean_mbp(df: pd.DataFrame, audit: CleanAudit,
              price_min: float, price_max: float,
              gap_threshold_sec: float) -> pd.DataFrame:
    """MBP cleaning pipeline. Validates book consistency."""
    df = _drop_nan_critical(df, audit, ['ts_event'])
    # MBP price (if it has a top-level 'price' col, filter; else skip)
    if 'price' in df.columns:
        df = _filter_price_range(df, audit, price_min, price_max)
    # Also filter the best bid/ask if present
    if 'bid_px_00' in df.columns:
        df = _filter_price_range(df, audit, price_min, price_max, 'bid_px_00')
    if 'ask_px_00' in df.columns:
        df = _filter_price_range(df, audit, price_min, price_max, 'ask_px_00')
    df = _drop_invalid_actions(df, audit)
    df = _drop_invalid_sides(df, audit)
    df = _flag_crossed_books(df, audit)
    df = _flag_wrong_ordered_levels(df, audit)
    df = _validate_single_symbol(df, audit)
    df = _ensure_sorted_ts(df, audit)
    df = _dedupe_composite_keys(df, audit, ['ts_event', 'price', 'side', 'action'])
    df = _flag_feed_gaps(df, audit, gap_threshold_sec)
    return df


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--kind', required=True, choices=['mbo', 'mbp'])
    p.add_argument('--price-min', type=float, default=0.5)
    p.add_argument('--price-max', type=float, default=5.0)
    p.add_argument('--gap-threshold-sec', type=float, default=60.0,
                  help='Flag is_feed_gap=1 for gaps > this (default 60s)')
    p.add_argument('--max-removed-pct', type=float, default=5.0,
                  help='Fail if more than this %% of rows removed (default 5%%)')
    args = p.parse_args()

    print(f"═══ Cleaning {args.kind.upper()} ═══")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Price band: [{args.price_min}, {args.price_max}]")
    print()

    print(f"📂 Loading…")
    df = pd.read_parquet(args.input)
    print(f"   {len(df):,} rows × {len(df.columns)} cols")
    print()

    audit = CleanAudit(
        input_path=args.input, output_path=args.output, kind=args.kind,
        n_input=len(df),
    )
    audit.pre_stats = _basic_stats(df)

    print(f"🧹 Cleaning…")
    if args.kind == 'mbo':
        df = clean_mbo(df, audit, args.price_min, args.price_max, args.gap_threshold_sec)
    else:
        df = clean_mbp(df, audit, args.price_min, args.price_max, args.gap_threshold_sec)

    audit.n_output = len(df)
    audit.post_stats = _basic_stats(df)

    removed_pct = (audit.n_input - audit.n_output) / max(audit.n_input, 1) * 100
    print()
    print(f"📊 Summary:")
    print(f"   Before: {audit.n_input:>12,} rows")
    print(f"   After:  {audit.n_output:>12,} rows")
    print(f"   Removed: {audit.n_input - audit.n_output:>11,} ({removed_pct:.3f}%)")
    print()

    # Hard-fail if too aggressive (indicates wrong thresholds or feed schema mismatch)
    if removed_pct > args.max_removed_pct:
        print(f"❌ FATAL: removed {removed_pct:.2f}% of rows (limit {args.max_removed_pct}%)")
        print(f"   Likely wrong --price-min/--price-max or feed schema mismatch.")
        print(f"   Inspect audit JSON to identify which step over-removed.")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(Path(args.output).with_suffix('.audit.json'), 'w') as f:
            json.dump(audit.to_dict(), f, indent=2, default=str)
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Writing…")
    df.to_parquet(args.output, index=False)
    out_size_gb = Path(args.output).stat().st_size / 1e9
    print(f"   {len(df):,} rows × {len(df.columns)} cols → {args.output} ({out_size_gb:.2f} GB)")

    audit_path = Path(args.output).with_suffix('.audit.json')
    with open(audit_path, 'w') as f:
        json.dump(audit.to_dict(), f, indent=2, default=str)
    print(f"📋 Audit: {audit_path}")


if __name__ == '__main__':
    main()
