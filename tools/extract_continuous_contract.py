"""
tools/extract_continuous_contract.py
═══════════════════════════════════════════════════════════════════════════════
Smart contract-rollover extractor — replaces the older "pick one contract per
quarter and drop the dying-tail" approach (now archived) with a clean,
continuous-price extractor.

Given a monthly Databento file containing BOTH the near-month and far-month
contracts (e.g., 2025-03.csv has 6BH5 + 6BM5), this script:

  1. Identifies all contracts present
  2. Computes per-day, per-contract volume
  3. Finds the rollover day = first day where far-month volume > near-month
  4. Outputs:
     • Near-month rows before rollover day
     • Far-month rows from rollover day onwards
     • Old-contract prices SHIFTED so they match the new contract's price
       level at the rollover (Panama back-adjustment, preserves most recent
       prices as-is)
  5. Writes a rollover_log.json with the detected switch and the offset

Algorithm (causal, no lookahead):
  • Volume comparison uses ONLY days from THIS month's data — never peeks
    into next month
  • Offset window for back-adjustment is the [rollover_day - 3, rollover_day - 1]
    day range — strictly before the rollover
  • If no rollover is detected (single contract dominates the whole month),
    that contract is used unchanged

Usage:
  # Basic — auto-detect rollover within the month
  python tools/extract_continuous_contract.py 2025-03.csv \\
      --root 6B --output 2025-03_continuous.parquet

  # Override rollover day (manual)
  python tools/extract_continuous_contract.py 2025-03.csv \\
      --root 6B --rollover-day 2025-03-12 --output 2025-03_continuous.parquet

  # Disable back-adjustment (raw output)
  python tools/extract_continuous_contract.py 2025-03.csv \\
      --root 6B --no-adjust --output 2025-03_continuous.parquet

The output preserves the input schema entirely (e.g., MBP-10 bid_px_00..09
+ ask_px_00..09 + sizes; MBO order_id + action + price + side). Only price
columns get adjusted when back-adjustment is enabled.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# CME contract month codes
CONTRACT_MONTH_CODES = "FGHJKMNQUVXZ"
# Quarterly codes (FX futures)
QUARTERLY_CODES = set("HMUZ")

# Default price columns to adjust (covers common Databento schemas)
DEFAULT_PRICE_COLS = (
    "price",                  # trade price (MBO, trades)
    "open", "high", "low", "close",
    "vwap",
    # MBP-10
    *[f"bid_px_{i:02d}" for i in range(10)],
    *[f"ask_px_{i:02d}" for i in range(10)],
)

CONTRACT_RE = re.compile(
    r"^(?P<root>.+?)(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{1,4})$",
    re.IGNORECASE,
)


def _parse_contract(symbol: str) -> tuple[str, str, int] | None:
    """Parse a Databento contract symbol like '6BH5' or '6BH25' into
    (root, month_code, year). Returns None on failure."""
    if not isinstance(symbol, str):
        return None
    s = symbol.strip().upper()
    m = CONTRACT_RE.match(s)
    if not m:
        return None
    root = m.group("root")
    month_code = m.group("month")
    year_raw = m.group("year")
    # Single digit year → 2020s; 2-digit → 20XX
    if len(year_raw) == 1:
        year = 2020 + int(year_raw)
    elif len(year_raw) == 2:
        year = 2000 + int(year_raw)
    else:
        year = int(year_raw)
    return root, month_code, year


def _contract_expiry_key(symbol: str) -> tuple[int, int]:
    """Sort key: (year, month_index). Earlier expiry sorts first."""
    parsed = _parse_contract(symbol)
    if parsed is None:
        return (9999, 99)
    _, code, year = parsed
    month_idx = CONTRACT_MONTH_CODES.find(code)
    if month_idx < 0:
        month_idx = 99
    return (year, month_idx)


def _read_input(path: Path) -> pd.DataFrame:
    """Read CSV or parquet. Detects from extension."""
    p = str(path).lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv") or p.endswith(".csv.gz") or p.endswith(".csv.zst"):
        return pd.read_csv(path, low_memory=False, compression="infer")
    raise ValueError(f"Unsupported file extension: {path}")


def _write_output(df: pd.DataFrame, path: Path) -> None:
    p = str(path).lower()
    if p.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif p.endswith(".csv"):
        df.to_csv(path, index=False)
    elif p.endswith(".csv.gz"):
        df.to_csv(path, index=False, compression="gzip")
    else:
        raise ValueError(f"Unsupported output extension: {path}")


def _resolve_columns(df: pd.DataFrame) -> tuple[str, str, list[str]]:
    """Identify the timestamp, symbol, and price columns actually present."""
    ts_col = next((c for c in ("ts_event", "ts_recv", "timestamp", "date")
                   if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"No timestamp column found. Tried: ts_event/ts_recv/timestamp/date")
    sym_col = next((c for c in ("symbol", "raw_symbol")
                    if c in df.columns), None)
    if sym_col is None:
        raise ValueError("No symbol column found. Tried: symbol/raw_symbol")
    price_cols = [c for c in DEFAULT_PRICE_COLS if c in df.columns]
    return ts_col, sym_col, price_cols


def detect_rollover_day(
    df: pd.DataFrame, ts_col: str, sym_col: str,
    near_symbol: str, far_symbol: str,
    size_col: str | None = "size",
) -> pd.Timestamp | None:
    """Find the first day where far-month daily volume strictly exceeds
    near-month. Returns None if no such crossover happens in the data.

    Uses `size_col` if present; otherwise falls back to row-count per
    contract per day (less precise but still ordinal).
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col])
    df["_day"] = df[ts_col].dt.floor("1D")

    use_size = size_col is not None and size_col in df.columns
    if use_size:
        df["_sz"] = pd.to_numeric(df[size_col], errors="coerce").fillna(0)
        daily = df.groupby([sym_col, "_day"])["_sz"].sum().unstack(0, fill_value=0)
    else:
        daily = df.groupby([sym_col, "_day"]).size().unstack(0, fill_value=0)

    if near_symbol not in daily.columns or far_symbol not in daily.columns:
        return None

    near = daily[near_symbol]
    far = daily[far_symbol]
    # First day where far > near
    cross = (far > near) & (far > 0)
    if not cross.any():
        return None
    return cross.idxmax()  # earliest True


def compute_offset_for_rollover(
    df: pd.DataFrame, ts_col: str, sym_col: str,
    near_symbol: str, far_symbol: str,
    rollover_day: pd.Timestamp,
    *,
    window_days: int = 3,
    price_col: str = "price",
) -> float:
    """Median(near) − median(far) in the [rollover_day - window_days,
    rollover_day - 1] window. Use this to SHIFT near-month prices DOWN
    so they line up with far-month level (Panama back-adjustment).

    Causal: window stops at rollover_day - 1.
    """
    if price_col not in df.columns:
        # Fall back to bid+ask midpoint if available
        if "bid_px_00" in df.columns and "ask_px_00" in df.columns:
            mid = (pd.to_numeric(df["bid_px_00"], errors="coerce")
                   + pd.to_numeric(df["ask_px_00"], errors="coerce")) / 2
            df = df.assign(_mid=mid)
            price_col = "_mid"
        else:
            raise ValueError(
                f"price column '{price_col}' not in dataframe and no bid/ask "
                f"fallback available"
            )

    win_start = rollover_day - pd.Timedelta(days=window_days)
    win_end = rollover_day - pd.Timedelta(seconds=1)  # strictly before rollover

    day = pd.to_datetime(df[ts_col], utc=True).dt.floor("1D")
    in_win = (day >= win_start) & (day <= win_end)
    near_prices = pd.to_numeric(
        df.loc[in_win & (df[sym_col] == near_symbol), price_col],
        errors="coerce",
    ).dropna()
    far_prices = pd.to_numeric(
        df.loc[in_win & (df[sym_col] == far_symbol), price_col],
        errors="coerce",
    ).dropna()
    if len(near_prices) == 0 or len(far_prices) == 0:
        return float("nan")
    return float(near_prices.median() - far_prices.median())


def extract_continuous(
    df: pd.DataFrame,
    *,
    root: str | None = None,
    rollover_day: pd.Timestamp | None = None,
    adjust_prices: bool = True,
    price_cols: list[str] | None = None,
    window_days: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Core algorithm. Returns (continuous_df, log).

    log contains the detected near/far symbols, the rollover day, the
    raw and applied offsets, and per-contract row counts.
    """
    ts_col, sym_col, default_price_cols = _resolve_columns(df)
    if price_cols is None:
        price_cols = default_price_cols

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    # ── Identify contracts ──
    all_symbols = df[sym_col].dropna().astype(str).unique().tolist()
    if root:
        candidates = [s for s in all_symbols
                      if (_parse_contract(s) or ("", "", 0))[0].upper()
                      == root.upper()]
    else:
        candidates = list(all_symbols)
    # Filter to ones we can parse
    candidates = [s for s in candidates if _parse_contract(s) is not None]
    candidates.sort(key=_contract_expiry_key)

    log: dict = {
        "input_rows": int(len(df)),
        "symbols_found": all_symbols,
        "root_filter": root,
        "candidates_after_filter": candidates,
    }

    if len(candidates) == 0:
        raise ValueError(
            f"No parseable contracts found. all_symbols={all_symbols[:5]}, "
            f"root_filter={root}"
        )
    if len(candidates) == 1:
        # Single contract — just filter and output as-is
        out = df[df[sym_col] == candidates[0]].copy()
        log.update(
            mode="single_contract",
            chosen=candidates[0],
            rollover_day=None,
            offset=0.0,
            output_rows=int(len(out)),
        )
        return out, log

    near, far = candidates[0], candidates[1]
    log["near"] = near
    log["far"] = far

    # ── Detect rollover ──
    if rollover_day is None:
        rollover_day = detect_rollover_day(df, ts_col, sym_col, near, far)
    if rollover_day is None:
        # No crossover — pick the contract with the most total volume
        if "size" in df.columns:
            sizes = df.groupby(sym_col)["size"].sum()
        else:
            sizes = df.groupby(sym_col).size()
        chosen = sizes.idxmax() if not sizes.empty else near
        out = df[df[sym_col] == chosen].copy()
        log.update(
            mode="no_rollover_picked_dominant",
            chosen=str(chosen),
            rollover_day=None,
            offset=0.0,
            output_rows=int(len(out)),
        )
        return out, log

    rollover_day = pd.Timestamp(rollover_day).tz_localize("UTC") \
        if pd.Timestamp(rollover_day).tz is None else pd.Timestamp(rollover_day)
    log["rollover_day"] = str(rollover_day)

    # ── Compute offset ──
    raw_offset = compute_offset_for_rollover(
        df, ts_col, sym_col, near, far, rollover_day,
        window_days=window_days,
    )
    log["raw_offset"] = raw_offset
    if not adjust_prices or not np.isfinite(raw_offset):
        applied_offset = 0.0
        log["adjust_prices"] = False
    else:
        applied_offset = float(raw_offset)
        log["adjust_prices"] = True
    log["applied_offset"] = applied_offset

    # ── Build continuous stream ──
    day_floor = df[ts_col].dt.floor("1D")
    is_near_day = day_floor < rollover_day
    is_far_day = day_floor >= rollover_day

    near_rows = df[(df[sym_col] == near) & is_near_day].copy()
    far_rows = df[(df[sym_col] == far) & is_far_day].copy()

    # ── Apply back-adjustment to NEAR rows (shift down by offset so prices
    #    line up with far at the rollover) ──
    if applied_offset != 0.0 and len(near_rows) > 0:
        for c in price_cols:
            if c in near_rows.columns:
                col_vals = pd.to_numeric(near_rows[c], errors="coerce")
                near_rows[c] = (col_vals - applied_offset).where(
                    col_vals.notna() & (col_vals != 0), col_vals,
                )

    out = pd.concat([near_rows, far_rows], ignore_index=True)
    out = out.sort_values(ts_col).reset_index(drop=True)

    log.update(
        mode="continuous_with_rollover",
        near_rows=int(len(near_rows)),
        far_rows=int(len(far_rows)),
        output_rows=int(len(out)),
        price_cols_adjusted=[c for c in price_cols if c in out.columns]
                              if applied_offset != 0.0 else [],
    )
    return out, log


def main() -> int:
    p = argparse.ArgumentParser(
        description="Extract continuous contract with smart volume-based rollover."
    )
    p.add_argument("input", help="Input file (CSV / CSV.GZ / parquet)")
    p.add_argument("--output", required=True, help="Output file (CSV / parquet)")
    p.add_argument("--root", default=None,
                   help="Root symbol filter (e.g., '6B'). Default: keep all contracts.")
    p.add_argument("--rollover-day", default=None,
                   help="Override the detected rollover day (ISO date, e.g., 2025-03-12).")
    p.add_argument("--no-adjust", action="store_true",
                   help="Disable price back-adjustment (raw concatenated output).")
    p.add_argument("--window-days", type=int, default=3,
                   help="Days before rollover used to compute the price offset.")
    p.add_argument("--log", default=None,
                   help="Path for rollover_log.json. Default: <output>.log.json")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    log_path = Path(args.log) if args.log else out_path.with_suffix(
        out_path.suffix + ".log.json"
    )

    print(f"Reading: {in_path}")
    df = _read_input(in_path)
    print(f"  rows: {len(df):,}  cols: {df.shape[1]}")

    rollover = None
    if args.rollover_day:
        rollover = pd.Timestamp(args.rollover_day, tz="UTC")

    out, log = extract_continuous(
        df,
        root=args.root,
        rollover_day=rollover,
        adjust_prices=not args.no_adjust,
        window_days=args.window_days,
    )

    print()
    print("Result:")
    print(f"  mode:            {log.get('mode')}")
    print(f"  near:            {log.get('near')}")
    print(f"  far:             {log.get('far')}")
    print(f"  rollover_day:    {log.get('rollover_day')}")
    print(f"  raw_offset:      {log.get('raw_offset')}")
    print(f"  applied_offset:  {log.get('applied_offset')}")
    print(f"  output_rows:     {log.get('output_rows'):,}"
          if isinstance(log.get('output_rows'), int) else "")
    print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_output(out, out_path)
    print(f"💾 {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"💾 {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
