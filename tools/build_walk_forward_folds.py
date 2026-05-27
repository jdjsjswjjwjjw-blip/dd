"""
tools/build_walk_forward_folds.py
═══════════════════════════════════════════════════════════════════════════════
Generate walk-forward fold definitions from a date range. Used by
scripts/walk_forward.sh so the fold list isn't hard-coded.

Default policy:
  • First fold trains on the first 12 months
  • Each subsequent fold expands the train window by 6 months
  • Test windows are 6 months
  • Output stops when the next 6-month test window would exceed end_ym

Usage:
  python tools/build_walk_forward_folds.py --start 2021-01 --end 2025-12
  → prints one fold per line: <fold_id> <tr_s> <tr_e> <te_s> <te_e>
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from dataclasses import dataclass


@dataclass
class Fold:
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def _ym(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _next_month(d: date, n: int = 1) -> date:
    """Add n months to a (year, month) date. Day is always day 1."""
    y, m = d.year, d.month + n
    while m > 12:
        m -= 12
        y += 1
    while m < 1:
        m += 12
        y -= 1
    return date(y, m, 1)


def _parse_ym(s: str) -> date:
    parts = s.split("-")
    return date(int(parts[0]), int(parts[1]), 1)


def build_folds(
    start_ym: str, end_ym: str,
    *,
    initial_train_months: int = 12,
    step_months: int = 6,
    test_window_months: int = 6,
) -> list[Fold]:
    start = _parse_ym(start_ym)
    end = _parse_ym(end_ym)

    folds: list[Fold] = []
    fold_id = 1
    # First fold: train_start..train_end = start .. start + initial_train_months - 1
    train_end = _next_month(start, initial_train_months - 1)
    while True:
        # Test window starts at month after train_end
        test_start = _next_month(train_end, 1)
        test_end = _next_month(test_start, test_window_months - 1)
        # Stop if test window exceeds end_ym
        if test_end > end:
            break
        folds.append(Fold(
            fold_id=fold_id,
            train_start=_ym(start),
            train_end=_ym(train_end),
            test_start=_ym(test_start),
            test_end=_ym(test_end),
        ))
        fold_id += 1
        # Expand train window
        train_end = _next_month(train_end, step_months)
    return folds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start YYYY-MM")
    p.add_argument("--end", required=True, help="End YYYY-MM (inclusive)")
    p.add_argument("--initial-train", type=int, default=12,
                   help="First-fold training window in months (default 12)")
    p.add_argument("--step", type=int, default=6,
                   help="Train-window expansion per fold in months (default 6)")
    p.add_argument("--test-window", type=int, default=6,
                   help="Test window in months (default 6)")
    p.add_argument("--format", choices=("bash", "json", "table"), default="bash",
                   help="Output format")
    args = p.parse_args()

    folds = build_folds(
        args.start, args.end,
        initial_train_months=args.initial_train,
        step_months=args.step,
        test_window_months=args.test_window,
    )

    if not folds:
        sys.stderr.write(
            f"❌ No folds could be built for {args.start}..{args.end} with "
            f"initial_train={args.initial_train} step={args.step} "
            f"test_window={args.test_window}\n"
        )
        sys.exit(1)

    if args.format == "bash":
        for f in folds:
            print(f"{f.fold_id} {f.train_start} {f.train_end} "
                  f"{f.test_start} {f.test_end}")
    elif args.format == "json":
        import json
        print(json.dumps([f.__dict__ for f in folds], indent=2))
    else:
        print(f"{'fold':>4} {'train_start':>11} {'train_end':>11} "
              f"{'test_start':>11} {'test_end':>11}")
        for f in folds:
            print(f"{f.fold_id:>4} {f.train_start:>11} {f.train_end:>11} "
                  f"{f.test_start:>11} {f.test_end:>11}")


if __name__ == "__main__":
    main()
