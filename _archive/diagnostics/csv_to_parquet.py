#!/usr/bin/env python3
"""
تحويل ملف/ملفات CSV إلى Parquet (.parquet بنفس الاسم بجانب الملف أو في --out-dir).
يتطلب: pandas + pyarrow (مثلاً: py -3 -m pip install pyarrow)

استخدام:
  py -3 csv_to_parquet.py "E:\\path\\mbo2.csv" "E:\\path\\mbp2.csv"
  py -3 csv_to_parquet.py *.csv --out-dir "E:\\out"
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert CSV file(s) to Parquet")
    ap.add_argument("inputs", nargs="+", help="Path(s) to .csv")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: same folder as each CSV)",
    )
    ap.add_argument(
        "--compression",
        default="snappy",
        help="Parquet compression: snappy, gzip, zstd, none (default: snappy)",
    )
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit(f"pandas missing: {e}") from e

    out_root = os.path.abspath(args.out_dir) if args.out_dir else None
    comp = None if str(args.compression).lower() in ("none", "") else args.compression

    for raw in args.inputs:
        src = os.path.abspath(os.path.expanduser(raw.strip()))
        if not os.path.isfile(src):
            print(f"skip (not found): {src}", file=sys.stderr)
            continue
        if not src.lower().endswith(".csv"):
            print(f"skip (not .csv): {src}", file=sys.stderr)
            continue

        base = os.path.splitext(os.path.basename(src))[0] + ".parquet"
        dest_dir = out_root or os.path.dirname(src)
        os.makedirs(dest_dir, exist_ok=True)
        dst = os.path.join(dest_dir, base)

        df = pd.read_csv(src, low_memory=False)
        try:
            df.to_parquet(dst, index=False, compression=comp)
        except ImportError as e:
            raise SystemExit(
                "to_parquet requires pyarrow or fastparquet.\n"
                "  py -3 -m pip install pyarrow\n"
                f"Original error: {e}"
            ) from e

        print(dst)


if __name__ == "__main__":
    main()
