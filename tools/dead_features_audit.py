"""
tools/dead_features_audit.py — Sprint 9
─────────────────────────────────────────
يحلّ المشكلة #7 من التقرير (41% Features ميتة):
    "36 من 87 feature std=0 (41%)
     2 فقط std=0 بعد الإصلاحات
     ينقص 34 feature ميت في الإنتاج"

تشغيل:
    python tools/dead_features_audit.py --input path/to/features.parquet \\
                                        --output audit_report.json

يكتب JSON بـ:
    {
      "total_columns": N,
      "dead": {"count": K, "list": [...]},
      "weak": {"count": K, "list": [...]},
      "null_heavy": {"count": K, "list": [...]},
      "low_unique": {"count": K, "list": [...]},
      "recommendations": [...]
    }

الـ checks:
  - std == 0          → dead (لا معلومات)
  - std < 0.001       → weak (potentially dead)
  - null > 50%        → null-heavy
  - n_unique < 3      → low cardinality
  - constant fraction > 95% → near-constant
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def audit_features(parquet_path: str | Path, low_std: float = 1e-3, null_threshold: float = 0.5) -> dict[str, Any]:
    """يدير الـ feature audit ويُعيد dict structured."""
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if "ts_event" in df.columns:
        # تحويل timestamps إلى float لتجنب false positives
        df = df.drop(columns=["ts_event"])

    num = df.apply(pd.to_numeric, errors="coerce")
    std_series = num.std(numeric_only=True)
    null_frac = df.isnull().mean()

    dead: list[str] = []
    weak: list[str] = []
    null_heavy: list[str] = []
    low_unique: list[str] = []
    near_constant: list[str] = []

    for col in df.columns:
        # null check
        if col in null_frac.index and float(null_frac[col]) > null_threshold:
            null_heavy.append(col)
            continue
        # std check (numeric only)
        if col in std_series.index and pd.notna(std_series[col]):
            s = float(std_series[col])
            if s == 0.0:
                dead.append(col)
                continue
            if s < low_std:
                weak.append(col)
                continue
        # unique check
        try:
            n_uniq = int(df[col].nunique(dropna=True))
            if n_uniq < 3:
                low_unique.append(col)
        except Exception:
            pass
        # near-constant
        try:
            top_freq = float(df[col].value_counts(dropna=True, normalize=True).iloc[0])
            if top_freq > 0.95:
                near_constant.append(col)
        except Exception:
            pass

    total = len(df.columns)
    n_dead = len(dead)
    n_weak = len(weak)
    n_null = len(null_heavy)

    recommendations: list[str] = []
    if n_dead > 0:
        recommendations.append(
            f"إسقاط {n_dead} feature ميتة (std=0). راجع الـ data generation logic."
        )
    if n_weak > 0:
        recommendations.append(
            f"تحقّق من {n_weak} feature ضعيفة (std<{low_std}). قد تحتاج normalization."
        )
    if n_null > 0:
        recommendations.append(
            f"{n_null} feature بـ null > {null_threshold:.0%}. تحقّق من coverage."
        )
    dead_ratio = n_dead / max(total, 1)
    if dead_ratio > 0.10:
        recommendations.append(
            f"⚠️ نسبة الـ dead features ({dead_ratio:.0%}) أعلى من 10% — "
            f"احتمال مشكلة في الـ pipeline."
        )

    return {
        "input_path": str(parquet_path),
        "total_columns": int(total),
        "rows": int(len(df)),
        "dead": {"count": n_dead, "list": sorted(dead)},
        "weak": {"count": n_weak, "list": sorted(weak)},
        "null_heavy": {
            "count": n_null,
            "list": sorted(null_heavy),
            "threshold": null_threshold,
        },
        "low_unique": {"count": len(low_unique), "list": sorted(low_unique)},
        "near_constant": {"count": len(near_constant), "list": sorted(near_constant)},
        "summary": {
            "dead_pct": round(100.0 * n_dead / max(total, 1), 2),
            "weak_pct": round(100.0 * n_weak / max(total, 1), 2),
            "null_pct": round(100.0 * n_null / max(total, 1), 2),
        },
        "recommendations": recommendations,
    }


def format_report(audit: dict[str, Any]) -> str:
    """Pretty-print audit للـ console."""
    lines = ["═" * 70, "Dead Features Audit Report", "═" * 70]
    lines.append(f"Input:  {audit['input_path']}")
    lines.append(f"Rows:   {audit['rows']:,}")
    lines.append(f"Cols:   {audit['total_columns']:,}")
    lines.append("")
    s = audit["summary"]
    lines.append(f"Dead       (std=0):    {audit['dead']['count']:>4} ({s['dead_pct']:.1f}%)")
    lines.append(f"Weak       (std<1e-3): {audit['weak']['count']:>4} ({s['weak_pct']:.1f}%)")
    lines.append(f"Null-heavy (>50%):     {audit['null_heavy']['count']:>4} ({s['null_pct']:.1f}%)")
    lines.append(f"Low unique (<3):       {audit['low_unique']['count']:>4}")
    lines.append(f"Near-constant (>95%):  {audit['near_constant']['count']:>4}")
    if audit["recommendations"]:
        lines.append("")
        lines.append("Recommendations:")
        for r in audit["recommendations"]:
            lines.append(f"  • {r}")
    if audit["dead"]["count"] > 0:
        lines.append("")
        lines.append("Dead feature list:")
        for col in audit["dead"]["list"]:
            lines.append(f"  ✗ {col}")
    lines.append("═" * 70)
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Dead features audit")
    p.add_argument("--input", "-i", required=True, help="مسار parquet للـ audit")
    p.add_argument("--output", "-o", default=None, help="مسار JSON output (اختياري)")
    p.add_argument("--low-std", type=float, default=1e-3, help="threshold للـ weak std")
    p.add_argument("--null-threshold", type=float, default=0.5)
    p.add_argument("--quiet", action="store_true", help="suppress console output")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input غير موجود: {args.input}", file=sys.stderr)
        return 1

    audit = audit_features(
        args.input, low_std=args.low_std, null_threshold=args.null_threshold,
    )
    if not args.quiet:
        print(format_report(audit))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"\n💾 Audit JSON: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
