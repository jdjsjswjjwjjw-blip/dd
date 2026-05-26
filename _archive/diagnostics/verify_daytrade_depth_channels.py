#!/usr/bin/env python3
"""
verify_daytrade_depth_channels.py — تحقق شامل من قنوات عمق السوق في parquet الـ DayTrade

يفحص:
  • وجود الأعمدة الأساسية ونسب NaN/صفر
  • هل القيم متغيرة (ليست ثابتة بالكامل)
  • هل OBI و lob_imbalance مستقلان أم منسوخان تقريبًا (ارتباط / تطابق)
  • علاقة bid_wall + ask_wall مع buy_ratio (بروكسي prepare_day_trading)
  • مؤشرات مساعدة إن وُجدت: order_flow_imbalance، أعمدة MBP

استخدام:
  py -3 verify_daytrade_depth_channels.py --parquet path/to/day_trading_features.parquet
  py -3 verify_daytrade_depth_channels.py --parquet ... --strict   # أي تحذير = خروج 1

خروج: 0 إذا لا توجد FAIL؛ 1 إذا FAIL؛ مع --strict أيضًا 1 عند WARN.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd


DEPTH_CORE = ("obi", "lob_imbalance", "bid_wall_strength", "ask_wall_strength")
OPTIONAL_BRIDGE = (
    "buy_ratio",
    "order_flow_imbalance",
    "mbp_bar_coverage",
    "mbp_roll_lob_coverage",
    "bar_cvd_delta",
)


def _reject_placeholder(path: str, label: str) -> None:
    s = str(path).strip()
    if "..." in s:
        raise SystemExit(
            f"[FAIL] {label}: مسار غير صالح (يحتوي على ...). الصق المسار الكامل للملف.\n   {path}"
        )


def _finite_mask(x: pd.Series) -> np.ndarray:
    v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    return np.isfinite(v)


def _stats_series(x: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    n = len(v)
    finite = np.isfinite(v)
    nf = int(finite.sum())
    arr = v[finite]
    out: dict[str, Any] = {
        "n": n,
        "null_pct": float(np.mean(~finite)) if n else 1.0,
        "finite_n": nf,
        "zero_pct": float(np.mean(np.abs(arr) < 1e-12)) if nf else 1.0,
        "std": float(np.nanstd(arr)) if nf > 1 else 0.0,
        "min": float(np.nanmin(arr)) if nf else None,
        "max": float(np.nanmax(arr)) if nf else None,
    }
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return None
    xa = a[m]
    xb = b[m]
    if float(np.std(xa)) < 1e-15 or float(np.std(xb)) < 1e-15:
        return None
    return float(np.corrcoef(xa, xb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify DayTrade depth channels (OBI / LOB / walls)")
    ap.add_argument("--parquet", required=True, help="day_trading_features*.parquet")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="كلمة سر خروج 1 عند أي WARN (لـ CI)",
    )
    ap.add_argument(
        "--obi-lob-corr-warn",
        type=float,
        default=0.999,
        help="تحذير إذا |corr(obi, lob_imbalance)| ≥ هذا",
    )
    ap.add_argument(
        "--obi-lob-identical-warn-pct",
        type=float,
        default=99.0,
        help="تحذير إذا نسبة الشموع حيث |obi−lob|<1e-9 تتجاوز هذا",
    )
    args = ap.parse_args()

    pq = os.path.abspath(args.parquet)
    _reject_placeholder(pq, "--parquet")
    if not os.path.isfile(pq):
        print(f"[FAIL] parquet not found: {pq}")
        return 1

    df = pd.read_parquet(pq)
    n = len(df)
    print("=" * 72)
    print("verify_daytrade_depth_channels")
    print("=" * 72)
    print(f"  File : {pq}")
    print(f"  Rows : {n:,} | cols : {len(df.columns)}")

    fails: list[str] = []
    warns: list[str] = []

    # ── Core columns ──────────────────────────────────────────────────────
    print("\n-- Core depth columns --")
    for col in DEPTH_CORE:
        if col not in df.columns:
            fails.append(f"missing column `{col}`")
            print(f"  [FAIL] `{col}`: MISSING")
            continue
        st = _stats_series(df[col])
        ok_var = st["finite_n"] > 10 and st["std"] > 1e-12
        tag = "[OK]" if ok_var else "[FAIL]"
        if not ok_var:
            fails.append(f"`{col}` degenerate (std~0 or too few finite)")
        print(
            f"  {tag} `{col}`: finite={st['finite_n']:,} null={st['null_pct']:.2%} "
            f"zero~{st['zero_pct']:.2%} std={st['std']:.6g} range=[{st['min']}, {st['max']}]"
        )

    # ── OBI vs LOB independence ─────────────────────────────────────────────
    if "obi" in df.columns and "lob_imbalance" in df.columns:
        print("\n-- OBI vs lob_imbalance (independence check) --")
        o = pd.to_numeric(df["obi"], errors="coerce").to_numpy(dtype=np.float64)
        l = pd.to_numeric(df["lob_imbalance"], errors="coerce").to_numpy(dtype=np.float64)
        m = np.isfinite(o) & np.isfinite(l)
        if int(m.sum()) < 3:
            warns.append("too few rows to correlate obi vs lob")
            print("  [WARN] too few finite pairs")
        else:
            corr = _pearson(o, l)
            diff = np.abs(o[m] - l[m])
            identical_pct = float(np.mean(diff < 1e-9) * 100.0)
            print(f"  Pearson corr : {corr if corr is not None else 'n/a'}")
            print(f"  |obi-lob|<1e-9 : {identical_pct:.2f}% of finite pairs")

            if corr is not None and abs(corr) >= float(args.obi_lob_corr_warn):
                warns.append(f"|corr(obi,lob)|={abs(corr):.6f} >= {args.obi_lob_corr_warn}")
                print("  [WARN] correlation very high - channels likely redundant")
            if identical_pct >= float(args.obi_lob_identical_warn_pct):
                warns.append(f"{identical_pct:.1f}% rows identical obi~lob")
                print("  [WARN] obi and lob_imbalance numerically identical on most rows")

    if {"obi", "order_flow_imbalance"} <= set(df.columns):
        print("\n-- order_flow_imbalance vs obi --")
        o = pd.to_numeric(df["obi"], errors="coerce").to_numpy(dtype=np.float64)
        f = pd.to_numeric(df["order_flow_imbalance"], errors="coerce").to_numpy(dtype=np.float64)
        m = np.isfinite(o) & np.isfinite(f)
        if int(m.sum()) >= 3:
            c = _pearson(o, f)
            print(f"  Pearson corr(obi, order_flow_imbalance): {c}")
            if c is not None and abs(c) >= 0.999:
                warns.append(f"|corr(obi,ofi)|={abs(c):.6f} >= 0.999")
                print("  [WARN] OFI and obi almost perfectly correlated")

    # ── Walls vs buy_ratio (prepare_day_trading proxy) ──────────────────────
    if {"bid_wall_strength", "ask_wall_strength", "buy_ratio"} <= set(df.columns):
        print("\n-- Walls vs buy_ratio (proxy consistency) --")
        br = pd.to_numeric(df["buy_ratio"], errors="coerce").clip(0.0, 1.0).to_numpy(dtype=np.float64)
        bw = pd.to_numeric(df["bid_wall_strength"], errors="coerce").to_numpy(dtype=np.float64)
        aw = pd.to_numeric(df["ask_wall_strength"], errors="coerce").to_numpy(dtype=np.float64)
        m = _finite_mask(df["buy_ratio"]) & _finite_mask(df["bid_wall_strength"]) & _finite_mask(
            df["ask_wall_strength"]
        )
        exp_bw = (br[m] * 2.0).astype(np.float64)
        exp_aw = ((1.0 - br[m]) * 2.0).astype(np.float64)
        rmse_bw = float(np.sqrt(np.mean((bw[m] - exp_bw) ** 2)))
        rmse_aw = float(np.sqrt(np.mean((aw[m] - exp_aw) ** 2)))
        sum_w = bw[m] + aw[m]
        mean_sum = float(np.mean(sum_w))
        print(f"  RMSE(bid_wall vs buy_ratio*2): {rmse_bw:.6f}")
        print(f"  RMSE(ask_wall vs (1-buy_ratio)*2): {rmse_aw:.6f}")
        print(f"  mean(bid_wall+ask_wall): {mean_sum:.6f} (proxy expects ~2.0)")
        if rmse_bw > 0.05 or rmse_aw > 0.05:
            print(
                "  [INFO] walls differ from simple buy_ratio proxy - likely real upstream columns or extra scaling."
            )
        else:
            warns.append("walls closely match buy_ratio proxy - may not be independent LOB walls")
            print(
                "  [WARN] walls track buy_ratio*2 / (1-br)*2 closely - limited independent signal vs buy_ratio"
            )

    elif "bid_wall_strength" in df.columns and "ask_wall_strength" in df.columns:
        print("\n-- Walls sum check --")
        bw = pd.to_numeric(df["bid_wall_strength"], errors="coerce").to_numpy(dtype=np.float64)
        aw = pd.to_numeric(df["ask_wall_strength"], errors="coerce").to_numpy(dtype=np.float64)
        m = np.isfinite(bw) & np.isfinite(aw)
        if int(m.sum()) > 0:
            s = bw[m] + aw[m]
            print(f"  mean(bid_wall+ask_wall)={float(np.mean(s)):.6f} std={float(np.std(s)):.6f}")

    # ── Optional bridge ─────────────────────────────────────────────────────
    print("\n-- Optional bridge columns --")
    for col in OPTIONAL_BRIDGE:
        if col not in df.columns:
            print(f"  - `{col}`: absent")
            continue
        st = _stats_series(df[col])
        print(
            f"  - `{col}`: finite={st['finite_n']:,} null={st['null_pct']:.2%} "
            f"std={st['std']:.6g}"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    if fails:
        print("[FAIL]:")
        for x in fails:
            print(f"   - {x}")
    else:
        print("[OK] No FAIL checks")

    if warns:
        print("[WARN]:")
        for x in warns:
            print(f"   - {x}")
    else:
        print("[OK] No WARN checks")

    print("=" * 72)

    if fails:
        return 1
    if args.strict and warns:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
