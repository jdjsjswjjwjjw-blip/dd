#!/usr/bin/env python3
"""
diagnose_feature_stationarity.py
================================

تشخيص عملي لثبات الميزات ومقياس السعر:
  • ما الذي يدخل CatBoost Stage1 فعلاً (الأساس 31 + VWAP roll إن وُجد)
  • أعمدة OHLC / سعر خام في الجدول لكن خارج Stage1 (أو العكس)
  • مدخلات Stage1: micro_price_rel / pdh_rel / pdl_rel (نسبية)؛ الأعمدة المطلقة القديمة تحذير إن بقيت
  • نسب VWAP، liquidity_density، ووجود scaler_params.json إن وُجد

الاستخدام (من مجلد QuantSystem-master):
  py -3 diagnose_feature_stationarity.py --data path/to/day_trading_features.parquet
  py -3 diagnose_feature_stationarity.py --data path/to/features_0.parquet --scaler pipeline/.../scaler_params.json
"""

from __future__ import annotations

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from prepare_training_data import (
        CATBOOST_ADVISOR_FEATURES,
        CATBOOST_DAY_TRADE_ROLL_VWAP_FEATURES,
        resolve_catboost_stat_columns,
    )
except ImportError as exc:
    raise SystemExit(
        "تشغيل السكربت من جذر QuantSystem-master حيث يوجد prepare_training_data.py\n"
        f"ImportError: {exc}"
    ) from exc

# أسماء شائعة لمستويات سعر خام (ليست بالضرورة كلها مدخلات Stage1)
OHLC_COLS = frozenset({"open", "high", "low", "close"})
PX_LEVEL_COLS = frozenset(
    {
        "price",
        "bid_px_00",
        "ask_px_00",
        "micro_price",
        "current_vwap",
        "pdh",
        "pdl",
        "pwh",
        "pwl",
        "vwap_roll_1h",
        "vwap_roll_4h",
    }
)
BOUNDED_OR_RATIO_HINT = frozenset(
    {
        "obi",
        "spoofing_ratio",
        "spoofing_duration",
        "liquidity_trap",
        "cancel_ratio",
        "buy_ratio",
        "fisher_signal",
        "anomaly",
        "price_position",
        "vwap_z_score",
    }
)


def _load_table(path: str, *, max_rows: int | None) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    suf = p.suffix.lower()
    if suf in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif suf in (".csv", ".txt"):
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported format: {suf} (use .parquet or .csv)")
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[: int(max_rows)].copy()
    return df


def _summarize(s: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce")
    n = len(x)
    n_nan = int(x.isna().sum())
    v = x.dropna().to_numpy(dtype=np.float64)
    if v.size == 0:
        return {
            "n": n,
            "nan_pct": n_nan / max(n, 1),
            "finite_n": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p50": None,
            "p99": None,
        }
    return {
        "n": n,
        "nan_pct": n_nan / max(n, 1),
        "finite_n": int(v.size),
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "mean": float(np.nanmean(v)),
        "std": float(np.nanstd(v, ddof=0)),
        "p01": float(np.nanquantile(v, 0.01)),
        "p50": float(np.nanquantile(v, 0.50)),
        "p99": float(np.nanquantile(v, 0.99)),
    }


def _fmt_row(st: dict[str, Any]) -> str:
    if st["finite_n"] == 0:
        return "(no finite values)"
    return (
        f"n={st['finite_n']:,} nan={st['nan_pct']:.1%} "
        f"std={st['std']:.6g} p01={st['p01']:.6g} p50={st['p50']:.6g} p99={st['p99']:.6g}"
    )


def _load_scaler(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    if not os.path.isfile(path):
        print(f"⚠️ scaler path missing: {path}")
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Feature stationarity / price-scale diagnostics for V19.")
    ap.add_argument("--data", required=True, help="Parquet or CSV (events or bars)")
    ap.add_argument("--max-rows", type=int, default=None, help="Sample first N rows only")
    ap.add_argument("--scaler", default=None, help="Optional scaler_params.json from refinery / train_full")
    args = ap.parse_args()

    df = _load_table(os.path.abspath(args.data), max_rows=args.max_rows)
    cols = set(df.columns.astype(str))
    print(f"📥 Loaded: {args.data}")
    print(f"   shape={df.shape[0]:,} × {df.shape[1]:,}")

    stage1 = resolve_catboost_stat_columns(df.columns)
    print("\n═══ CatBoost Stage1 (resolved) ═══")
    print(f"   count={len(stage1)} → {stage1}")
    missing_base = [c for c in CATBOOST_ADVISOR_FEATURES if c not in cols]
    if missing_base:
        print(f"   ⚠️ Missing CATBOOST_ADVISOR_FEATURES in frame: {missing_base}")
    extras = [c for c in stage1 if c not in CATBOOST_ADVISOR_FEATURES]
    if extras:
        print(f"   ✓ Extras (e.g. roll VWAP): {extras}")
    for ex in CATBOOST_DAY_TRADE_ROLL_VWAP_FEATURES:
        if ex not in cols:
            print(f"   ℹ️ Optional day-trade column absent: {ex}")

    rel_in_stage1 = sorted(set(stage1) & {"micro_price_rel", "pdh_rel", "pdl_rel"})
    print("\n═══ مقاييس Stage1 النسبية (*_rel ≈ انحراف عن الإغلاق / ATR) ═══")
    if rel_in_stage1:
        for name in rel_in_stage1:
            if name in cols:
                print(f"   ✓ {name}: {_fmt_row(_summarize(df[name]))}")
    else:
        print("   ⚠️ لا توجد *_rel في Stage1 — باركيه قديم أو لم يُحدَّث المسار.")

    legacy_abs = sorted(set(stage1) & {"micro_price", "pdh", "pdl"})
    if legacy_abs:
        print("\n═══ تحذير: أسماء سعر مطلق لا تزال ضمن Stage1 (يفضّل *_rel) ═══")
        for name in legacy_abs:
            if name in cols:
                print(f"   ⚠️ {name}: {_fmt_row(_summarize(df[name]))}")

    abs_other = sorted((set(stage1) & (PX_LEVEL_COLS | OHLC_COLS)) - set(rel_in_stage1) - set(legacy_abs))
    if abs_other:
        print("\n═══ مطلق سعر آخر داخل Stage1 ═══")
        for name in abs_other:
            if name in cols:
                print(f"   ⚠️ {name}: {_fmt_row(_summarize(df[name]))}")

    print("\n═══ OHLC في الجدول (غالباً للشمع / باك‌تيست وليس Stage1) ═══")
    for name in sorted(OHLC_COLS & cols):
        in_s1 = "→ Stage1" if name in stage1 else "(ليست من Stage1)"
        print(f"   {name} {in_s1}: {_fmt_row(_summarize(df[name]))}")

    print("\n═══ أعمدة سعر خام أخرى إن وُجدت ═══")
    for name in sorted((PX_LEVEL_COLS | OHLC_COLS) & cols):
        if name in OHLC_COLS:
            continue
        tag = "Stage1" if name in stage1 else "meta/other"
        print(f"   [{tag}] {name}: {_fmt_row(_summarize(df[name]))}")

    print("\n═══ سيولة / VWAP (مقياس ودلالة) ═══")
    for name in ("liquidity_density", "vwap_dist", "vwap_dist_1h_roll"):
        if name not in cols:
            continue
        st = _summarize(df[name])
        print(f"   {name}: {_fmt_row(st)}")
        if name == "vwap_dist" and st["finite_n"] and float(st["std"] or 0) > 50:
            print(
                "      ⚠️ تباين كبير جداً لـ vwap_dist؛ المتوقع نسبة (~وحدة صغيرة). "
                "تحقق من الأعمدة أو من صحة current_vwap/close."
            )
        if name == "vwap_dist_1h_roll" and st["finite_n"]:
            ax = float(np.nanmax(np.abs([st["p01"], st["p99"]]))) if st["p01"] is not None else 0.0
            if ax > 30:
                print(
                    "      ℹ️ قيم كبيرة مقارنة بـ ATR-الطبيعي؛ قد يكون تقلب منخفض أو شمع نادرة."
                )

    print("\n═══ مدخلات Stage1 «محدودة النطاق» المتوقعة ═══")
    for name in sorted(BOUNDED_OR_RATIO_HINT & set(stage1) & cols):
        print(f"   {name}: {_fmt_row(_summarize(df[name]))}")

    scaler = _load_scaler(args.scaler)
    if scaler:
        print("\n═══ scaler_params.json (Robust / binary حسب التدريب) ═══")
        for name in stage1:
            p = scaler.get(name)
            if p is None:
                continue
            typ = p.get("type", "?")
            if typ == "robust":
                print(
                    f"   {name}: robust median={p.get('median')} iqr={p.get('iqr')} "
                    f"min_robust_iqr={p.get('min_robust_iqr')}"
                )
            elif typ == "binary":
                print(f"   {name}: binary")
            else:
                print(f"   {name}: {p}")

    print(
        "\n💡 الخلاصة: OHLC عادةً لا تدخل الـ31؛ Stage1 يستخدم micro_price_rel/pdh_rel/pdl_rel "
        "لتقليل انزلاق المستوى السعري. أعد توليد البيانات والتدريب بعد التحديث."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
