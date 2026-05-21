#!/usr/bin/env python3
"""
diagnose_soft_labels_daytrade.py — التحقق من معايرة Soft Labels على مخرجات DayTrade

يقرأ parquet الشموع (نفس الصفوف التي مرّت على attach_soft_labels_dt) ويجري:
  1) فحوصات بنيوية: ts_event أحادي، NEUTRAL ↔ soft_label≈0.5، إحصاءات اتجاهي
  2) اختياري — إعادة تشغيل Monte Carlo بنفس الإعدادات في day_trading_manifest.json ومقارنة الأعمدة
  3) اختياري — عدّ التيكات لكل شمعة من مجلد MBO parquet للتأكد أن الشمعة مبنية على داتا خام

⚠️ لا تستخدم مسارات تحتوي على «...» — انسخ المسار الكامل من مستكشف الملفات.

مثال (عدّل التاغ إذا كان اسم الملف مختلفًا على جهازك):

  py -3 diagnose_soft_labels_daytrade.py ^
    --parquet "E:\\QuantSystem-master (3) (2)\\QuantSystem-master\\pipeline_mbo2_refinery_latest\\daytrade_run\\day_trading_features_mbo2_refinery_latest.parquet" ^
    --manifest "E:\\QuantSystem-master (3) (2)\\QuantSystem-master\\pipeline_mbo2_refinery_latest\\daytrade_run\\day_trading_manifest_mbo2_refinery_latest.json" ^
    --out-json "E:\\QuantSystem-master (3) (2)\\QuantSystem-master\\soft_label_diag.json"

مع تيكات (نفس مساري parquet و manifest كما فوق):

  py -3 diagnose_soft_labels_daytrade.py ^
    --parquet "E:\\QuantSystem-master (3) (2)\\QuantSystem-master\\pipeline_mbo2_refinery_latest\\daytrade_run\\day_trading_features_mbo2_refinery_latest.parquet" ^
    --manifest "E:\\QuantSystem-master (3) (2)\\QuantSystem-master\\pipeline_mbo2_refinery_latest\\daytrade_run\\day_trading_manifest_mbo2_refinery_latest.json" ^
    --mbo-dir "E:\\QuantSystem-master (3) (2)\\QuantSystem-master\\pipeline_mbo2_refinery_latest\\normalized\\mbo" ^
    --freq 5min

استخدم --skip-mc-replay إذا كان عدد الشموع كبيرًا والـ MC إعادة تشغيل بطيئة.
"""

from __future__ import annotations

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd


DIR_LONG = 0
DIR_SHORT = 1
DIR_NEUTRAL = 2


def _load_manifest(path: str | None) -> dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _need_cols_mc(df: pd.DataFrame) -> list[str]:
    need = ["close", "bias_label", "signal_quality", "effective_horizon"]
    missing = [c for c in need if c not in df.columns]
    return missing


def _dyn_thr_ok(df: pd.DataFrame) -> bool:
    if "label_dynamic_threshold" in df.columns:
        s = pd.to_numeric(df["label_dynamic_threshold"], errors="coerce")
        return bool(s.notna().any())
    if "micro_atr" in df.columns:
        s = pd.to_numeric(df["micro_atr"], errors="coerce")
        return bool(s.notna().any())
    if "forward_return" in df.columns and "path_outcome" in df.columns:
        return True
    return False


def structural_report(df: pd.DataFrame) -> dict[str, Any]:
    ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    mono = bool(ts.is_monotonic_increasing)
    dup_ts = int(ts.duplicated().sum())

    bias = pd.to_numeric(df["bias_label"], errors="coerce").fillna(DIR_NEUTRAL).astype(np.int8).to_numpy()
    sl = pd.to_numeric(df["soft_label"], errors="coerce").fillna(0.5).astype(np.float64).to_numpy()

    neu = bias == DIR_NEUTRAL
    dir_m = (bias == DIR_LONG) | (bias == DIR_SHORT)

    neu_half = (
        float(np.mean(np.abs(sl[neu] - 0.5) < 1e-5))
        if np.any(neu)
        else float("nan")
    )
    dir_sl_min = float(np.min(sl[dir_m])) if np.any(dir_m) else float("nan")
    dir_sl_max = float(np.max(sl[dir_m])) if np.any(dir_m) else float("nan")

    long_m = bias == DIR_LONG
    short_m = bias == DIR_SHORT
    # MC يُنتج soft_label = P(ربح)؛ LONG قد يكون <0.5 وSHORT قد يكون >0.5 دون أن يكون ذلك خطأًا.
    directional_soft_p50 = float(np.median(sl[dir_m])) if np.any(dir_m) else float("nan")

    return {
        "rows": int(len(df)),
        "ts_monotonic_increasing": mono,
        "duplicate_ts_event": dup_ts,
        "neutral_share": float(np.mean(neu)),
        "neutral_rows_soft_is_exactly_05_share": neu_half,
        "directional_soft_label_min": dir_sl_min,
        "directional_soft_label_max": dir_sl_max,
        "directional_soft_label_median": directional_soft_p50,
        "directional_long_share_soft_below_half": float(np.mean(sl[long_m] < 0.5)) if np.any(long_m) else float("nan"),
        "directional_short_share_soft_above_half": float(np.mean(sl[short_m] > 0.5)) if np.any(short_m) else float("nan"),
        "dynamic_threshold_source_ok": _dyn_thr_ok(df),
    }


def optional_tick_density(
    df: pd.DataFrame,
    mbo_dir: str,
    freq: str,
    *,
    max_mbo_rows: int | None = 2_000_000,
) -> dict[str, Any] | None:
    """عدّ التيكات لكل bar_key ودمجه مع الشموع."""
    import glob

    path = os.path.abspath(mbo_dir)
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        return {"available": False, "reason": f"no parquet shards in {path}"}

    chunks = []
    total = 0
    for fp in files:
        part = pd.read_parquet(fp, columns=["ts_event"])
        chunks.append(part)
        total += len(part)
        if max_mbo_rows is not None and total >= max_mbo_rows:
            break
    if not chunks:
        return {"available": False, "reason": "empty mbo read"}

    mbo = pd.concat(chunks, ignore_index=True)
    mbo["ts_event"] = pd.to_datetime(mbo["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    mbo = mbo.dropna(subset=["ts_event"])
    mbo["bar_key"] = mbo["ts_event"].dt.floor(freq)
    # اسم فريد: parquet الشموع قد يحوي بالفعل `tick_count` → merge كان يُنشئ tick_count_x/y فيسبب KeyError
    tick_counts = mbo.groupby("bar_key", sort=False).size().rename("_mbo_ticks_diag")

    bars = df.copy()
    bars["bar_key"] = pd.to_datetime(bars["ts_event"], utc=True, errors="coerce").dt.tz_localize(None).dt.floor(
        freq
    )
    merged = bars.merge(tick_counts, on="bar_key", how="left")
    tc = pd.to_numeric(merged["_mbo_ticks_diag"], errors="coerce").fillna(0.0).to_numpy()

    bias = pd.to_numeric(merged["bias_label"], errors="coerce").fillna(DIR_NEUTRAL).astype(np.int8).to_numpy()
    dir_m = (bias == DIR_LONG) | (bias == DIR_SHORT)
    sl = pd.to_numeric(merged["soft_label"], errors="coerce").fillna(0.5).to_numpy()

    return {
        "available": True,
        "mbo_rows_used": int(len(mbo)),
        "bars_zero_ticks": int(np.sum(tc <= 0)),
        "bars_zero_ticks_share": float(np.mean(tc <= 0)),
        "tick_count_mean": float(np.mean(tc)),
        "tick_count_directional_mean": float(np.mean(tc[dir_m])) if np.any(dir_m) else float("nan"),
        "spearman_ticks_vs_abs_soft_minus_half": _spearman_safe(tc, np.abs(sl - 0.5)),
    }


def _spearman_safe(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 5:
        return None
    da = pd.Series(a)
    db = pd.Series(b)
    if float(da.std(ddof=0)) < 1e-12 or float(db.std(ddof=0)) < 1e-12:
        return None
    return float(da.corr(db, method="spearman"))


def run_mc_replay(
    df: pd.DataFrame,
    *,
    n_scenarios: int,
    horizon_std: float,
    tp_std: float,
    sl_std: float,
    random_seed: int,
) -> dict[str, Any]:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from modules.soft_label_engine import SoftLabelConfig, SoftLabelEngine

    cfg = SoftLabelConfig(
        mode="monte_carlo",
        n_scenarios=int(max(1, n_scenarios)),
        horizon_std=float(max(0.0, horizon_std)),
        tp_std=float(max(0.0, tp_std)),
        sl_std=float(max(0.0, sl_std)),
        random_seed=int(random_seed),
    )
    engine = SoftLabelEngine(cfg)
    work = df.copy()
    replay = engine._attach_monte_carlo(work)

    stored = pd.to_numeric(df["soft_label"], errors="coerce").fillna(0.5).to_numpy(dtype=np.float64)
    replay_sl = replay["soft_label"].to_numpy(dtype=np.float64)

    bias = pd.to_numeric(df["bias_label"], errors="coerce").fillna(DIR_NEUTRAL).astype(np.int8).to_numpy()
    dir_m = (bias == DIR_LONG) | (bias == DIR_SHORT)
    neu = bias == DIR_NEUTRAL

    diff_all = np.abs(replay_sl - stored)
    max_diff_dir = float(np.max(diff_all[dir_m])) if np.any(dir_m) else 0.0
    mean_diff_dir = float(np.mean(diff_all[dir_m])) if np.any(dir_m) else 0.0
    max_diff_neu = float(np.max(diff_all[neu])) if np.any(neu) else 0.0

    ok = max_diff_dir < 1e-4 and max_diff_neu < 1e-4

    return {
        "replay_ran": True,
        "config": {
            "n_scenarios": cfg.n_scenarios,
            "horizon_std": cfg.horizon_std,
            "tp_std": cfg.tp_std,
            "sl_std": cfg.sl_std,
            "random_seed": cfg.random_seed,
        },
        "max_abs_diff_directional": max_diff_dir,
        "mean_abs_diff_directional": mean_diff_dir,
        "max_abs_diff_neutral": max_diff_neu,
        "replay_matches_parquet": ok,
    }


def _resolve_existing_file(kind: str, path_raw: str) -> str:
    """مسار مطلق + رسالة واضحة إذا كان placeholder أو الملف غير موجود."""
    p = os.path.abspath(os.path.expanduser(str(path_raw).strip()))
    bad_placeholder = (
        path_raw.strip().startswith("...")
        or "/..." in path_raw
        or "\\..." in path_raw
        or p.endswith(os.path.join("\\...", "..."))
        or ".\\..." in path_raw
        or "E:\\..." in path_raw
        or 'E:/...' in path_raw
    )
    if bad_placeholder or "..." in path_raw:
        raise SystemExit(
            f"❌ مسار {kind} يحتوي على «...» — هذا كان مثالًا في التوثيق وليس مسارًا حقيقيًا.\n"
            f"   مرّر المسار الكامل للملف على جهازك (انسخه من مستكشف الملفات).\n"
            f"   استلمت: {path_raw!r}"
        )
    if not os.path.isfile(p):
        raise SystemExit(
            f"❌ ملف {kind} غير موجود:\n   {p}\n"
            "   تأكد من اسم الملف (مثل day_trading_features_<tag>.parquet) ومجلد --output عند تشغيل prepare_day_trading.py."
        )
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose soft labels on DayTrade parquet")
    ap.add_argument("--parquet", required=True, help="day_trading_features_*.parquet")
    ap.add_argument("--manifest", default=None, help="day_trading_manifest_*.json (MC params + freq)")
    ap.add_argument("--mbo-dir", default=None, help="optional normalized/mbo shard folder for tick counts")
    ap.add_argument("--freq", default=None, help="bar freq for tick grouping (default from manifest or 5min)")
    ap.add_argument("--skip-mc-replay", action="store_true", help="skip slow Monte Carlo replay")
    ap.add_argument(
        "--force-mc-replay",
        action="store_true",
        help="replay Monte Carlo even if manifest soft_label_mode ≠ monte_carlo (uses manifest/CLI MC params)",
    )
    ap.add_argument("--mc-scenarios", type=int, default=None)
    ap.add_argument("--mc-seed", type=int, default=None)
    ap.add_argument("--mc-h-std", type=float, default=None)
    ap.add_argument("--mc-tp-std", type=float, default=None)
    ap.add_argument("--mc-sl-std", type=float, default=None)
    ap.add_argument("--out-json", default=None, help="write full report JSON path")
    args = ap.parse_args()

    out_json_resolved: str | None = None
    if args.out_json:
        oj = str(args.out_json).strip()
        if "..." in oj:
            raise SystemExit(
                "❌ --out-json يحتوي على «...». مرّر مسار ملف JSON كامل (مثال: "
                "E:\\QuantSystem-master (3) (2)\\QuantSystem-master\\soft_label_diag.json)."
            )
        out_json_resolved = os.path.abspath(os.path.expanduser(oj))
        parent = os.path.dirname(out_json_resolved)
        if parent and not os.path.isdir(parent):
            raise SystemExit(f"❌ مجلد --out-json غير موجود: {parent}")

    if args.mbo_dir:
        md = str(args.mbo_dir).strip()
        if "..." in md:
            raise SystemExit(
                "❌ --mbo-dir يحتوي على «...». مرّر المسار الكامل لمجلد normalized\\mbo."
            )
        md_abs = os.path.abspath(os.path.expanduser(md))
        if not os.path.isdir(md_abs):
            raise SystemExit(f"❌ مجلد MBO غير موجود:\n   {md_abs}")
        args.mbo_dir = md_abs

    parquet_path = _resolve_existing_file("parquet", args.parquet)
    manifest_path = None
    if args.manifest:
        manifest_path = _resolve_existing_file("manifest", args.manifest)

    manifest = _load_manifest(manifest_path)
    freq = args.freq or manifest.get("freq") or "5min"

    df = pd.read_parquet(parquet_path)
    if "ts_event" not in df.columns:
        raise SystemExit("❌ parquet missing ts_event")

    df_sorted = df.sort_values("ts_event").reset_index(drop=True)

    report: dict[str, Any] = {
        "parquet": parquet_path,
        "manifest": manifest_path,
        "freq_used": freq,
        "structural_unsorted": structural_report(df),
        "structural_sorted": structural_report(df_sorted),
    }

    miss = _need_cols_mc(df_sorted)
    if miss:
        report["mc_replay"] = {"replay_ran": False, "reason": f"missing columns: {miss}"}
    elif args.skip_mc_replay:
        report["mc_replay"] = {"replay_ran": False, "reason": "--skip-mc-replay"}
    else:
        mode_m = str(manifest.get("soft_label_mode", "")).strip().lower()
        if mode_m == "analytical" and not args.force_mc_replay:
            report["mc_replay"] = {
                "replay_ran": False,
                "reason": (
                    "manifest soft_label_mode=analytical — MC replay skipped "
                    "(use --force-mc-replay if parquet was produced with monte_carlo)"
                ),
            }
        else:
            def _pick(key: str, cli_val, default):
                if cli_val is not None:
                    return cli_val
                if key in manifest:
                    return manifest[key]
                return default

            mc_scenarios = int(_pick("soft_label_n_scenarios", args.mc_scenarios, 200))
            mc_seed = int(_pick("soft_label_random_seed", args.mc_seed, 42))
            mc_h = float(_pick("soft_label_horizon_std", args.mc_h_std, 0.30))
            mc_tp = float(_pick("soft_label_tp_std", args.mc_tp_std, 0.20))
            mc_sl = float(_pick("soft_label_sl_std", args.mc_sl_std, 0.20))
            if len(df_sorted) > 8000:
                print(
                    f"  ⚠️ {len(df_sorted):,} bars — MC replay may take a while; "
                    "use --skip-mc-replay for structural-only.",
                    file=sys.stderr,
                )
            report["mc_replay"] = run_mc_replay(
                df_sorted,
                n_scenarios=mc_scenarios,
                horizon_std=mc_h,
                tp_std=mc_tp,
                sl_std=mc_sl,
                random_seed=mc_seed,
            )

    if args.mbo_dir:
        report["tick_density"] = optional_tick_density(df_sorted, str(args.mbo_dir), freq)

    # ── Human-readable summary ─────────────────────────────────────────────
    print("=" * 62)
    print("🔎 Soft Labels — تشخيص DayTrade (صفوف الشموع)")
    print("=" * 62)
    su = report["structural_unsorted"]
    ss = report["structural_sorted"]
    print(f"  الصفوف           : {su['rows']:,}")
    print(f"  ts أحادي (كما في الملف)     : {su['ts_monotonic_increasing']} | duplicate_ts={su['duplicate_ts_event']}")
    print(f"  ts أحادي (بعد sort)       : {ss['ts_monotonic_increasing']}")
    print(f"  NEUTRAL share    : {su['neutral_share']:.1%}")
    print(f"  NEUTRAL و soft≈0.5 : {su['neutral_rows_soft_is_exactly_05_share']:.1%}")
    print(f"  اتجاهي soft ∈    : [{su['directional_soft_label_min']:.4f}, {su['directional_soft_label_max']:.4f}] "
          f"| وسيط={su['directional_soft_label_median']:.4f}")
    print(
        "  اتجاهي LONG مع soft<0.5  : "
        f"{su['directional_long_share_soft_below_half']:.1%} "
        "(طبيعي تحت MC إذا السيناريوهات تميل للخسارة)"
    )
    print(
        "  اتجاهي SHORT مع soft>0.5 : "
        f"{su['directional_short_share_soft_above_half']:.1%} "
        "(طبيعي تحت MC إذا السيناريوهات تميل للربح)"
    )
    print(f"  مصدر dyn_thr جاهز : {su['dynamic_threshold_source_ok']}")

    mc = report.get("mc_replay") or {}
    print()
    print("── Monte Carlo replay (نفس manifest + ترتيب زمني) ──")
    if not mc.get("replay_ran"):
        print(f"  ⏭️  تخطي: {mc.get('reason', 'unknown')}")
    else:
        cfg = mc.get("config") or {}
        print(f"  scenarios={cfg.get('n_scenarios')} seed={cfg.get('random_seed')} "
              f"h_std={cfg.get('horizon_std')} tp_std={cfg.get('tp_std')} sl_std={cfg.get('sl_std')}")
        print(f"  max|Δsoft| اتجاهي: {mc.get('max_abs_diff_directional'):.6g}")
        print(f"  mean|Δsoft| اتجاهي: {mc.get('mean_abs_diff_directional'):.6g}")
        print(f"  max|Δsoft| NEUTRAL: {mc.get('max_abs_diff_neutral'):.6g}")
        if mc.get("replay_matches_parquet"):
            print("  ✅ إعادة التشغيل تطابق أعمدة parquet (معيار تعويم)")
        else:
            print("  ❌ اختلاف — غالبًا ترتيب الصفوف عند الحفظ ≠ زمني، أو معاملات MC مختلفة عن manifest")

    td = report.get("tick_density")
    if td:
        print()
        print("── كثافة التيكات (اختياري) ──")
        if not td.get("available"):
            print(f"  ⚠️ {td.get('reason')}")
        else:
            print(f"  صفوف MBO مستخدمة: {td['mbo_rows_used']:,}")
            print(f"  شموع بدون تيكات: {td['bars_zero_ticks']} ({td['bars_zero_ticks_share']:.1%})")
            print(f"  Spearman(tick_count, |soft−0.5|): {td.get('spearman_ticks_vs_abs_soft_minus_half')}")

    print("=" * 62)

    if out_json_resolved:
        with open(out_json_resolved, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Wrote JSON → {out_json_resolved}")


if __name__ == "__main__":
    main()
