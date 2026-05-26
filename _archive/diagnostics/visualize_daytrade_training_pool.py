#!/usr/bin/env python3
"""
visualize_daytrade_training_pool.py — رسم صفوف بركة التدريب DayTrade + تكوين كل شمعة

يعرض الشموع الزمنية، يظلّل قطعة train/holdout من refinery_split، ويُبرز الصفوف حيث:
  train_event_flag == 1  و bias_label ∈ {LONG, SHORT}

اختياريًا يطبّق نفس فلتر train_v19 الأولي: mbp_bar_coverage >= 0.30

مع --mbo-dir يُعاد عدّ التيكات لكل شمعة من shards الـ MBO للتحقق مقابل num_trades في الباركيه.

مثال:

  py -3 visualize_daytrade_training_pool.py ^
    --parquet "E:\\QuantSystem-master (3) (2)\\QuantSystem-master\\pipeline_mbo2_refinery_latest\\daytrade_run\\day_trading_features_mbo2_refinery_latest.parquet" ^
    --split-json "E:\\QuantSystem-master (3) (2)\\QuantSystem-master\\pipeline_mbo2_refinery_latest\\daytrade_run\\refinery_split_mbo2_refinery_latest.json" ^
    --manifest "E:\\QuantSystem-master (3) (2)\\QuantSystem-master\\pipeline_mbo2_refinery_latest\\daytrade_run\\day_trading_manifest_mbo2_refinery_latest.json" ^
    --mbo-dir "E:\\QuantSystem-master (3) (2)\\QuantSystem-master\\pipeline_mbo2_refinery_latest\\normalized\\mbo" ^
    --out-png "E:\\QuantSystem-master (3) (2)\\QuantSystem-master\\training_pool.png" ^
    --out-csv "E:\\QuantSystem-master (3) (2)\\QuantSystem-master\\training_pool_rows.csv"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("❌ مطلوب matplotlib — ثبّتها: pip install matplotlib") from e

DIR_LONG = 0
DIR_SHORT = 1
DIR_NEUTRAL = 2

COMPOSITION_PRIORITY: tuple[str, ...] = (
    "tick_count_mbo",
    "num_trades",
    "buy_volume",
    "sell_volume",
    "buy_ratio",
    "order_flow_imbalance",
    "bar_cvd_delta",
    "volume_burst",
    "mbp_bar_coverage",
    "mbp_roll_lob_coverage",
    "event_score",
    "signal_quality",
    "regime_label",
    "trade_duration",
    "is_event",
    "train_event_flag",
    "bias_label",
    "soft_label",
    "forward_return",
    "path_outcome",
    "neutral_reason",
)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tick_counts_per_bar(mbo_dir: str, freq: str, *, max_rows: int | None = 5_000_000) -> pd.Series:
    """Series indexed by bar_key (Timestamp naive UTC stripped) -> tick count."""
    path = os.path.abspath(mbo_dir)
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no *.parquet under {path}")

    parts: list[pd.DataFrame] = []
    n = 0
    for fp in files:
        part = pd.read_parquet(fp, columns=["ts_event"])
        parts.append(part)
        n += len(part)
        if max_rows is not None and n >= max_rows:
            break
    mbo = pd.concat(parts, ignore_index=True)
    mbo["ts_event"] = pd.to_datetime(mbo["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    mbo = mbo.dropna(subset=["ts_event"])
    mbo["bar_key"] = mbo["ts_event"].dt.floor(freq)
    return mbo.groupby("bar_key", sort=False).size().rename("tick_count_mbo")


def _training_masks(
    df: pd.DataFrame,
    *,
    apply_train_v19_mbp: bool,
    mbp_threshold: float,
    restrict_train_slice: bool,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (pool_gate, pool_effective, chronological_train_zone)."""
    te = pd.to_numeric(df.get("train_event_flag", 0), errors="coerce").fillna(0).astype(int)
    bias = pd.to_numeric(df.get("bias_label", DIR_NEUTRAL), errors="coerce").fillna(DIR_NEUTRAL).astype(int)
    directional = bias.isin([DIR_LONG, DIR_SHORT])

    pool_gate = (te == 1) & directional

    pool_eff = pool_gate.copy()
    if apply_train_v19_mbp and "mbp_bar_coverage" in df.columns:
        cov = pd.to_numeric(df["mbp_bar_coverage"], errors="coerce").fillna(0.0)
        pool_eff = pool_eff & (cov >= float(mbp_threshold))

    if restrict_train_slice and "is_train_slice" in df.columns:
        its = pd.to_numeric(df["is_train_slice"], errors="coerce").fillna(0).astype(int)
        pool_eff = pool_eff & (its == 1)

    chrono_train = pd.Series(True, index=df.index)
    if restrict_train_slice and "is_train_slice" in df.columns:
        chrono_train = its == 1

    return pool_gate, pool_eff, chrono_train


def _bias_name(v: int) -> str:
    if int(v) == DIR_LONG:
        return "LONG"
    if int(v) == DIR_SHORT:
        return "SHORT"
    return "NEUTRAL"


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize DayTrade training pool rows")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--manifest", default=None, help="for freq default")
    ap.add_argument("--split-json", default=None, help="refinery_split_*.json (train/holdout shading)")
    ap.add_argument("--freq", default=None)
    ap.add_argument("--mbo-dir", default=None)
    ap.add_argument("--apply-train-v19-mbp-filter", action="store_true", help="require mbp_bar_coverage>=threshold")
    ap.add_argument("--mbp-coverage-threshold", type=float, default=0.30)
    ap.add_argument(
        "--restrict-train-slice",
        action="store_true",
        help="only rows with is_train_slice==1 (زمنيًا قبل split)",
    )
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--title", default="DayTrade — بركة التدريب (train_event_flag ∧ اتجاهي)")
    args = ap.parse_args()

    pq = os.path.abspath(args.parquet)
    if not os.path.isfile(pq):
        raise SystemExit(f"❌ parquet not found: {pq}")

    manifest: dict[str, Any] = {}
    if args.manifest and os.path.isfile(args.manifest):
        manifest = _load_json(args.manifest)

    freq = args.freq or manifest.get("freq") or "5min"

    df = pd.read_parquet(pq)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.sort_values("ts_event").reset_index(drop=True)

    pool_gate, pool_eff, zone_train = _training_masks(
        df,
        apply_train_v19_mbp=args.apply_train_v19_mbp_filter,
        mbp_threshold=args.mbp_coverage_threshold,
        restrict_train_slice=args.restrict_train_slice,
    )

    split_meta: dict[str, Any] = {}
    split_time = None
    if args.split_json and os.path.isfile(args.split_json):
        split_meta = _load_json(args.split_json)
        st = split_meta.get("split_time")
        if st:
            split_time = pd.Timestamp(st)

    if args.mbo_dir:
        tick_ser = _tick_counts_per_bar(args.mbo_dir, freq)
        df["bar_key"] = df["ts_event"].dt.floor(freq)
        df["tick_count_mbo"] = df["bar_key"].map(tick_ser).fillna(0).astype(np.int64)
    else:
        df["tick_count_mbo"] = np.nan

    comp_cols = [c for c in COMPOSITION_PRIORITY if c in df.columns]
    extra = [c for c in df.columns if c not in comp_cols and c not in ("ts_event", "bar_key")]
    # أعمدة رقمية/مهمة إضافية حتى ~25 عمودًا في CSV
    tail = [c for c in extra if df[c].dtype != object][: max(0, 40 - len(comp_cols))]
    export_cols = ["ts_event"] + comp_cols + tail

    pool_rows = df.loc[pool_eff].copy()
    summary_csv = args.out_csv or os.path.splitext(args.out_png)[0] + "_rows.csv"
    pool_rows[export_cols].to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    ts = df["ts_event"]
    close = pd.to_numeric(df["close"], errors="coerce")

    if zone_train is not None and split_time is not None:
        ax.axvspan(ts.min(), split_time, color="#c8e6c9", alpha=0.35, label="train slice (chrono)")
        ax.axvspan(split_time, ts.max(), color="#bbdefb", alpha=0.35, label="holdout slice")
        ax.axvline(split_time, color="gray", linestyle="--", linewidth=1, alpha=0.8)

    ax.plot(ts, close, color="#37474f", linewidth=1.2, label="close")

    long_sel = pool_eff & (pd.to_numeric(df["bias_label"], errors="coerce").fillna(2).astype(int) == DIR_LONG)
    short_sel = pool_eff & (pd.to_numeric(df["bias_label"], errors="coerce").fillna(2).astype(int) == DIR_SHORT)
    ax.scatter(
        df.loc[long_sel, "ts_event"],
        pd.to_numeric(df.loc[long_sel, "close"], errors="coerce"),
        color="#2e7d32",
        s=140,
        marker="^",
        zorder=5,
        edgecolors="black",
        linewidths=0.8,
        label=f"pool LONG ({int(long_sel.sum())})",
    )
    ax.scatter(
        df.loc[short_sel, "ts_event"],
        pd.to_numeric(df.loc[short_sel, "close"], errors="coerce"),
        color="#c62828",
        s=140,
        marker="v",
        zorder=5,
        edgecolors="black",
        linewidths=0.8,
        label=f"pool SHORT ({int(short_sel.sum())})",
    )

    # تمييز صفوف ضمن البوابة لكن أُسقطت بفلتر MBP أو train slice
    dropped = pool_gate & (~pool_eff)
    if dropped.any():
        ax.scatter(
            df.loc[dropped, "ts_event"],
            pd.to_numeric(df.loc[dropped, "close"], errors="coerce"),
            color="#ff9800",
            s=90,
            marker="x",
            zorder=4,
            linewidths=1.5,
            label=f"train_event لكن خارج الفلتر الفعّال ({int(dropped.sum())})",
        )

    ax.set_title(args.title, fontsize=11)
    ax.set_ylabel("close")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()

    notes = (
        f"freq={freq} | gate_rows={int(pool_gate.sum())} | effective_pool={int(pool_eff.sum())}\n"
        f"train_v19_mbp_filter={args.apply_train_v19_mbp_filter} "
        f"(≥{args.mbp_coverage_threshold}) | restrict_train_slice={args.restrict_train_slice}"
    )
    fig.text(0.01, 0.02, notes, fontsize=8, family="monospace", transform=fig.transFigure)

    out_png = os.path.abspath(args.out_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # طباعة جدول نصي للصفوف الفعّالة
    print("=" * 70)
    print("📌 بركة التدريب الفعّالة (effective_pool)")
    print("=" * 70)
    print(f"  الصفوف الكلية في الباركيه     : {len(df):,}")
    print(f"  train_event_flag ∧ اتجاهي (gate): {int(pool_gate.sum()):,}")
    print(f"  بعد الفلاتر الإضافية (effective): {int(pool_eff.sum()):,}")
    if dropped.any():
        print(f"  ⚠️ صفوف ذات train_event لكن مُستثناة بالفلتر: {int(dropped.sum()):,}")
    print(f"\n  PNG → {out_png}")
    print(f"  CSV → {os.path.abspath(summary_csv)}")
    print()

    show_cols = [c for c in ["ts_event", "bias_label", "num_trades", "tick_count_mbo", "buy_ratio", "mbp_bar_coverage", "event_score", "regime_label", "soft_label"] if c in pool_rows.columns]
    if len(pool_rows):
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(pool_rows[show_cols].to_string(index=False))
    else:
        print("  (لا توجد صفوف في effective_pool — جرّب إزالة --restrict-train-slice أو فلتر MBP)")
    print("=" * 70)


if __name__ == "__main__":
    main()
