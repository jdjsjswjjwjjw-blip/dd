"""
diagnose_feature_transfer_to_daytrade.py
=======================================

Experimental signal-preservation diagnostics from Stage1 trade-level features
to DayTrade/Entry bars. This is stronger than schema checks: it asks whether
features still carry variance and whether DayTrade columns can be explained by
time-causal aggregations of Stage1 features.
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


STAGE1_DEPTH_FEATURES = [
    "obi",
    "micro_price_rel",
    "bid_wall_strength",
    "ask_wall_strength",
    "distance_to_wall",
    "gap_size",
    "liquidity_density",
    "spoofing_ratio",
    "spoofing_duration",
    "liquidity_trap",
    "cancel_ratio",
]

STAGE1_FLOW_FEATURES = [
    "cvd",
    "kyle_lambda",
    "hawkes_intensity",
    "absorption_intensity",
    "vnet",
    "volume_burst",
    "inter_event_time",
]

DAYTRADE_BRIDGE_FEATURES = [
    "mbp_bar_coverage",
    "mbp_roll_lob_coverage",
    "lob_imbalance",
    "mbp_imbalance_peak",
    "mbp_imbalance_direction_pct",
    "mbp_depth_bid_max",
    "mbp_depth_ask_max",
    "mbp_depth_sum_max",
    "mbp_wall_bid_peak",
    "mbp_wall_ask_peak",
    "mbp_bid_slope_intrabar",
    "mbp_ask_slope_intrabar",
    "mbp_depth_accel",
    "order_flow_imbalance",
    "bar_cvd_delta",
    "cvd_velocity",
    "cvd_net_direction",
    "obi_direction",
]

LABEL_COLS = {
    "bias_label",
    "soft_label",
    "label_confidence",
    "soft_sample_weight",
    "event_flag",
    "train_event_flag",
    "signal_quality",
    "forward_return",
    "path_outcome",
    "neutral_reason",
}


def _load_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.is_file():
        return pd.read_parquet(p)
    files = sorted(p.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _to_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col == "ts_event" or col in LABEL_COLS:
            continue
        s = _num(df[col])
        if s.notna().sum() >= 3:
            cols.append(col)
    return cols


def _profile_series(s: pd.Series) -> dict[str, Any]:
    x = _num(s)
    v = x.dropna()
    if v.empty:
        return {"present": True, "n": int(len(s)), "nan_rate": 1.0, "std": 0.0, "zero_rate": 1.0, "nunique": 0}
    return {
        "present": True,
        "n": int(len(s)),
        "nan_rate": float(x.isna().mean()),
        "std": float(v.std(ddof=0)),
        "zero_rate": float((v.abs() <= 1e-12).mean()),
        "nunique": int(v.nunique(dropna=True)),
        "p01": float(v.quantile(0.01)),
        "p50": float(v.quantile(0.50)),
        "p99": float(v.quantile(0.99)),
    }


def _aggregate_stage1(stage1: pd.DataFrame, freq: str, max_features: int | None = None) -> pd.DataFrame:
    work = stage1.copy()
    work["ts_event"] = _to_ts(work["ts_event"])
    work = work.dropna(subset=["ts_event"]).sort_values("ts_event")
    work["bar_key"] = work["ts_event"].dt.floor(freq)

    cols = _numeric_columns(work)
    if max_features is not None:
        priority = list(dict.fromkeys(STAGE1_DEPTH_FEATURES + STAGE1_FLOW_FEATURES))
        ordered = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
        cols = ordered[: max(int(max_features), len([c for c in priority if c in cols]))]

    agged = work.groupby("bar_key", sort=True)[cols].agg(["mean", "last", "std", "min", "max"])
    agged.columns = [f"{base}__{func}" for base, func in agged.columns]
    agged = agged.reset_index().rename(columns={"bar_key": "ts_event"})
    return agged


def _prepare_daytrade(daytrade: pd.DataFrame, freq: str) -> pd.DataFrame:
    out = daytrade.copy()
    out["ts_event"] = _to_ts(out["ts_event"]).dt.floor(freq)
    return out.sort_values("ts_event").reset_index(drop=True)


def _safe_spearman(a: pd.Series, b: pd.Series) -> float | None:
    x = _num(a)
    y = _num(b)
    m = x.notna() & y.notna()
    if int(m.sum()) < 8:
        return None
    if float(x[m].std(ddof=0)) <= 1e-12 or float(y[m].std(ddof=0)) <= 1e-12:
        return None
    val = x[m].corr(y[m], method="spearman")
    return None if pd.isna(val) else float(val)


def _best_daytrade_sources(merged: pd.DataFrame, stage_cols: list[str], day_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dcol in day_cols:
        if dcol == "ts_event":
            continue
        best_col = None
        best_corr = None
        for scol in stage_cols:
            c = _safe_spearman(merged[scol], merged[dcol])
            if c is None:
                continue
            if best_corr is None or abs(c) > abs(best_corr):
                best_col = scol
                best_corr = c
        prof = _profile_series(merged[dcol])
        rows.append(
            {
                "daytrade_feature": dcol,
                "best_stage1_aggregate": best_col,
                "best_spearman": best_corr,
                "abs_best_spearman": None if best_corr is None else abs(best_corr),
                "daytrade_std": prof["std"],
                "daytrade_zero_rate": prof["zero_rate"],
                "daytrade_nunique": prof["nunique"],
            }
        )
    return rows


def _stage_feature_survival(merged: pd.DataFrame, stage_feature_names: list[str], day_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in stage_feature_names:
        candidates = [c for c in merged.columns if c.startswith(f"{base}__")]
        best_day = None
        best_agg = None
        best_corr = None
        for scol in candidates:
            for dcol in day_cols:
                c = _safe_spearman(merged[scol], merged[dcol])
                if c is None:
                    continue
                if best_corr is None or abs(c) > abs(best_corr):
                    best_corr = c
                    best_agg = scol
                    best_day = dcol
        abs_corr = None if best_corr is None else abs(best_corr)
        if abs_corr is None:
            status = "no_signal_or_no_match"
        elif abs_corr >= 0.70:
            status = "strong"
        elif abs_corr >= 0.35:
            status = "transformed"
        else:
            status = "weak"
        rows.append(
            {
                "stage1_feature": base,
                "best_stage1_aggregate": best_agg,
                "best_daytrade_feature": best_day,
                "best_spearman": best_corr,
                "abs_best_spearman": abs_corr,
                "status": status,
            }
        )
    return rows


def _mutual_information_group(df: pd.DataFrame, cols: list[str], label_col: str = "bias_label") -> dict[str, Any]:
    present = [c for c in cols if c in df.columns]
    if label_col not in df.columns or not present:
        return {"available": False, "reason": "missing label or features", "features": len(present)}
    try:
        from sklearn.feature_selection import mutual_info_classif
    except Exception as exc:
        return {"available": False, "reason": f"sklearn unavailable: {exc}", "features": len(present)}

    x = df[present].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    keep = [c for c in x.columns if float(x[c].std(ddof=0)) > 1e-12]
    if not keep:
        return {"available": True, "features": len(present), "nonconstant": 0, "mean_mi": 0.0, "max_mi": 0.0, "top": []}
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(2).astype(int)
    mi = mutual_info_classif(x[keep].to_numpy(dtype=np.float64), y.to_numpy(), discrete_features=False, random_state=42)
    order = np.argsort(mi)[::-1]
    top = [{"feature": keep[int(i)], "mi": float(mi[int(i)])} for i in order[:10]]
    return {
        "available": True,
        "features": len(present),
        "nonconstant": len(keep),
        "mean_mi": float(np.mean(mi)) if len(mi) else 0.0,
        "max_mi": float(np.max(mi)) if len(mi) else 0.0,
        "top": top,
    }


def _summarize_survival(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [r["abs_best_spearman"] for r in rows if r.get("abs_best_spearman") is not None]
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {
        "features": len(rows),
        "status_counts": statuses,
        "median_abs_best_spearman": None if not vals else float(np.median(vals)),
        "min_abs_best_spearman": None if not vals else float(np.min(vals)),
        "max_abs_best_spearman": None if not vals else float(np.max(vals)),
    }


def _markdown(report: dict[str, Any]) -> str:
    depth = report["survival"]["depth_summary"]
    flow = report["survival"]["flow_summary"]
    raw = report["lineage"]["raw_prefix_summary"]
    lines = [
        "# Feature Transfer To DayTrade Diagnostic",
        "",
        "## Executive Summary",
        f"- Verdict: {report['verdict']}",
        f"- Common bars: {report['alignment']['common_bars']}/{report['alignment']['daytrade_bars']}",
        f"- Stage1 aggregated features tested: {report['alignment']['stage1_aggregate_features']}",
        f"- DayTrade numeric features tested: {report['alignment']['daytrade_numeric_features']}",
        f"- Depth survival: median_abs_corr={depth['median_abs_best_spearman']} statuses={depth['status_counts']}",
        f"- Flow survival: median_abs_corr={flow['median_abs_best_spearman']} statuses={flow['status_counts']}",
        f"- raw__ lineage: present={raw['present']} strong={raw['strong']} weak={raw['weak']}",
        "",
        "## Depth Feature Survival",
    ]
    for row in report["survival"]["depth"][:20]:
        lines.append(
            f"- {row['stage1_feature']}: {row['status']} | best={row['best_daytrade_feature']} "
            f"via {row['best_stage1_aggregate']} | rho={row['best_spearman']}"
        )
    lines += ["", "## Flow Feature Survival"]
    for row in report["survival"]["flow"][:20]:
        lines.append(
            f"- {row['stage1_feature']}: {row['status']} | best={row['best_daytrade_feature']} "
            f"via {row['best_stage1_aggregate']} | rho={row['best_spearman']}"
        )
    lines += ["", "## Mutual Information With Bias Label"]
    for name, mi in report["mutual_information"].items():
        top = ", ".join(f"{x['feature']}={x['mi']:.4g}" for x in mi.get("top", [])[:5])
        lines.append(
            f"- {name}: available={mi.get('available')} features={mi.get('features')} "
            f"mean_mi={mi.get('mean_mi')} max_mi={mi.get('max_mi')} top=[{top}]"
        )
    lines += ["", "## Weakest DayTrade Features By Stage1 Explainability"]
    weak_day = sorted(
        report["daytrade_explainability"],
        key=lambda x: -1.0 if x["abs_best_spearman"] is None else x["abs_best_spearman"],
    )
    weak_day = [x for x in weak_day if x["abs_best_spearman"] is None or x["abs_best_spearman"] < 0.35][:20]
    for row in weak_day:
        lines.append(
            f"- {row['daytrade_feature']}: rho={row['best_spearman']} best_stage={row['best_stage1_aggregate']} "
            f"std={row['daytrade_std']} zero={row['daytrade_zero_rate']:.2%}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Test whether Stage1 signals survive into DayTrade features.")
    ap.add_argument("--stage1", required=True, help="Stage1 final parquet file or directory")
    ap.add_argument("--daytrade", required=True, help="DayTrade day_trading_features.parquet")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--freq", default="5min")
    ap.add_argument("--max_stage1_features", type=int, default=None)
    ap.add_argument("--print_report", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stage1 = _load_parquet(args.stage1)
    daytrade = _prepare_daytrade(_load_parquet(args.daytrade), args.freq)
    stage_agg = _aggregate_stage1(stage1, args.freq, max_features=args.max_stage1_features)

    merged = stage_agg.merge(daytrade, on="ts_event", how="inner", suffixes=("_stageagg", "_daytrade"))
    stage_cols = [c for c in stage_agg.columns if c != "ts_event"]
    day_cols = [c for c in _numeric_columns(daytrade) if c in merged.columns]
    bridge_cols = [c for c in DAYTRADE_BRIDGE_FEATURES if c in merged.columns]
    raw_cols = [c for c in day_cols if c.startswith("raw__")]

    day_explain = _best_daytrade_sources(merged, stage_cols, day_cols)
    depth_survival = _stage_feature_survival(merged, [c for c in STAGE1_DEPTH_FEATURES if c in stage1.columns], day_cols)
    flow_survival = _stage_feature_survival(merged, [c for c in STAGE1_FLOW_FEATURES if c in stage1.columns], day_cols)

    raw_strong = 0
    raw_weak = 0
    raw_lineage: list[dict[str, Any]] = []
    for raw_col in raw_cols:
        base = raw_col.replace("raw__", "", 1)
        candidates = [c for c in stage_cols if c.startswith(f"{base}__")]
        best = None
        best_c = None
        for cand in candidates:
            corr = _safe_spearman(merged[cand], merged[raw_col])
            if corr is not None and (best_c is None or abs(corr) > abs(best_c)):
                best = cand
                best_c = corr
        if best_c is not None and abs(best_c) >= 0.70:
            raw_strong += 1
        else:
            raw_weak += 1
        raw_lineage.append({"raw_feature": raw_col, "best_stage1_aggregate": best, "best_spearman": best_c})

    mi_groups = {
        "stage1_depth_aggregates": [c for c in stage_cols if c.split("__", 1)[0] in STAGE1_DEPTH_FEATURES],
        "stage1_flow_aggregates": [c for c in stage_cols if c.split("__", 1)[0] in STAGE1_FLOW_FEATURES],
        "daytrade_bridge": bridge_cols,
        "daytrade_raw_prefix": raw_cols,
    }

    report = {
        "inputs": {
            "stage1": args.stage1,
            "daytrade": args.daytrade,
            "freq": args.freq,
        },
        "alignment": {
            "stage1_bars": int(len(stage_agg)),
            "daytrade_bars": int(len(daytrade)),
            "common_bars": int(len(merged)),
            "stage1_aggregate_features": int(len(stage_cols)),
            "daytrade_numeric_features": int(len(day_cols)),
        },
        "lineage": {
            "raw_prefix_summary": {
                "present": int(len(raw_cols)),
                "strong": int(raw_strong),
                "weak": int(raw_weak),
            },
            "raw_prefix": raw_lineage,
        },
        "survival": {
            "depth": depth_survival,
            "depth_summary": _summarize_survival(depth_survival),
            "flow": flow_survival,
            "flow_summary": _summarize_survival(flow_survival),
        },
        "daytrade_explainability": day_explain,
        "mutual_information": {name: _mutual_information_group(merged, cols) for name, cols in mi_groups.items()},
    }

    depth_median = report["survival"]["depth_summary"]["median_abs_best_spearman"] or 0.0
    flow_median = report["survival"]["flow_summary"]["median_abs_best_spearman"] or 0.0
    report["verdict"] = "PASS" if len(merged) == len(daytrade) and depth_median >= 0.35 and flow_median >= 0.35 else "REVIEW"

    json_path = os.path.join(args.out_dir, "feature_transfer_to_daytrade.json")
    md_path = os.path.join(args.out_dir, "feature_transfer_to_daytrade.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    markdown = _markdown(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"saved_json={json_path}")
    print(f"saved_md={md_path}")
    if args.print_report:
        print()
        print(markdown)


if __name__ == "__main__":
    main()
