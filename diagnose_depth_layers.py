"""
diagnose_depth_layers.py
========================

Layer-by-layer diagnostics for the depth path:

1) Stage1 refinery depth/microstructure features.
2) DayTrade bar bridge features that combine order-flow with MBP depth.
3) LOB tensor health and timestamp alignment.

The goal is explanatory diagnostics, not model scoring.
"""

from __future__ import annotations

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


def _load_parquet(path: str, max_files: int = 50) -> pd.DataFrame:
    p = Path(path)
    if p.is_file():
        return pd.read_parquet(p)
    files = sorted(p.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {path}")
    use_files = files[: max(int(max_files), 1)]
    return pd.concat([pd.read_parquet(f) for f in use_files], ignore_index=True)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _feature_profile(df: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(df)
    for feat in features:
        if feat not in df.columns:
            rows.append({"feature": feat, "present": False})
            continue
        s = _num(df[feat])
        v = s.dropna()
        rows.append(
            {
                "feature": feat,
                "present": True,
                "nan_rate": float(s.isna().mean()) if n else 0.0,
                "zero_rate": float((v == 0).mean()) if len(v) else 1.0,
                "non_zero_rate": float((v != 0).mean()) if len(v) else 0.0,
                "nunique": int(v.nunique(dropna=True)) if len(v) else 0,
                "mean": float(v.mean()) if len(v) else None,
                "std": float(v.std(ddof=0)) if len(v) else None,
                "p01": float(v.quantile(0.01)) if len(v) else None,
                "p50": float(v.quantile(0.50)) if len(v) else None,
                "p99": float(v.quantile(0.99)) if len(v) else None,
            }
        )
    return rows


def _summary_from_profile(profile: list[dict[str, Any]]) -> dict[str, Any]:
    present = [r for r in profile if r.get("present")]
    dead = [
        r["feature"]
        for r in present
        if float(r.get("zero_rate", 0.0)) >= 0.999
        or int(r.get("nunique", 0)) <= 1
        or float(r.get("std") or 0.0) <= 1e-10
    ]
    weak = [
        r["feature"]
        for r in present
        if r["feature"] not in dead
        and (
            float(r.get("zero_rate", 0.0)) >= 0.98
            or int(r.get("nunique", 0)) <= 3
            or float(r.get("std") or 0.0) < 1e-6
        )
    ]
    return {
        "present": len(present),
        "missing": [r["feature"] for r in profile if not r.get("present")],
        "dead": dead,
        "weak": weak,
    }


def _tensor_health(lob_path: str | None, lob_ts_path: str | None) -> dict[str, Any]:
    if not lob_path or not os.path.exists(lob_path):
        return {"available": False, "reason": "lob file missing"}
    arr = np.load(lob_path, mmap_mode="r")
    sample_n = min(len(arr), 512)
    sample = np.asarray(arr[:sample_n], dtype=np.float32)
    finite = np.isfinite(sample)
    out: dict[str, Any] = {
        "available": True,
        "path": str(lob_path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sample_rows": int(sample_n),
        "nan_ratio": float(np.isnan(sample).mean()) if sample.size else 0.0,
        "inf_ratio": float(np.isinf(sample).mean()) if sample.size else 0.0,
        "zero_ratio": float((sample == 0).mean()) if sample.size else 0.0,
        "finite_ratio": float(finite.mean()) if sample.size else 0.0,
        "global_std": float(np.nanstd(sample)) if sample.size else 0.0,
    }
    if sample.ndim == 4:
        snap_std = np.nanstd(sample, axis=(1, 2, 3))
        out["collapsed_snapshot_ratio"] = float(np.mean(snap_std < 1e-8))
        out["snapshot_std_p10"] = float(np.percentile(snap_std, 10))
        out["snapshot_std_p50"] = float(np.percentile(snap_std, 50))
        # prepare_day_trading builds tensors differently for MBP vs MBO-only:
        # - MBP rolling: ch0=log depth, ch1=(buy_fp−sell_fp)/tot (often ~0 when balanced), ch2=log1p(total footprint)
        # - MBO-only: ch0=bar order imbalance (broadcast across levels), ch1=buy trade mass, ch2=sell trade mass
        std_ch0_across_levels = np.nanstd(sample[..., 0], axis=2)
        mean_lvl_std_ch0 = float(np.mean(std_ch0_across_levels)) if std_ch0_across_levels.size else 0.0
        out["ch0_cross_level_std_mean"] = mean_lvl_std_ch0
        if mean_lvl_std_ch0 < 1e-5:
            out["lob_layout_guess"] = "mbo_bar_scalars_broadcast"
            channel_names = [
                "bar_order_imbalance",
                "buy_trade_mass_norm",
                "sell_trade_mass_norm",
            ]
        else:
            out["lob_layout_guess"] = "mbp_depth_plus_footprint_imbalance_and_total"
            channel_names = [
                "depth_log",
                "footprint_imbalance",
                "footprint_total_log",
            ]
        channels = {}
        for i in range(sample.shape[-1]):
            ch = sample[..., i]
            name = channel_names[i] if i < len(channel_names) else f"channel_{i}"
            channels[name] = {
                "mean": float(np.nanmean(ch)),
                "std": float(np.nanstd(ch)),
                "zero_ratio": float((ch == 0).mean()),
                "p01": float(np.nanpercentile(ch, 1)),
                "p99": float(np.nanpercentile(ch, 99)),
            }
        out["channels"] = channels
    if lob_ts_path and os.path.exists(lob_ts_path):
        ts = np.load(lob_ts_path)
        out["timestamps"] = {
            "path": str(lob_ts_path),
            "rows": int(len(ts)),
            "matches_tensor_rows": bool(len(ts) == len(arr)),
        }
        if len(ts):
            tss = pd.to_datetime(ts.astype(np.int64), unit="ns", utc=True, errors="coerce").tz_localize(None)
            out["timestamps"]["min"] = None if pd.isna(tss.min()) else str(tss.min())
            out["timestamps"]["max"] = None if pd.isna(tss.max()) else str(tss.max())
    return out


def _asof_lob_alignment(day_df: pd.DataFrame, lob_ts_path: str | None, tolerance: str = "10min") -> dict[str, Any]:
    if not lob_ts_path or not os.path.exists(lob_ts_path):
        return {"available": False, "reason": "lob timestamp file missing"}
    if "ts_event" not in day_df.columns:
        return {"available": False, "reason": "daytrade ts_event missing"}

    ts_raw = np.load(lob_ts_path)
    lob_ts = pd.to_datetime(ts_raw.astype(np.int64), unit="ns", utc=True, errors="coerce").tz_localize(None)
    lob = pd.DataFrame({"ts_event": pd.Series(lob_ts), "tensor_idx": np.arange(len(lob_ts), dtype=np.int32)})
    lob = lob.dropna(subset=["ts_event"]).sort_values("ts_event")
    rows = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(day_df["ts_event"], utc=True, errors="coerce").dt.tz_localize(None),
            "row_idx": np.arange(len(day_df), dtype=np.int32),
        }
    ).dropna(subset=["ts_event"]).sort_values("ts_event")
    if rows.empty or lob.empty:
        return {"available": True, "rows": int(len(day_df)), "matched_rows": 0, "coverage_ratio": 0.0}
    merged = pd.merge_asof(
        rows,
        lob,
        on="ts_event",
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )
    matched = merged["tensor_idx"].notna()
    return {
        "available": True,
        "rows": int(len(day_df)),
        "matched_rows": int(matched.sum()),
        "coverage_ratio": float(matched.mean()) if len(merged) else 0.0,
        "unique_tensors": int(merged.loc[matched, "tensor_idx"].nunique()),
        "tolerance": tolerance,
    }


def _stage1_to_daytrade_bridge(stage1_df: pd.DataFrame, day_df: pd.DataFrame, freq: str = "5min") -> dict[str, Any]:
    if "ts_event" not in stage1_df.columns or "ts_event" not in day_df.columns:
        return {"available": False, "reason": "ts_event missing"}
    st = stage1_df.copy()
    st["ts_event"] = pd.to_datetime(st["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    st = st.dropna(subset=["ts_event"])
    st["bar_key"] = st["ts_event"].dt.floor(freq)

    day = day_df.copy()
    day["bar_key"] = pd.to_datetime(day["ts_event"], utc=True, errors="coerce").dt.tz_localize(None).dt.floor(freq)
    bar_counts = st.groupby("bar_key").size()
    mapped_counts = day["bar_key"].map(bar_counts).fillna(0).astype(int)

    out = {
        "available": True,
        "freq": freq,
        "daytrade_bars": int(len(day)),
        "bars_with_stage1_rows": int((mapped_counts > 0).sum()),
        "coverage_ratio": float((mapped_counts > 0).mean()) if len(day) else 0.0,
        "stage1_rows_per_bar_mean": float(mapped_counts.mean()) if len(day) else 0.0,
        "stage1_rows_per_bar_p10": float(np.percentile(mapped_counts, 10)) if len(day) else 0.0,
        "stage1_rows_per_bar_p50": float(np.percentile(mapped_counts, 50)) if len(day) else 0.0,
        "stage1_rows_per_bar_p90": float(np.percentile(mapped_counts, 90)) if len(day) else 0.0,
    }
    return out


def _safe_corr(df: pd.DataFrame, a: str, b: str) -> float | None:
    if a not in df.columns or b not in df.columns:
        return None
    x = _num(df[a])
    y = _num(df[b])
    m = x.notna() & y.notna()
    if int(m.sum()) < 5:
        return None
    if float(x[m].std(ddof=0)) <= 1e-12 or float(y[m].std(ddof=0)) <= 1e-12:
        return None
    return float(x[m].corr(y[m], method="spearman"))


def _bridge_correlations(day_df: pd.DataFrame) -> dict[str, Any]:
    pairs = [
        ("lob_imbalance", "mbp_imbalance_peak"),
        ("lob_imbalance", "order_flow_imbalance"),
        ("order_flow_imbalance", "mbp_depth_accel"),
        ("bar_cvd_delta", "mbp_depth_accel"),
        ("mbp_roll_lob_coverage", "mbp_bar_coverage"),
    ]
    return {f"{a}__vs__{b}": _safe_corr(day_df, a, b) for a, b in pairs}


def _load_json(path: str | None) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _markdown(report: dict[str, Any]) -> str:
    stage1 = report["stage1"]
    day = report["daytrade"]
    tensor = report["lob_tensor"]
    lines = [
        "# Depth Layer Diagnostic",
        "",
        "## Executive Summary",
        f"- Stage1 depth features: present={stage1['depth_summary']['present']} dead={len(stage1['depth_summary']['dead'])} weak={len(stage1['depth_summary']['weak'])}",
        f"- Stage1 flow features: present={stage1['flow_summary']['present']} dead={len(stage1['flow_summary']['dead'])} weak={len(stage1['flow_summary']['weak'])}",
        f"- DayTrade bridge features: present={day['bridge_summary']['present']} dead={len(day['bridge_summary']['dead'])} weak={len(day['bridge_summary']['weak'])}",
        f"- LOB tensor available={tensor.get('available')} shape={tensor.get('shape')} collapsed_snapshot_ratio={tensor.get('collapsed_snapshot_ratio')}",
        f"- LOB timestamp alignment coverage={report['alignment'].get('coverage_ratio')}",
        f"- Stage1 rows feeding DayTrade bars coverage={report['stage1_to_daytrade'].get('coverage_ratio')}",
        "",
        "## How The Layers Work",
        "1. Stage1 reads raw MBP10 levels and builds microstructure depth features such as OBI, wall strength, wall distance, liquidity gaps, and liquidity density.",
        "2. Stage1 also builds event-rich LOB tensors from MBP snapshots around directional/train-event rows for the visual depth path.",
        "3. DayTrade aggregates MBO/order-flow into bars, then attaches MBP coverage and intrabar depth features (`mbp_*`).",
        "4. DayTrade builds rolling LOB tensors per bar: 50-bar history x 20 levels x 3 channels.",
        "5. At inference/backtest, EntryData maps each scored row to a raw LOB tensor, so the LOBTransformer receives raw depth while CatBoost/XGBoost/meta features carry order-flow and regime context.",
        "",
        "## Stage1 Depth Feature Summary",
        f"- Missing: {stage1['depth_summary']['missing']}",
        f"- Dead: {stage1['depth_summary']['dead']}",
        f"- Weak: {stage1['depth_summary']['weak']}",
        "",
        "## DayTrade Bridge Summary",
        f"- Missing: {day['bridge_summary']['missing']}",
        f"- Dead: {day['bridge_summary']['dead']}",
        f"- Weak: {day['bridge_summary']['weak']}",
        "",
        "## LOB Tensor Channels",
        f"- layout_guess={tensor.get('lob_layout_guess')} "
        f"(ch0 cross-level std mean={tensor.get('ch0_cross_level_std_mean')})",
        "",
        "Interpretation: `footprint_imbalance` is signed (buy−sell)/total trade mass per bar — "
        "a high zero-rate is normal when flow is balanced or when there are no trades in the slice; "
        "it is **not** the raw buy-side depth channel.",
        "",
    ]
    for name, stats in (tensor.get("channels") or {}).items():
        lines.append(
            f"- {name}: std={stats.get('std'):.6g} zero_rate={stats.get('zero_ratio'):.2%} "
            f"p01={stats.get('p01'):.4g} p99={stats.get('p99'):.4g}"
        )
    lines += [
        "",
        "## Bridge Correlations (Spearman)",
    ]
    for k, v in report["bridge_correlations"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Verdict",
    ]
    bad = []
    if stage1["depth_summary"]["dead"]:
        bad.append("Stage1 has dead depth features.")
    if day["bridge_summary"]["dead"]:
        bad.append("DayTrade has dead bridge features.")
    collapsed_ratio = tensor.get("collapsed_snapshot_ratio")
    collapsed_ratio = 1.0 if collapsed_ratio is None else float(collapsed_ratio)
    if not tensor.get("available") or collapsed_ratio > 0.01:
        bad.append("LOB tensor path may be collapsed or missing.")
    if float(report["alignment"].get("coverage_ratio") or 0.0) < 0.8:
        bad.append("LOB timestamp alignment coverage is weak.")
    if bad:
        lines.extend([f"- WARNING: {x}" for x in bad])
    else:
        lines.append("- PASS: Stage1 depth, DayTrade bridge, and LOB tensor alignment look coherent.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose Stage1/DayTrade depth layers")
    ap.add_argument("--stage1", required=True, help="Stage1 final parquet file or directory")
    ap.add_argument("--daytrade", required=True, help="DayTrade day_trading_features.parquet")
    ap.add_argument("--lob", required=True, help="DayTrade lob_tensors.npy")
    ap.add_argument("--lob_ts", required=True, help="DayTrade lob_tensor_timestamps.npy")
    ap.add_argument("--daytrade_manifest", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--freq", default="5min")
    ap.add_argument("--max_stage1_files", type=int, default=50)
    ap.add_argument("--print_report", action="store_true", help="Print the Markdown report to stdout")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stage1_df = _load_parquet(args.stage1, max_files=args.max_stage1_files)
    day_df = _load_parquet(args.daytrade, max_files=1)

    stage1_depth_profile = _feature_profile(stage1_df, STAGE1_DEPTH_FEATURES)
    stage1_flow_profile = _feature_profile(stage1_df, STAGE1_FLOW_FEATURES)
    day_bridge_profile = _feature_profile(day_df, DAYTRADE_BRIDGE_FEATURES)

    report = {
        "inputs": {
            "stage1": args.stage1,
            "daytrade": args.daytrade,
            "lob": args.lob,
            "lob_ts": args.lob_ts,
            "daytrade_manifest": args.daytrade_manifest,
        },
        "rows": {
            "stage1": int(len(stage1_df)),
            "daytrade": int(len(day_df)),
        },
        "stage1": {
            "depth_profile": stage1_depth_profile,
            "depth_summary": _summary_from_profile(stage1_depth_profile),
            "flow_profile": stage1_flow_profile,
            "flow_summary": _summary_from_profile(stage1_flow_profile),
        },
        "daytrade": {
            "bridge_profile": day_bridge_profile,
            "bridge_summary": _summary_from_profile(day_bridge_profile),
            "manifest": _load_json(args.daytrade_manifest),
        },
        "lob_tensor": _tensor_health(args.lob, args.lob_ts),
        "alignment": _asof_lob_alignment(day_df, args.lob_ts),
        "stage1_to_daytrade": _stage1_to_daytrade_bridge(stage1_df, day_df, freq=args.freq),
        "bridge_correlations": _bridge_correlations(day_df),
    }

    json_path = os.path.join(args.out_dir, "depth_layer_diagnostic.json")
    md_path = os.path.join(args.out_dir, "depth_layer_diagnostic.md")
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
