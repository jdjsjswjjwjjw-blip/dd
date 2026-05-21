"""
diagnose_stage1_daytrade_consistency.py
=======================================

Quantitative consistency checks between Stage1 trade-level output and the
DayTrade bar dataset built from it.
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


def _size_column(df: pd.DataFrame) -> str | None:
    for col in ("size", "qty", "volume"):
        if col in df.columns:
            return col
    return None


def _basic_time_profile(df: pd.DataFrame, ts_col: str = "ts_event") -> dict[str, Any]:
    if ts_col not in df.columns:
        return {"available": False, "reason": f"{ts_col} missing"}
    ts = _to_ts(df[ts_col])
    valid = ts.dropna()
    return {
        "available": True,
        "rows": int(len(df)),
        "invalid_ts": int(ts.isna().sum()),
        "duplicates": int(ts.duplicated().sum()),
        "monotonic_increasing": bool(valid.is_monotonic_increasing),
        "min": None if valid.empty else str(valid.min()),
        "max": None if valid.empty else str(valid.max()),
    }


def _aggregate_stage1_to_bars(stage1: pd.DataFrame, freq: str) -> pd.DataFrame:
    required = {"ts_event", "price"}
    missing = sorted(required - set(stage1.columns))
    if missing:
        raise KeyError(f"Stage1 data missing required columns: {missing}")

    size_col = _size_column(stage1)
    if size_col is None:
        raise KeyError("Stage1 data must include one of: size/qty/volume")

    work = stage1[["ts_event", "price", size_col]].copy()
    work["ts_event"] = _to_ts(work["ts_event"])
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work[size_col] = pd.to_numeric(work[size_col], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["ts_event", "price"]).sort_values("ts_event")
    work["bar_key"] = work["ts_event"].dt.floor(freq)

    grouped = work.groupby("bar_key", sort=True)
    bars = grouped["price"].agg(open="first", high="max", low="min", close="last")
    bars["volume"] = grouped[size_col].sum()
    bars["tick_count"] = grouped.size()
    bars = bars.reset_index().rename(columns={"bar_key": "ts_event"})
    return bars


def _prepare_daytrade_bars(daytrade: pd.DataFrame, freq: str) -> pd.DataFrame:
    required = {"ts_event", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(daytrade.columns))
    if missing:
        raise KeyError(f"DayTrade data missing required columns: {missing}")

    cols = ["ts_event", "open", "high", "low", "close", "volume"]
    if "tick_count" in daytrade.columns:
        cols.append("tick_count")
    out = daytrade[cols].copy()
    out["ts_event"] = _to_ts(out["ts_event"]).dt.floor(freq)
    for col in [c for c in cols if c != "ts_event"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "tick_count" not in out.columns:
        out["tick_count"] = np.nan
    return out.sort_values("ts_event").reset_index(drop=True)


def _diff_stats(delta: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(delta, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {"n": 0, "max_abs": None, "mean_abs": None, "p99_abs": None}
    ax = x.abs()
    return {
        "n": int(len(x)),
        "max_abs": float(ax.max()),
        "mean_abs": float(ax.mean()),
        "p99_abs": float(ax.quantile(0.99)),
    }


def _compare_bars(stage_bars: pd.DataFrame, day_bars: pd.DataFrame, price_tol: float, volume_tol: float) -> dict[str, Any]:
    merged = stage_bars.merge(
        day_bars,
        on="ts_event",
        how="outer",
        suffixes=("_stage1", "_daytrade"),
        indicator=True,
    ).sort_values("ts_event")

    common = merged["_merge"].eq("both")
    only_stage1 = merged["_merge"].eq("left_only")
    only_daytrade = merged["_merge"].eq("right_only")

    out: dict[str, Any] = {
        "stage1_bars": int(len(stage_bars)),
        "daytrade_bars": int(len(day_bars)),
        "common_bars": int(common.sum()),
        "only_stage1_bars": int(only_stage1.sum()),
        "only_daytrade_bars": int(only_daytrade.sum()),
        "coverage_stage1_to_daytrade": float(common.sum() / max(len(stage_bars), 1)),
        "coverage_daytrade_to_stage1": float(common.sum() / max(len(day_bars), 1)),
        "price_tolerance": float(price_tol),
        "volume_tolerance": float(volume_tol),
        "fields": {},
    }

    common_df = merged.loc[common].copy()
    price_ok = True
    for col in ("open", "high", "low", "close"):
        delta = common_df[f"{col}_stage1"] - common_df[f"{col}_daytrade"]
        st = _diff_stats(delta)
        st["within_tolerance"] = bool((pd.to_numeric(delta, errors="coerce").abs() <= price_tol).all())
        price_ok = price_ok and st["within_tolerance"]
        out["fields"][col] = st

    v_delta = common_df["volume_stage1"] - common_df["volume_daytrade"]
    v_stats = _diff_stats(v_delta)
    v_stats["within_tolerance"] = bool((pd.to_numeric(v_delta, errors="coerce").abs() <= volume_tol).all())
    out["fields"]["volume"] = v_stats

    if "tick_count_stage1" in common_df.columns and "tick_count_daytrade" in common_df.columns:
        t_delta = common_df["tick_count_stage1"] - common_df["tick_count_daytrade"]
        t_stats = _diff_stats(t_delta)
        t_stats["exact_match"] = bool((pd.to_numeric(t_delta, errors="coerce").fillna(0).abs() == 0).all())
        out["fields"]["tick_count"] = t_stats

    out["pass"] = bool(
        out["only_daytrade_bars"] == 0
        and out["coverage_daytrade_to_stage1"] >= 0.999
        and price_ok
        and bool(out["fields"]["volume"]["within_tolerance"])
        and bool(out["fields"].get("tick_count", {}).get("exact_match", True))
    )
    return out


def _markdown(report: dict[str, Any]) -> str:
    cmp = report["bar_comparison"]
    lines = [
        "# Stage1 vs DayTrade Consistency",
        "",
        "## Executive Summary",
        f"- Verdict: {'PASS' if cmp['pass'] else 'WARNING'}",
        f"- Common bars: {cmp['common_bars']}/{cmp['daytrade_bars']} daytrade bars",
        f"- Stage1->DayTrade coverage: {cmp['coverage_stage1_to_daytrade']:.2%}",
        f"- DayTrade->Stage1 coverage: {cmp['coverage_daytrade_to_stage1']:.2%}",
        f"- Extra bars: stage1_only={cmp['only_stage1_bars']} daytrade_only={cmp['only_daytrade_bars']}",
        "",
        "## Time Ranges",
        f"- Stage1: {report['stage1_time'].get('min')} -> {report['stage1_time'].get('max')}",
        f"- DayTrade: {report['daytrade_time'].get('min')} -> {report['daytrade_time'].get('max')}",
        f"- Stage1 monotonic: {report['stage1_time'].get('monotonic_increasing')} | duplicates={report['stage1_time'].get('duplicates')}",
        f"- DayTrade monotonic: {report['daytrade_time'].get('monotonic_increasing')} | duplicates={report['daytrade_time'].get('duplicates')}",
        "",
        "## OHLCV Differences",
    ]
    for name, stats in cmp["fields"].items():
        status = stats.get("within_tolerance", stats.get("exact_match"))
        lines.append(
            f"- {name}: max_abs={stats.get('max_abs')} mean_abs={stats.get('mean_abs')} "
            f"p99_abs={stats.get('p99_abs')} ok={status}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Stage1 final rows aggregated to bars with DayTrade bars.")
    ap.add_argument("--stage1", required=True, help="Stage1 final parquet file or directory")
    ap.add_argument("--daytrade", required=True, help="DayTrade day_trading_features.parquet")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--freq", default="5min")
    ap.add_argument("--price_tol", type=float, default=1e-9)
    ap.add_argument("--volume_tol", type=float, default=1e-9)
    ap.add_argument("--print_report", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stage1 = _load_parquet(args.stage1)
    daytrade = _load_parquet(args.daytrade)

    stage_bars = _aggregate_stage1_to_bars(stage1, args.freq)
    day_bars = _prepare_daytrade_bars(daytrade, args.freq)

    report = {
        "inputs": {
            "stage1": args.stage1,
            "daytrade": args.daytrade,
            "freq": args.freq,
        },
        "stage1_time": _basic_time_profile(stage1),
        "daytrade_time": _basic_time_profile(daytrade),
        "bar_comparison": _compare_bars(stage_bars, day_bars, args.price_tol, args.volume_tol),
    }

    json_path = os.path.join(args.out_dir, "stage1_daytrade_consistency.json")
    md_path = os.path.join(args.out_dir, "stage1_daytrade_consistency.md")
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
