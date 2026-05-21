"""
diagnose_lob_entry_quality.py
=============================

LOB/Entry quality smoke test for DayTrade artifacts.
Checks tensor integrity, timestamp alignment, simple LOB/return relationships,
session coverage, and MBP feature quality.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _rho(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 8:
        return None
    if float(np.std(a[m])) <= 1e-12 or float(np.std(b[m])) <= 1e-12:
        return None
    val = spearmanr(a[m], b[m]).statistic
    return None if pd.isna(val) else float(val)


def _channel_report(name: str, arr: np.ndarray) -> str:
    finite = np.isfinite(arr)
    return (
        f"{name:20s} std={np.nanstd(arr):.6f} | zero={(arr == 0).mean():.1%} | "
        f"finite={finite.mean():.1%} | p01={np.nanpercentile(arr, 1):.4g} | "
        f"p99={np.nanpercentile(arr, 99):.4g}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose DayTrade LOB tensors and Entry feature quality.")
    ap.add_argument("--daytrade_dir", required=True)
    ap.add_argument("--freq", default="5min")
    args = ap.parse_args()

    base = args.daytrade_dir
    tensors_path = os.path.join(base, "lob_tensors.npy")
    ts_path = os.path.join(base, "lob_tensor_timestamps.npy")
    data_path = os.path.join(base, "day_trading_features.parquet")

    tensors = np.load(tensors_path)
    ts = np.load(ts_path)
    df = pd.read_parquet(data_path)

    print("=== TEST 1: LOB Tensor Integrity ===")
    print(f"Tensors: {tensors.shape} dtype={tensors.dtype}")
    print(f"Rows match parquet: {tensors.shape[0] == len(df)} ({tensors.shape[0]} vs {len(df)})")
    print(_channel_report("Channel 0 depth", tensors[..., 0]))
    print(_channel_report("Channel 1 buy", tensors[..., 1]))
    print(_channel_report("Channel 2 sell", tensors[..., 2]))
    row_std = np.nanstd(tensors, axis=(1, 2, 3))
    print(f"Collapsed rows: {(row_std <= 1e-8).sum()}/{len(row_std)} ({(row_std <= 1e-8).mean():.1%})")

    print("\n=== TEST 2: Timestamp Alignment ===")
    ts_dt = pd.to_datetime(ts.astype(np.int64), unit="ns", utc=True, errors="coerce").tz_localize(None)
    bar_ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    exact = bar_ts.isin(set(ts_dt)).sum()
    left = pd.DataFrame({"ts_event": bar_ts}).sort_values("ts_event")
    right = pd.DataFrame({"ts_event": ts_dt, "tensor_idx": np.arange(len(ts_dt))}).sort_values("ts_event")
    asof = pd.merge_asof(left, right, on="ts_event", direction="backward", tolerance=pd.Timedelta(args.freq))
    asof_match = int(asof["tensor_idx"].notna().sum())
    print(f"Exact bars matched: {exact}/{len(bar_ts)} ({exact / max(len(bar_ts), 1):.1%})")
    print(f"Asof bars matched : {asof_match}/{len(bar_ts)} ({asof_match / max(len(bar_ts), 1):.1%})")
    print(f"Tensor ts range   : {ts_dt.min()} -> {ts_dt.max()}")
    print(f"Bar ts range      : {bar_ts.min()} -> {bar_ts.max()}")

    print("\n=== TEST 3: LOB vs Price/Label Correlation ===")
    depth_mean = tensors[..., 0].mean(axis=(1, 2))
    buy_mean = tensors[..., 1].mean(axis=(1, 2))
    sell_mean = tensors[..., 2].mean(axis=(1, 2))
    net_flow = buy_mean - sell_mean
    close = pd.to_numeric(df["close"], errors="coerce")
    ret_now = close.pct_change().fillna(0.0).to_numpy()
    ret_next = close.pct_change().shift(-1).fillna(0.0).to_numpy()
    print(f"depth vs same-bar return:     rho={_rho(depth_mean, ret_now)}")
    print(f"buy   vs same-bar return:     rho={_rho(buy_mean, ret_now)}")
    print(f"sell  vs same-bar return:     rho={_rho(sell_mean, ret_now)}")
    print(f"buy-sell vs same-bar return:  rho={_rho(net_flow, ret_now)}")
    print(f"buy-sell vs next-bar return:  rho={_rho(net_flow, ret_next)}")
    if "bias_label" in df.columns:
        y_dir = pd.to_numeric(df["bias_label"], errors="coerce").map({0: 1, 1: -1, 2: 0}).fillna(0).to_numpy()
        print(f"buy-sell vs label direction:  rho={_rho(net_flow, y_dir)}")

    print("\n=== TEST 4: Session Coverage ===")
    hours = bar_ts.dt.hour.value_counts().sort_index()
    print(hours.to_string())
    for col in ("is_london", "is_overlap", "is_ny"):
        if col in df.columns:
            print(f"{col:12s} mean: {pd.to_numeric(df[col], errors='coerce').fillna(0).mean():.1%}")

    print("\n=== TEST 5: MBP Depth Quality ===")
    mbp_cols = [c for c in df.columns if c.startswith("mbp_")]
    print(f"MBP columns: {len(mbp_cols)}")
    for col in mbp_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().sum() == 0:
            print(f"  {col}: all_null")
            continue
        print(f"  {col}: std={x.std(ddof=0):.6f} | zero={(x.fillna(0) == 0).mean():.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
