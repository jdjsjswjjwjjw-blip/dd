#!/usr/bin/env python3
"""
Pre-training checks for prepare_day_trading.py output.

Usage:
  py -3.13 verify_day_trading_dataset.py --data pipeline_.../day_trading_features.parquet
  py -3.13 verify_day_trading_dataset.py --data ... --lob .../lob_tensors.npy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Columns train_v19.load_training_csv hard-requires (minimal gate)
REQUIRED_FOR_TRAIN = (
    "ts_event",
    "label_end_ts",
    "bias_label",
    "soft_label",
    "label_confidence",
    "soft_sample_weight",
    "event_flag",
    "train_event_flag",
    "signal_quality",
    "forward_return",
)

# prepare_day_trading لم يعد يصدّر raw__ (كان مطابقاً للعمود الأساسي). ملفات قديمة قد تحوي 31 عموداً.
EXPECTED_RAW_PREFIX = "raw__"
EXPECTED_RAW_LEGACY_COUNT = 31

try:
    from prepare_day_trading import ORIGINAL_FEATURES as DAYTRADE_ORIGINAL_FEATURES
except ImportError:
    DAYTRADE_ORIGINAL_FEATURES = ()

# أعمدة MBP intrabar الاختيارية — قد لا تُصدَّر في المخطط المختصر (ORIGINAL_FEATURES فقط)
INTRABAR_MBP_OPTIONAL = (
    "mbp_spread_max",
    "mbp_spread_mean",
    "mbp_spread_std",
    "mbp_depth_bid_max",
    "mbp_depth_ask_max",
    "mbp_depth_sum_max",
    "mbp_imbalance_peak",
    "mbp_imbalance_direction_pct",
    "mbp_microprice_dev_max",
    "mbp_wall_bid_peak",
    "mbp_wall_ask_peak",
    "mbp_depth_shock_flag",
)


def _series_stats(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce")
    finite = np.isfinite(x.to_numpy(dtype=np.float64, na_value=np.nan))
    n = len(x)
    n_null = int(x.isna().sum())
    n_fin = int(finite.sum())
    arr = x.to_numpy(dtype=np.float64, copy=False)
    arr = arr[finite]
    if arr.size == 0:
        return {"n": n, "null_pct": n_null / max(n, 1), "zero_pct": 1.0, "min": None, "max": None, "mean": None}
    zero_pct = float(np.mean(np.abs(arr) < 1e-12)) if arr.size else 1.0
    return {
        "n": n,
        "null_pct": n_null / max(n, 1),
        "zero_pct": zero_pct,
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Verify day_trading_features.parquet (+ optional LOB) before train_v19.")
    p.add_argument("--data", required=True, help="Path to day_trading_features.parquet")
    p.add_argument("--lob", default=None, help="Path to lob_tensors.npy (optional)")
    p.add_argument("--lob_ts", default=None, help="Path to lob_tensor_timestamps.npy (default: next to --lob)")
    args = p.parse_args()

    data_path = os.path.abspath(args.data)
    if not os.path.isfile(data_path):
        print(f"❌ Missing parquet: {data_path}")
        return 1

    print(f"📥 Loading: {data_path}")
    df = pd.read_parquet(data_path)
    n, ncol = len(df), len(df.columns)
    print(f"   Rows: {n:,} | Columns: {ncol}")

    ok = True

    missing = [c for c in REQUIRED_FOR_TRAIN if c not in df.columns]
    if missing:
        ok = False
        print(f"❌ Missing required columns for train_v19: {missing}")
    else:
        print("✅ Required train_v19 columns present")

    raw_cols = [c for c in df.columns if c.startswith(EXPECTED_RAW_PREFIX)]
    if not raw_cols:
        print(
            f"✅ {EXPECTED_RAW_PREFIX}*: absent (expected) — advisor cols unscaled here; "
            "same std vs duplicate raw__ was wasted storage."
        )
    elif len(raw_cols) >= EXPECTED_RAW_LEGACY_COUNT:
        print(
            f"ℹ️ {EXPECTED_RAW_PREFIX}*: {len(raw_cols)} columns (legacy export); "
            "identical std vs base stats is normal — regenerate without raw__ to drop duplication."
        )
    else:
        print(
            f"⚠️ {EXPECTED_RAW_PREFIX}*: partial set ({len(raw_cols)} cols) — unusual; "
            f"prefer 0 or full legacy {EXPECTED_RAW_LEGACY_COUNT}."
        )

    if "ts_event" in df.columns:
        ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        if ts.isna().any():
            ok = False
            print(f"❌ ts_event has {int(ts.isna().sum())} invalid timestamps")
        dup = int(ts.duplicated().sum())
        if dup:
            ok = False
            print(f"❌ Duplicate ts_event rows: {dup}")
        if not ts.is_monotonic_increasing:
            print("⚠️ ts_event not strictly increasing — train_v19 will sort; better to sort parquet once.")
        else:
            print("✅ ts_event: no duplicates, monotonic increasing")

    if "bias_label" in df.columns:
        vc = df["bias_label"].value_counts().sort_index()
        print(f"✅ bias_label distribution:\n{vc.to_string()}")

    def _check_group(name: str, cols: tuple[str, ...], *, hard_fail: bool = False) -> None:
        nonlocal ok
        if not cols:
            print(f"ℹ️ {name}: (skipped — empty column list)")
            return
        present = [c for c in cols if c in df.columns]
        absent = [c for c in cols if c not in df.columns]
        if absent:
            msg = f"⚠️ {name}: missing {len(absent)}/{len(cols)} → {absent[:8]}{'...' if len(absent) > 8 else ''}"
            print(msg)
            if hard_fail:
                ok = False
            return
        print(f"✅ {name}: all {len(cols)} columns present")
        bad = []
        sparse_ok = (
            "mbp_depth_shock_flag",
            "spoof_burst_flag",
            "liquidity_gaps",
            "is_london",
            "is_overlap",
            "is_ny",
        )
        for c in present:
            st = _series_stats(df[c])
            if st["null_pct"] > 0.99:
                bad.append(f"{c}(all_null)")
            elif st["zero_pct"] > 0.999 and c not in sparse_ok:
                bad.append(f"{c}(~all_zero)")
        if bad:
            print(f"⚠️ {name}: weak/degenerate columns: {bad[:8]}{'...' if len(bad) > 8 else ''}")
        else:
            print(f"   (sanity: no column ~all-null; sparse/binary cols exempt from zero check)")

    if DAYTRADE_ORIGINAL_FEATURES:
        _check_group("ORIGINAL_FEATURES", DAYTRADE_ORIGINAL_FEATURES, hard_fail=False)
    else:
        print("ℹ️ ORIGINAL_FEATURES: import failed — skipped")

    _check_group("MBP intrabar (optional export)", INTRABAR_MBP_OPTIONAL, hard_fail=False)

    vwap_roll = ("vwap_roll_1h", "vwap_dist_1h_roll")
    _check_group("Rolling VWAP (~1h causal, optional)", vwap_roll, hard_fail=False)

    lob_path = args.lob
    if lob_path:
        lob_path = os.path.abspath(lob_path)
        ts_path = args.lob_ts
        if ts_path is None:
            ts_path = os.path.join(os.path.dirname(lob_path), "lob_tensor_timestamps.npy")
        ts_path = os.path.abspath(ts_path)

        if not os.path.isfile(lob_path):
            ok = False
            print(f"❌ LOB file missing: {lob_path}")
        else:
            tensors = np.load(lob_path, mmap_mode="r")
            print(f"✅ LOB tensors: shape={tensors.shape} dtype={tensors.dtype}")
            if tensors.shape[0] != n:
                ok = False
                print(f"❌ LOB row count {tensors.shape[0]} != parquet rows {n}")
            if len(tensors.shape) != 4:
                ok = False
                print(f"❌ Expected 4D LOB (N, 50, 20, 3); got {len(tensors.shape)}D")
            arr = np.asarray(tensors[: min(4096, tensors.shape[0])], dtype=np.float64)
            if not np.isfinite(arr).all():
                ok = False
                print("❌ LOB sample contains NaN/Inf")
            else:
                print(f"   LOB sample finite: min={arr.min():.6g} max={arr.max():.6g}")

        if not os.path.isfile(ts_path):
            ok = False
            print(f"❌ LOB timestamps missing: {ts_path}")
        else:
            lob_ts = np.load(ts_path)
            print(f"✅ LOB timestamps: len={lob_ts.shape[0]} dtype={lob_ts.dtype}")
            if lob_ts.shape[0] != n:
                ok = False
                print(f"❌ LOB ts length {lob_ts.shape[0]} != parquet rows {n}")

    if ok:
        print("\n✅ Pre-training verification passed.")
        return 0
    print("\n❌ Pre-training verification failed — fix issues above before train_v19.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
