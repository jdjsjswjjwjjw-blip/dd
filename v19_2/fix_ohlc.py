"""
fix_ohlc.py — إصلاح صارم لـ OHLC outliers
══════════════════════════════════════════════════════════════════════

المنهج (مهم):
  ① close = المرجع الأنظف (آخر تيك في الشمعة)
  ② نظّف close أولاً (rolling Z-score على close نفسه)
  ③ open: نفس المنطق
  ④ high/low: نعيد بناءهما من |close,open| + tolerance
     لأن أي قيمة شاذة في high/low تأتي من تيك خاطئ
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

MARKET_RANGES = {
    "6B":  (1.15, 1.45),
    "6E":  (0.95, 1.25),
    "ES":  (3000, 6500),
    "NQ":  (10000, 25000),
}
MAX_BAR_RANGE_PCT = {"6B": 0.0080, "6E": 0.0080, "ES": 0.005, "NQ": 0.008}


def fix_ohlc(df, symbol="6B"):
    df = df.copy()
    lo, hi = MARKET_RANGES.get(symbol, (0, 1e9))
    max_pct = MAX_BAR_RANGE_PCT.get(symbol, 0.01)
    stats = {"fixes": {}}

    # ① clip strict
    for col in ("open","high","low","close"):
        s = pd.to_numeric(df[col], errors="coerce")
        out = (s < lo) | (s > hi)
        n = int(out.sum())
        if n > 0:
            df.loc[out, col] = np.nan
            stats["fixes"][f"{col}_out_range"] = n

    # ② close نظافة (المرجع) — دورتان للتقارب
    df["close"] = df["close"].interpolate("linear", limit=5).ffill().bfill()
    for pass_n in (1, 2):
        rmed = df["close"].rolling(20, min_periods=5).median().bfill().ffill()
        rstd = df["close"].rolling(20, min_periods=5).std().bfill().ffill().clip(lower=0.0001)
        z = (df["close"] - rmed) / rstd
        bad = z.abs() > 3.5  # أصرم من 5
        if bad.any():
            df.loc[bad, "close"] = rmed[bad]
            stats["fixes"][f"close_z_pass{pass_n}"] = int(bad.sum())

    # ③ open
    df["open"] = df["open"].interpolate("linear", limit=5).ffill().bfill()
    z_o = (df["open"] - rmed) / rstd
    bad_o = z_o.abs() > 5
    if bad_o.any():
        df.loc[bad_o, "open"] = df.loc[bad_o, "close"]
        stats["fixes"]["open_z_outliers"] = int(bad_o.sum())

    # ④ high/low: rebuild from open+close
    o = df["open"].astype(float)
    c = df["close"].astype(float)
    body_hi = np.maximum(o, c)
    body_lo = np.minimum(o, c)
    max_dev = c * max_pct

    h_raw = pd.to_numeric(df["high"], errors="coerce")
    h_max_ok = body_hi + max_dev
    h_invalid = (h_raw.isna() | (h_raw < body_hi) | (h_raw > h_max_ok)
                 | (h_raw < lo) | (h_raw > hi))
    if h_invalid.any():
        df.loc[h_invalid, "high"] = body_hi[h_invalid] + max_dev[h_invalid] * 0.1
        stats["fixes"]["high_invalid"] = int(h_invalid.sum())

    l_raw = pd.to_numeric(df["low"], errors="coerce")
    l_min_ok = body_lo - max_dev
    l_invalid = (l_raw.isna() | (l_raw > body_lo) | (l_raw < l_min_ok)
                 | (l_raw < lo) | (l_raw > hi))
    if l_invalid.any():
        df.loc[l_invalid, "low"] = body_lo[l_invalid] - max_dev[l_invalid] * 0.1
        stats["fixes"]["low_invalid"] = int(l_invalid.sum())

    # ⑤ enforce: high >= max(o,c), low <= min(o,c)
    df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
    df["low"]  = np.minimum(df["low"],  np.minimum(df["open"], df["close"]))

    # ⑥ recompute ATR
    if "bar_range" in df.columns:
        df["bar_range"] = df["high"] - df["low"]
    h2, l2 = df["high"], df["low"]
    c_prev = df["close"].shift(1)
    tr = pd.concat([(h2-l2).abs(), (h2-c_prev).abs(), (l2-c_prev).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14, min_periods=1).mean()

    # ⑦ recompute PDH/PDL/PD_50
    if any(c in df.columns for c in ("pdh","PDH","pdl","PDL")):
        df["__day"] = pd.to_datetime(df["ts_event"]).dt.date
        daily = df.groupby("__day").agg(dh=("high","max"), dl=("low","min")).reset_index()
        daily["pdh_n"] = daily["dh"].shift(1)
        daily["pdl_n"] = daily["dl"].shift(1)
        df = df.merge(daily[["__day","pdh_n","pdl_n"]], on="__day", how="left")
        for src, dst in (("pdh_n","pdh"), ("pdh_n","PDH"), ("pdl_n","pdl"), ("pdl_n","PDL")):
            if dst in df.columns:
                df[dst] = df[src]
        for dst in ("pd_50", "PD_50"):
            if dst in df.columns:
                df[dst] = (df["pdh_n"] + df["pdl_n"]) / 2
        if "dist_to_pdh" in df.columns:
            atr = df["atr_14"].clip(lower=1e-9)
            df["dist_to_pdh"] = ((df["close"] - df["pdh_n"]) / atr).clip(-20, 20)
        df = df.drop(columns=["__day", "pdh_n", "pdl_n"])

    return df, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--symbol", default="6B")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    print(f"📥 {len(df):,} × {len(df.columns)}")
    print(f"قبل: open=[{df.open.min():.4f},{df.open.max():.4f}] "
          f"high=[{df.high.min():.4f},{df.high.max():.4f}] "
          f"low=[{df.low.min():.4f},{df.low.max():.4f}] "
          f"close=[{df.close.min():.4f},{df.close.max():.4f}]")

    fixed, st = fix_ohlc(df, args.symbol)

    print(f"\nبعد: open=[{fixed.open.min():.4f},{fixed.open.max():.4f}] "
          f"high=[{fixed.high.min():.4f},{fixed.high.max():.4f}] "
          f"low=[{fixed.low.min():.4f},{fixed.low.max():.4f}] "
          f"close=[{fixed.close.min():.4f},{fixed.close.max():.4f}]")

    print(f"\nfixes:")
    for k, v in st["fixes"].items():
        print(f"  {k}: {v}")

    fixed["bar_range"] = fixed["high"] - fixed["low"]
    print(f"\nbar_range: median={fixed.bar_range.median()*10000:.1f}p, "
          f"max={fixed.bar_range.max()*10000:.1f}p")
    print(f"ATR_14 median: {fixed.atr_14.median()*10000:.1f}p")
    if "pdh" in fixed.columns:
        print(f"PDH: [{fixed.pdh.min():.4f}, {fixed.pdh.max():.4f}]")
        print(f"PDL: [{fixed.pdl.min():.4f}, {fixed.pdl.max():.4f}]")
        sp = (fixed.pdh - fixed.pdl).median()
        print(f"PDH-PDL median: {sp*10000:.0f}p")

    out = args.output or args.input.replace(".parquet", "_FIXED.parquet")
    fixed.to_parquet(out, index=False)
    print(f"💾 {out}")


if __name__ == "__main__":
    main()
