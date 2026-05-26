"""
data_pipeline.py — طبقة تجهيز الداتا قبل المحاكيات وكشف الـ Edge
════════════════════════════════════════════════════════════════════

الهدف:
  تحويل MBO خام + MBP عمق → داتا هجينة نظيفة جاهزة للمحاكيات

المراحل:
  ① تنظيف التيكات   : أسعار خاطئة، تكرار، BBO متقاطع
  ② تجميع ذكي       : شمعات دقيقة مع تصحيح aggregation المكسور
  ③ عمق هجين        : دمج MBP snapshot مع كل شمعة
  ④ تنظيف الشمعات   : close outliers، clip الـ features
  ⑤ normalization    : كل feature بطريقته الصح (rank/zscore/ratio)
  ⑥ fwd_ret_clean    : هدف صح لـ IC
  ⑦ تصدير           : parquet جاهز للمحاكيات

الإصلاحات عن prepare_day_trading.py الأصلي:
  - hawkes: delta داخل الشمعة بدل max (كان cumulative معكوس)
  - kyle_lambda: mean بدل max
  - absorption_intensity: clip(0, 5) بدل clip(0, 1)
  - inter_event_time: clip(upper=3600) — max كان 27 يوم!
  - close outliers: ffill بدل إبقاء القفزات
  - dist_to_pdh: clip(-20, 20) — كان ±13,587
  - distance_to_wall: clip(0, 50)
  - liquidity_density: clip عند P99
  - MBP depth: يُدمج مع كل شمعة مباشرة

مثال تشغيل:
  python data_pipeline.py \\
    --mbo  data/6BM5_mbo.parquet \\
    --mbp  data/6BM5_mbp.parquet \\
    --output pipeline_output/ \\
    --freq 5min \\
    --horizon 6

  # من parquet موجود مع تنظيف فقط:
  python data_pipeline.py \\
    --from-parquet pipeline_day_trading/features/day_trading_features.parquet \\
    --output pipeline_output/ \\
    --horizon 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — ثوابت وإعدادات
# ══════════════════════════════════════════════════════════════════

# نوافذ الجلسات UTC
SESSIONS = {
    "asia":    ("00:00", "07:00"),
    "london":  ("07:00", "12:00"),
    "overlap": ("13:30", "16:00"),
    "ny":      ("13:30", "20:00"),
}

# Features التي تحتاج normalization خاص
CLIP_RULES: dict[str, tuple[float, float]] = {
    "dist_to_pdh":       (-20.0,  20.0),
    "distance_to_wall":  (0.0,    50.0),
    "inter_event_time":  (0.0,   3600.0),
    "vwap_z_score":      (-10.0,  10.0),
    "gap_size":          (0.0,    10.0),
}

# Features تُحسب بـ percentile rank (مستقل عن الـ scale)
RANK_FEATURES = [
    "absorption_intensity",
    "volume_burst",
    "hawkes_intensity",
    "kyle_lambda",
    "liquidity_density",
]

# Features تُحسب بـ rolling z-score
ZSCORE_FEATURES = [
    "cvd",
    "vnet",
    "session_cvd",
    "cvd_momentum",
]

# عمق السوق — أعمدة MBP
MBP_LEVELS = 10
MBP_BID_PX = [f"bid_px_{i:02d}" for i in range(MBP_LEVELS)]
MBP_ASK_PX = [f"ask_px_{i:02d}" for i in range(MBP_LEVELS)]
MBP_BID_SZ = [f"bid_sz_{i:02d}" for i in range(MBP_LEVELS)]
MBP_ASK_SZ = [f"ask_sz_{i:02d}" for i in range(MBP_LEVELS)]


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — تنظيف التيكات الخام
# ══════════════════════════════════════════════════════════════════

def sanitize_mbo_ticks(df: pd.DataFrame) -> pd.DataFrame:
    """
    تنظيف تيكات MBO الخام:
    ① فرز زمني
    ② حذف صفوف بدون ts_event أو سعر صالح
    ③ حذف التكرار الكامل
    ④ حذف تكرار المفتاح (order_id + symbol إن وُجدا)
    ⑤ تنظيف أسعار = صفر أو سالبة
    """
    out = df.copy()
    n0 = len(out)

    # ① فرز
    out["ts_event"] = pd.to_datetime(out["ts_event"], errors="coerce")
    out = out.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)

    # ② أسعار صالحة
    price_col = next((c for c in ("price", "px") if c in out.columns), None)
    if price_col:
        out["price"] = pd.to_numeric(out[price_col], errors="coerce")
        out = out[out["price"] > 0].copy()

    # ③ تكرار كامل
    out = out.drop_duplicates().reset_index(drop=True)

    # ④ تكرار مفتاح
    key_cols = [c for c in ("order_id", "sequence", "symbol") if c in out.columns]
    if len(key_cols) >= 2:
        out = out.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)

    # ⑤ حجم صالح
    size_col = next((c for c in ("size", "qty", "volume") if c in out.columns), None)
    if size_col:
        out[size_col] = pd.to_numeric(out[size_col], errors="coerce").fillna(0.0).clip(lower=0)

    n1 = len(out)
    print(f"  🧹 MBO ticks: {n0:,} → {n1:,} (حذف {n0-n1:,} سطر)")
    return out


def sanitize_mbp_ticks(df: pd.DataFrame) -> pd.DataFrame:
    """
    تنظيف snapshots MBP:
    ① فرز زمني
    ② حذف BBO متقاطع (ask < bid)
    ③ حذف snapshots بأسعار صفر
    """
    out = df.copy()
    n0 = len(out)

    out["ts_event"] = pd.to_datetime(out["ts_event"], errors="coerce")
    out = out.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)

    # BBO متقاطع
    if "bid_px_00" in out.columns and "ask_px_00" in out.columns:
        bid0 = pd.to_numeric(out["bid_px_00"], errors="coerce").fillna(0)
        ask0 = pd.to_numeric(out["ask_px_00"], errors="coerce").fillna(0)
        crossed = (ask0 > 0) & (bid0 > 0) & (ask0 < bid0)
        out = out[~crossed].copy()

    # snapshots بسعر صفر
    if "bid_px_00" in out.columns:
        out = out[pd.to_numeric(out["bid_px_00"], errors="coerce").fillna(0) > 0].copy()

    n1 = len(out)
    print(f"  🧹 MBP snapshots: {n0:,} → {n1:,} (حذف {n0-n1:,} سطر)")
    return out.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — تجميع ذكي MBO → شمعات (الإصلاحات مضمّنة)
# ══════════════════════════════════════════════════════════════════

def aggregate_ticks_to_bars(df_mbo: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """
    التجميع الصح — يصلح المشاكل المكتشفة في prepare_day_trading:

    hawkes_intensity:
      كان: max (قيمة cumulative = معكوس)
      صار: delta = last - first داخل الشمعة

    kyle_lambda:
      كان: max
      صار: mean (متوسط استجابة السعر للحجم)

    absorption_intensity:
      كان: max بدون clip منطقي
      صار: max مع clip(0, 20) — ثم rank لاحقاً

    inter_event_time:
      كان: mean بدون حد (max=2.3M ثانية)
      صار: mean مع clip(upper=3600)
    """
    df = df_mbo.copy()
    df["ts_event"] = pd.to_datetime(df["ts_event"])
    df = df.set_index("ts_event").sort_index()

    size_col = next((c for c in ("size", "qty", "volume") if c in df.columns), "size")
    size_s = pd.to_numeric(df[size_col], errors="coerce").fillna(0.0)

    action_s = df["action"].astype(str).str.upper() if "action" in df.columns \
               else pd.Series("T", index=df.index)
    side_s = df["side"].astype(str).str.upper() if "side" in df.columns \
             else pd.Series("", index=df.index)

    TRADE_ACTIONS = {"T", "F", "TRADE", "E", "0"}
    BUY_SIDES  = {"A", "ASK", "BUY", "BOT"}
    SELL_SIDES = {"B", "BID", "S", "SELL"}

    # ── OHLCV ──
    bars = df["price"].resample(freq).agg(open="first", high="max", low="min", close="last")
    bars["volume"] = size_s.resample(freq).sum()
    bars = bars[bars["volume"] > 0].copy()

    def _rs(col: str, how: str, default: float = 0.0) -> pd.Series:
        if col not in df.columns:
            return pd.Series(default, index=bars.index, dtype=np.float64)
        s = pd.to_numeric(df[col], errors="coerce")
        if how == "mean":   r = s.resample(freq).mean()
        elif how == "max":  r = s.resample(freq).max()
        elif how == "sum":  r = s.resample(freq).sum()
        elif how == "last": r = s.resample(freq).last()
        elif how == "first":r = s.resample(freq).first()
        elif how == "std":  r = s.resample(freq).std()
        elif how == "delta":
            # الفرق بين آخر وأول قيمة — للـ hawkes
            r = s.resample(freq).agg(lambda x: float(x.iloc[-1] - x.iloc[0]) if len(x) > 1 else 0.0)
        else:
            r = s.resample(freq).mean()
        return pd.to_numeric(r, errors="coerce").reindex(bars.index).fillna(default).astype(np.float64)

    # ── CVD ──
    is_trade = action_s.isin(TRADE_ACTIONS)
    is_buy   = side_s.isin(BUY_SIDES)
    is_sell  = side_s.isin(SELL_SIDES)
    signed   = np.zeros(len(df), dtype=np.float64)
    signed[(is_trade & is_buy).to_numpy()]  =  size_s[(is_trade & is_buy)].to_numpy()
    signed[(is_trade & is_sell).to_numpy()] = -size_s[(is_trade & is_sell)].to_numpy()
    tick_cvd = pd.Series(signed, index=df.index).cumsum()

    bars["cvd"]          = tick_cvd.resample(freq).last().reindex(bars.index).ffill().fillna(0)
    bars["bar_cvd_delta"]= tick_cvd.resample(freq).agg(
        lambda x: float(x.iloc[-1] - x.iloc[0]) if len(x) > 1 else 0.0
    ).reindex(bars.index).fillna(0)

    # session CVD (يُعاد كل يوم)
    signed_step  = tick_cvd.diff().fillna(tick_cvd)
    session_cvd  = signed_step.groupby(signed_step.index.normalize()).cumsum()
    bars["session_cvd"] = session_cvd.resample(freq).last().reindex(bars.index).fillna(0)

    # ── Hawkes: DELTA ← إصلاح ──
    if "hawkes_intensity" in df.columns:
        bars["hawkes_intensity"] = _rs("hawkes_intensity", "delta", 0.0).clip(lower=0)
    else:
        bars["hawkes_intensity"] = 0.0

    # ── Kyle Lambda: MEAN ← إصلاح ──
    bars["kyle_lambda"] = _rs("kyle_lambda", "mean", 0.0)

    # ── Absorption: MAX مع clip منطقي ← إصلاح ──
    bars["absorption_intensity"] = _rs("absorption_intensity", "max", 0.0).clip(0, 20)

    # ── inter_event_time: MEAN + clip ← إصلاح ──
    bars["inter_event_time"] = _rs("inter_event_time", "mean", 0.0).clip(0, 3600)

    # ── باقي الـ features ──
    bars["cancel_ratio"]       = _rs("cancel_ratio", "max", 0.0).clip(0, 1)
    bars["spoofing_ratio"]     = _rs("spoofing_ratio", "max", 0.0).clip(0, 1)
    bars["spoofing_duration"]  = _rs("spoofing_duration", "max", 0.0)
    bars["liquidity_trap"]     = _rs("liquidity_trap", "max", 0.0).clip(0, 1)
    bars["vnet"]               = _rs("vnet", "sum", 0.0)
    bars["volume_burst"]       = _rs("volume_burst", "max", 0.0)
    bars["liquidity_sweep"]    = _rs("liquidity_sweep", "max", 0.0)
    bars["micro_atr"]          = _rs("micro_atr", "mean", 0.0)
    bars["fisher_signal"]      = _rs("fisher_signal", "last", 0.0)
    bars["anomaly"]            = _rs("anomaly", "max", 0.0)
    bars["cvd_momentum"]       = _rs("cvd_momentum", "last", 0.0)
    bars["cvd_price_divergence"]= _rs("cvd_price_divergence", "last", 0.0)
    bars["trend_strength"]     = _rs("trend_strength", "last", 0.0)
    bars["correction_depth"]   = _rs("correction_depth", "last", 0.0)
    bars["vwap_z_score"]       = _rs("vwap_z_score", "last", 0.0)
    bars["current_vwap"]       = _rs("current_vwap", "last", np.nan)

    # OBI + Walls
    for col, how in [
        ("obi", "mean"),
        ("bid_wall_strength", "mean"),
        ("ask_wall_strength", "mean"),
        ("distance_to_wall", "mean"),
        ("gap_size", "mean"),
        ("liquidity_density", "mean"),
        ("micro_price", "last"),
    ]:
        bars[col] = _rs(col, how, 0.0)

    # Buy/Sell volumes
    buy_vol  = size_s[is_buy].resample(freq).sum().reindex(bars.index).fillna(0)
    sell_vol = size_s[is_sell].resample(freq).sum().reindex(bars.index).fillna(0)
    bars["buy_volume"]  = buy_vol
    bars["sell_volume"] = sell_vol
    total = (buy_vol + sell_vol).clip(lower=1)
    bars["buy_ratio"]   = (buy_vol / total).fillna(0.5)

    bars["tick_count"] = df["price"].resample(freq).count().reindex(bars.index).fillna(0)

    bars = bars.reset_index()
    print(f"  ✅ Aggregated: {len(bars):,} bars @ {freq}")
    return bars


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — دمج MBP depth مع الشمعات
# ══════════════════════════════════════════════════════════════════

def attach_mbp_depth_to_bars(
    df_bars: pd.DataFrame,
    df_mbp: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """
    يدمج عمق السوق الحقيقي مع كل شمعة.

    لكل شمعة يأخذ أقوى snapshot (peak imbalance) داخلها:
    - bid_depth_total / ask_depth_total: إجمالي العمق
    - book_imbalance: (bid - ask) / (bid + ask) ∈ [-1, 1]
    - spread_mean: متوسط الفارق داخل الشمعة
    - depth_ratio: عمق البيد ÷ عمق الأسك
    - mbp_bar_coverage: نسبة snapshots المتاحة
    """
    out = df_bars.copy()
    mbp = df_mbp.copy()
    mbp["ts_event"] = pd.to_datetime(mbp["ts_event"], errors="coerce")
    mbp = mbp.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)

    bar_ts = pd.to_datetime(out["ts_event"])
    freq_td = pd.to_timedelta(freq)
    expected_snaps = max(1, int(freq_td.total_seconds() / 0.5))  # snapshot كل 0.5 ثانية

    bid_depth_total = np.zeros(len(out))
    ask_depth_total = np.zeros(len(out))
    book_imbalance  = np.full(len(out), np.nan)
    spread_mean     = np.full(len(out), np.nan)
    depth_ratio     = np.full(len(out), np.nan)
    coverage        = np.zeros(len(out))

    mbp_ts_arr = mbp["ts_event"].to_numpy(dtype="datetime64[ns]")

    def _col_arr(col: str) -> np.ndarray:
        if col in mbp.columns:
            return pd.to_numeric(mbp[col], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        return np.zeros(len(mbp))

    bid_sz = np.column_stack([_col_arr(c) for c in MBP_BID_SZ])
    ask_sz = np.column_stack([_col_arr(c) for c in MBP_ASK_SZ])
    bid_px = np.column_stack([_col_arr(c) for c in MBP_BID_PX]) if all(c in mbp.columns for c in MBP_BID_PX) else None
    ask_px = np.column_stack([_col_arr(c) for c in MBP_ASK_PX]) if all(c in mbp.columns for c in MBP_ASK_PX) else None

    for i, ts in enumerate(bar_ts):
        t0 = np.datetime64(ts, "ns")
        t1 = np.datetime64(ts + freq_td, "ns")
        lo = int(np.searchsorted(mbp_ts_arr, t0, side="left"))
        hi = int(np.searchsorted(mbp_ts_arr, t1, side="left"))
        n_snaps = hi - lo
        coverage[i] = min(n_snaps / expected_snaps, 1.0)

        if n_snaps == 0:
            continue

        sl = slice(lo, hi)
        b_total = bid_sz[sl].sum(axis=1)
        a_total = ask_sz[sl].sum(axis=1)
        tot = b_total + a_total + 1e-9
        imb = (b_total - a_total) / tot

        # أقوى imbalance snapshot
        best = int(np.argmax(np.abs(imb)))
        bid_depth_total[i] = float(b_total[best])
        ask_depth_total[i] = float(a_total[best])
        book_imbalance[i]  = float(imb[best])
        depth_ratio[i]     = float(b_total[best] / max(a_total[best], 1e-9))

        if bid_px is not None and ask_px is not None:
            b0 = bid_px[sl, 0]
            a0 = ask_px[sl, 0]
            valid = (b0 > 0) & (a0 > b0)
            if valid.any():
                spread_mean[i] = float(np.mean(a0[valid] - b0[valid]))

    out["bid_depth_total"] = bid_depth_total
    out["ask_depth_total"] = ask_depth_total
    out["book_imbalance"]  = book_imbalance    # [-1, 1]
    out["spread_mean"]     = spread_mean
    out["depth_ratio"]     = np.clip(depth_ratio, 0, 10)
    out["mbp_bar_coverage"]= coverage.astype(np.float32)

    n_covered = (coverage > 0.3).sum()
    print(f"  ✅ MBP depth merged: {n_covered}/{len(out)} bars با coverage > 30%")
    return out


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — تنظيف الشمعات وإصلاح الـ features
# ══════════════════════════════════════════════════════════════════

def clean_bar_prices(df: pd.DataFrame, max_move_pct: float = 0.015) -> pd.DataFrame:
    """
    يصلح close outliers بـ ffill:
    شمعات بحركة > max_move_pct → close/high/low = ffill
    """
    out = df.copy().sort_values("ts_event").reset_index(drop=True)
    ret = out["close"].pct_change().abs()
    bad = ret > max_move_pct
    n_bad = bad.sum()
    if n_bad > 0:
        print(f"  🔧 إصلاح {n_bad} شمعة بـ close outlier (>{max_move_pct*100:.1f}%)")
        for col in ("open", "high", "low", "close"):
            out.loc[bad, col] = np.nan
        out[["open", "high", "low", "close"]] = (
            out[["open", "high", "low", "close"]].ffill()
        )
    return out


def clip_feature_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    يُقيّد الـ features المعروفة بـ outliers:
    dist_to_pdh / distance_to_wall / inter_event_time / etc.
    """
    out = df.copy()
    for col, (lo, hi) in CLIP_RULES.items():
        if col in out.columns:
            n_clipped = ((out[col] < lo) | (out[col] > hi)).sum()
            if n_clipped > 0:
                out[col] = out[col].clip(lo, hi)
                print(f"  📐 clip {col}: [{lo}, {hi}] → {n_clipped} قيمة مُقيّدة")

    # liquidity_density: clip عند P99
    if "liquidity_density" in out.columns:
        p99 = out["liquidity_density"].quantile(0.99)
        n_clipped = (out["liquidity_density"] > p99).sum()
        if n_clipped > 0:
            out["liquidity_density"] = out["liquidity_density"].clip(0, p99)
            print(f"  📐 clip liquidity_density @ P99={p99:.0f}: {n_clipped} قيمة مُقيّدة")

    return out


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — Normalization التكيفي
# ══════════════════════════════════════════════════════════════════

def normalize_features(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """
    كل feature بطريقة normalization مناسبة لطبيعته:

    Percentile Rank (absorption, hawkes, volume_burst):
      يعطي [0,1] مستقل عن تغير الـ scale
      لو السوق صار أكثر نشاطاً، الـ rank يتكيف تلقائياً

    Rolling Z-Score (cvd, vnet, session_cvd):
      يعطي انحراف عن المتوسط المحلي
      مستقل عن المستوى المطلق

    كل الأعمدة المُعالجة تُضاف بـ _norm suffix
    الأعمدة الأصلية تبقى كما هي
    """
    out = df.copy()

    # ① Percentile rank
    for feat in RANK_FEATURES:
        if feat not in out.columns:
            continue
        s = pd.to_numeric(out[feat], errors="coerce").fillna(0)
        ranked = s.rolling(window, min_periods=max(2, min(window // 10, window))).rank(pct=True)
        out[f"{feat}_rank"] = ranked.fillna(0.5).astype(np.float32)

    # ② Rolling Z-Score
    for feat in ZSCORE_FEATURES:
        if feat not in out.columns:
            continue
        s  = pd.to_numeric(out[feat], errors="coerce").fillna(0)
        mu = s.rolling(window, min_periods=max(2, min(window // 10, window))).mean()
        sd = s.rolling(window, min_periods=max(2, min(window // 10, window))).std().clip(lower=1e-9)
        out[f"{feat}_z"] = ((s - mu) / sd).clip(-4, 4).fillna(0).astype(np.float32)

    # ③ book_imbalance من MBP: يبقى [-1, 1] بلا تعديل
    if "book_imbalance" in out.columns:
        out["book_imbalance"] = pd.to_numeric(
            out["book_imbalance"], errors="coerce"
        ).clip(-1, 1).fillna(0).astype(np.float32)

    # ④ depth_ratio: log scale
    if "depth_ratio" in out.columns:
        dr = pd.to_numeric(out["depth_ratio"], errors="coerce").clip(0.1, 10).fillna(1)
        out["depth_ratio_log"] = np.log(dr).astype(np.float32)

    print(f"  ✅ Normalization: {len(RANK_FEATURES)} rank + {len(ZSCORE_FEATURES)} zscore features")
    return out


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — إضافة Features تقنية ومستويات
# ══════════════════════════════════════════════════════════════════

def add_technical_features(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """
    ATR، RSI، session tags، dist_to_pdh، price_position
    كل الحسابات سببية (لا lookahead)
    """
    out = df.copy().sort_values("ts_event").reset_index(drop=True)

    close = pd.to_numeric(out["close"], errors="coerce")
    high  = pd.to_numeric(out["high"],  errors="coerce")
    low   = pd.to_numeric(out["low"],   errors="coerce")
    open_ = pd.to_numeric(out["open"],  errors="coerce")
    vol   = pd.to_numeric(out["volume"],errors="coerce").fillna(0)

    # ATR
    hl  = high - low
    hcp = (high - close.shift(1)).abs()
    lcp = (low  - close.shift(1)).abs()
    tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14, min_periods=1).mean()
    out["bar_range"] = hl

    # Bar body / wick
    body = (close - open_).abs()
    out["body_ratio"] = (body / hl.clip(lower=1e-8)).clip(0, 1)
    body_hi  = pd.concat([open_, close], axis=1).max(axis=1)
    body_lo  = pd.concat([open_, close], axis=1).min(axis=1)
    out["upper_wick"] = (high - body_hi).clip(lower=0)
    out["lower_wick"] = (body_lo - low).clip(lower=0)
    out["wick_ratio"] = (
        (out["lower_wick"] - out["upper_wick"]) / out["atr_14"].clip(lower=1e-9)
    ).clip(-3, 3).fillna(0)

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs    = gain / loss.clip(lower=1e-8)
    out["rsi_14"] = (100 - 100 / (1 + rs)).clip(0, 100)

    # Session tags
    t = pd.to_datetime(out["ts_event"]).dt.time
    def _sess_mask(start_hhmm: str, end_hhmm: str) -> pd.Series:
        st = pd.to_datetime(start_hhmm).time()
        en = pd.to_datetime(end_hhmm).time()
        if st <= en:
            return (t >= st) & (t < en)
        return (t >= st) | (t < en)

    out["is_london"]  = _sess_mask("07:00", "12:00").astype(np.int8)
    out["is_overlap"] = _sess_mask("13:30", "16:00").astype(np.int8)
    out["is_ny"]      = _sess_mask("13:30", "20:00").astype(np.int8)
    out["is_asia"]    = _sess_mask("00:00", "07:00").astype(np.int8)

    # dist_to_pdh / price_position (سببي — يوم أمس فقط)
    ts_dt = pd.to_datetime(out["ts_event"])
    day   = ts_dt.dt.normalize()

    # PDH/PDL من يوم أمس
    daily_hi = (
        pd.DataFrame({"d": day, "h": high})
        .groupby("d")["h"].max()
        .shift(1)
    )
    daily_lo = (
        pd.DataFrame({"d": day, "l": low})
        .groupby("d")["l"].min()
        .shift(1)
    )
    pdh_map = daily_hi.to_dict()
    pdl_map = daily_lo.to_dict()

    pdh = day.map(pdh_map).astype(float)
    pdl = day.map(pdl_map).astype(float)
    atr = out["atr_14"].clip(lower=1e-9)

    out["pdh"] = pdh
    out["pdl"] = pdl
    out["dist_to_pdh"] = ((close - pdh) / atr).clip(-20, 20).fillna(0)
    day_range = (pdh - pdl).clip(lower=1e-9)
    out["price_position"] = ((close - pdl) / day_range).clip(0, 1).fillna(0.5)

    # LR / HR classification
    avg_range = hl.rolling(20, min_periods=5).mean().clip(lower=1e-9)
    out["range_ratio"]  = (hl / avg_range).fillna(1.0).clip(0, 5)
    out["is_lr_bar"]    = (out["range_ratio"] < 0.7).astype(np.int8)
    out["is_hr_bar"]    = (out["range_ratio"] > 1.3).astype(np.int8)
    out["prev_is_lr"]   = out["is_lr_bar"].shift(1).fillna(0).astype(np.int8)

    # Volume vs average
    vol_avg = vol.rolling(20, min_periods=5).mean().clip(lower=1)
    out["volume_ratio"] = (vol / vol_avg).clip(0, 10).fillna(1)

    print(f"  ✅ Technical features added ({len([c for c in out.columns if c not in df.columns])} جديد)")
    return out


# ══════════════════════════════════════════════════════════════════
# SECTION 8 — Forward Return نظيف للـ IC
# ══════════════════════════════════════════════════════════════════

def add_fwd_ret_clean(df: pd.DataFrame, n_bars: int) -> pd.DataFrame:
    """
    fwd_ret_clean = (close[t+n] - close[t]) / close[t]
    سببي، أفق ثابت، لا lookahead
    آخر n_bars صف = NaN
    """
    out = df.copy()
    c = out["close"].to_numpy(dtype=float)
    arr = np.full(len(c), np.nan, dtype=np.float32)
    if len(c) > n_bars:
        lo = c[:-n_bars]
        hi = c[n_bars:]
        m  = np.isfinite(lo) & np.isfinite(hi) & (np.abs(lo) > 1e-9)
        arr[: len(lo)] = np.where(m, (hi / lo - 1.0), np.nan).astype(np.float32)
    out["fwd_ret_clean"] = arr
    n_valid = (~np.isnan(arr)).sum()
    print(f"  ✅ fwd_ret_clean: N={n_bars} | {n_valid:,} قيمة صالحة")
    return out


# ══════════════════════════════════════════════════════════════════
# SECTION 9 — Pipeline الكامل
# ══════════════════════════════════════════════════════════════════

def run_pipeline(
    mbo_path: str | None = None,
    mbp_path: str | None = None,
    from_parquet: str | None = None,
    output_dir: str = "pipeline_output",
    freq: str = "5min",
    horizon: int = 6,
    max_move_pct: float = 0.015,
    norm_window: int = 200,
    artifact_tag: str | None = None,
    max_rows: int | None = None,
) -> str:
    """
    Pipeline كامل من MBO+MBP خام → parquet جاهز للمحاكيات
    أو من parquet موجود مع تنظيف فقط
    """
    os.makedirs(output_dir, exist_ok=True)
    tag    = artifact_tag or ""
    suffix = f"_{tag}" if tag else ""
    out_fn = f"clean_features{suffix}.parquet"

    print("\n" + "=" * 60)
    print("🔧 Data Pipeline — تجهيز الداتا للمحاكيات")
    print(f"   freq={freq} | horizon={horizon} | tag={tag or 'none'}")
    print("=" * 60)

    # ── مسار A: من MBO+MBP خام ──
    if from_parquet is None:
        if mbo_path is None:
            raise ValueError("يجب تمرير --mbo أو --from-parquet")

        print("\n📥 تحميل MBO...")
        ext = Path(mbo_path).suffix.lower()
        df_mbo = pd.read_parquet(mbo_path) if ext in (".parquet", ".pq") \
                 else pd.read_csv(mbo_path, low_memory=False)
        df_mbo = sanitize_mbo_ticks(df_mbo)

        df_mbp = None
        if mbp_path:
            print("\n📥 تحميل MBP...")
            ext_m = Path(mbp_path).suffix.lower()
            df_mbp_raw = pd.read_parquet(mbp_path) if ext_m in (".parquet", ".pq") \
                         else pd.read_csv(mbp_path, low_memory=False)
            df_mbp = sanitize_mbp_ticks(df_mbp_raw)

        print(f"\n⏱  تجميع في {freq} bars...")
        df_bars = aggregate_ticks_to_bars(df_mbo, freq=freq)

        if df_mbp is not None and len(df_mbp) > 0:
            print("\n📊 دمج عمق السوق MBP...")
            df_bars = attach_mbp_depth_to_bars(df_bars, df_mbp, freq=freq)
        else:
            df_bars["mbp_bar_coverage"] = np.float32(0.0)
            df_bars["book_imbalance"]   = np.float32(0.0)
            df_bars["depth_ratio"]      = np.float32(1.0)
            print("  ⚠️  بدون MBP — book_imbalance = 0")

    # ── مسار B: من parquet موجود ──
    else:
        print(f"\n📂 تحميل من parquet: {from_parquet}")
        df_bars = pd.read_parquet(from_parquet)
        df_bars["ts_event"] = pd.to_datetime(df_bars["ts_event"])
        print(f"  ✅ {len(df_bars):,} rows | {len(df_bars.columns)} cols")

    # ── تنظيف وإصلاح ──
    print("\n🧹 تنظيف الشمعات...")
    df_bars = clean_bar_prices(df_bars, max_move_pct=max_move_pct)
    df_bars = clip_feature_outliers(df_bars)

    print("\n📈 إضافة features تقنية...")
    df_bars = add_technical_features(df_bars, freq=freq)

    print("\n📐 Normalization التكيفي...")
    df_bars = normalize_features(df_bars, window=norm_window)

    print("\n🎯 إضافة fwd_ret_clean...")
    df_bars = add_fwd_ret_clean(df_bars, n_bars=horizon)

    # ── تقرير جودة ──
    print("\n" + "─" * 50)
    print("📊 تقرير الجودة:")
    print(f"   الصفوف: {len(df_bars):,}")
    print(f"   الفترة: {df_bars['ts_event'].min()} → {df_bars['ts_event'].max()}")

    fwd = df_bars["fwd_ret_clean"].dropna()
    print(f"   fwd_ret_clean std: {fwd.std():.5f} "
          f"({'✅ طبيعي' if 0.001 < fwd.std() < 0.05 else '⚠️ راجع'})")

    # features مكسورة؟
    for col in ("kyle_lambda", "hawkes_intensity", "absorption_intensity"):
        if col not in df_bars.columns:
            continue
        s = df_bars[col].dropna()
        flat = s.std() < 1e-4
        print(f"   {col}: std={s.std():.4f} {'❌ FLAT' if flat else '✅'}")

    # MBP coverage
    if "mbp_bar_coverage" in df_bars.columns:
        cov = df_bars["mbp_bar_coverage"].mean()
        print(f"   MBP coverage: {cov:.1%} "
              f"({'✅' if cov > 0.3 else '⚠️ منخفض — MBP data ضعيفة'})")

    # ── تصدير ──
    out_path = os.path.join(output_dir, out_fn)
    df_bars.to_parquet(out_path, index=False)

    # manifest
    manifest = {
        "pipeline_version": "v1",
        "freq": freq,
        "horizon": horizon,
        "rows": len(df_bars),
        "cols": len(df_bars.columns),
        "ts_min": str(df_bars["ts_event"].min()),
        "ts_max": str(df_bars["ts_event"].max()),
        "fwd_ret_std": float(fwd.std()),
        "mbp_coverage": float(df_bars.get("mbp_bar_coverage", pd.Series([0])).mean()),
        "fixes_applied": [
            "hawkes_delta_not_max",
            "kyle_mean_not_max",
            "absorption_clip_0_20",
            "inter_event_time_clip_3600",
            "close_outliers_ffill",
            "dist_to_pdh_clip_20",
            "distance_to_wall_clip_50",
            "liquidity_density_clip_p99",
            "adaptive_normalization",
        ],
    }
    manifest_path = os.path.join(output_dir, f"pipeline_manifest{suffix}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n💾 مُحفظ: {out_path}")
    print(f"💾 Manifest: {manifest_path}")
    print("\n✅ Pipeline اكتمل — الداتا جاهزة للمحاكيات")
    print("=" * 60)
    return out_path


# ══════════════════════════════════════════════════════════════════
# SECTION 10 — CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    try:
        for _stream in (sys.stdout, sys.stderr):
            if hasattr(_stream, "reconfigure"):
                _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Data Pipeline — تجهيز الداتا قبل المحاكيات")

    src = ap.add_mutually_exclusive_group()
    src.add_argument("--mbo", default=None, help="مسار MBO parquet/csv")
    src.add_argument(
        "--from-parquet",
        default=None,
        help="تحميل من parquet موجود (تنظيف + normalization فقط)",
    )

    ap.add_argument("--mbp",     default=None,               help="مسار MBP parquet/csv (اختياري)")
    ap.add_argument("--output",  default="pipeline_output",  help="مجلد المخرجات")
    ap.add_argument("--freq",    default="5min",             help="تردد الشمعة (5min, 1min, 15min, 1h)")
    ap.add_argument("--horizon", type=int, default=6,        help="أفق fwd_ret_clean بالشمعات")
    ap.add_argument(
        "--max-move-pct",
        type=float,
        default=0.015,
        help="حد أقصى لحركة close (نسبة مئوية) — فوقه = outlier (افتراضي: 0.015 = 1.5%%)",
    )
    ap.add_argument(
        "--norm-window",
        type=int,
        default=200,
        help="نافذة normalization المتكيف (افتراضي: 200 شمعة)",
    )
    ap.add_argument("--tag", default=None, help="suffix لأسماء الملفات")

    args = ap.parse_args()

    run_pipeline(
        mbo_path     = args.mbo,
        mbp_path     = args.mbp,
        from_parquet = args.from_parquet,
        output_dir   = args.output,
        freq         = args.freq,
        horizon      = args.horizon,
        max_move_pct = args.max_move_pct,
        norm_window  = args.norm_window,
        artifact_tag = args.tag,
    )


if __name__ == "__main__":
    main()
