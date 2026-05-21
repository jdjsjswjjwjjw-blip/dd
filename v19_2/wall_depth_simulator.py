"""
wall_depth_simulator.py — محاكي الجدران العميق
══════════════════════════════════════════════════════════════════════

يقرأ الدفتر الكامل (MBP 10 مستويات) ويفهم الجدران فيزيائياً:
  - أين الحائط بالضبط (أي مستوى عمق)
  - هل يكبر أم يصغر عبر الزمن
  - كم استُهلك (orders نُفِّذت ضده)
  - كم لقطة صمد
  - هل ينتقل (يقترب/يبتعد عن السعر)

الفرق عن Wall Engine في feature_simulators.py:
  ذاك يقرأ bid_wall_strength (رقم مُجمَّع فقير)
  هذا يقرأ bid_sz_00..09 / ask_sz_00..09 (الدفتر الحقيقي)

المدخل: MBP خام
  ts_event
  bid_px_00..09, bid_sz_00..09
  ask_px_00..09, ask_sz_00..09

المخرج: 8 أعمدة wall_* لكل شمعة

الاستخدام:
  from wall_depth_simulator import simulate_wall_depth
  bars = simulate_wall_depth(bars, mbp_df, freq='5min')

CLI:
  python wall_depth_simulator.py --bars sim.parquet --mbp mbp.csv --output walls.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# ثوابت
# ══════════════════════════════════════════════════════════════════

LEVELS = 10
BID_SZ = [f"bid_sz_{i:02d}" for i in range(LEVELS)]
ASK_SZ = [f"ask_sz_{i:02d}" for i in range(LEVELS)]
BID_PX = [f"bid_px_{i:02d}" for i in range(LEVELS)]
ASK_PX = [f"ask_px_{i:02d}" for i in range(LEVELS)]


# ══════════════════════════════════════════════════════════════════
# أدوات
# ══════════════════════════════════════════════════════════════════

def _rank_causal(s: pd.Series, window: int = 200) -> pd.Series:
    """percentile rank متحرك سببي."""
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return s.rolling(window, min_periods=max(2, min(window // 10, window))).rank(pct=True).fillna(0.5)


# ══════════════════════════════════════════════════════════════════
# تحليل الدفتر داخل شمعة واحدة
# ══════════════════════════════════════════════════════════════════

def _analyze_book_window(
    bid_sz: np.ndarray,   # (n_snaps, 10)
    ask_sz: np.ndarray,
    bid_px: np.ndarray,
    ask_px: np.ndarray,
) -> dict:
    """
    يحلل كل لقطات الدفتر داخل شمعة واحدة.

    يكتشف الجدران الحقيقية:
      الحائط = مستوى حجمه أكبر بكثير من متوسط بقية المستويات

    يرجع dict بـ 8 مقاييس خام.
    """
    n_snaps = len(bid_sz)
    if n_snaps == 0:
        return {
            "bid_wall_level": 0.0, "ask_wall_level": 0.0,
            "bid_wall_size": 0.0, "ask_wall_size": 0.0,
            "wall_growth": 0.0, "wall_consumed": 0.0,
            "wall_persist": 0.0, "wall_shift": 0.0,
        }

    # ── ① موقع الحائط = المستوى الأكبر حجماً ──
    # لكل لقطة، أي مستوى فيه أكبر حجم
    bid_wall_lvls = np.argmax(bid_sz, axis=1)   # (n_snaps,)
    ask_wall_lvls = np.argmax(ask_sz, axis=1)

    # الموقع الأكثر تكراراً = الحائط المستقر
    bid_wall_level = float(np.median(bid_wall_lvls))
    ask_wall_level = float(np.median(ask_wall_lvls))

    # ── ② حجم الحائط نسبياً ──
    # حجم أكبر مستوى ÷ متوسط بقية المستويات
    bid_max = bid_sz.max(axis=1)
    ask_max = ask_sz.max(axis=1)
    bid_mean = bid_sz.mean(axis=1) + 1e-9
    ask_mean = ask_sz.mean(axis=1) + 1e-9
    bid_wall_ratio = np.median(bid_max / bid_mean)   # كم الحائط أكبر من المتوسط
    ask_wall_ratio = np.median(ask_max / ask_mean)

    # ── ③ نمو الحائط: أول نصف vs آخر نصف الشمعة ──
    mid = max(1, n_snaps // 2)
    bid_dominant_first = bid_sz[:mid].max(axis=1).mean()
    bid_dominant_last  = bid_sz[mid:].max(axis=1).mean() if mid < n_snaps else bid_dominant_first
    ask_dominant_first = ask_sz[:mid].max(axis=1).mean()
    ask_dominant_last  = ask_sz[mid:].max(axis=1).mean() if mid < n_snaps else ask_dominant_first

    # الحائط المهيمن (الأكبر) — يكبر أم يصغر
    if bid_dominant_last + ask_dominant_last > bid_dominant_first + ask_dominant_first:
        wall_growth = 1.0
    else:
        wall_growth = -1.0
    # شدّة النمو
    total_first = bid_dominant_first + ask_dominant_first + 1e-9
    total_last  = bid_dominant_last + ask_dominant_last
    wall_growth *= min(abs(total_last / total_first - 1.0), 1.0)

    # ── ④ استهلاك الحائط ──
    # الحائط يُستهلَك = حجمه ينخفض بين لقطات متتالية في *نفس المستوى*
    # نتتبع المستوى الأكثر هيمنة
    dominant_bid_lvl = int(np.bincount(bid_wall_lvls).argmax())
    dominant_ask_lvl = int(np.bincount(ask_wall_lvls).argmax())
    bid_wall_series = bid_sz[:, dominant_bid_lvl]
    ask_wall_series = ask_sz[:, dominant_ask_lvl]
    # الانخفاضات = استهلاك
    bid_drops = np.clip(-np.diff(bid_wall_series), 0, None).sum()
    ask_drops = np.clip(-np.diff(ask_wall_series), 0, None).sum()
    bid_total = bid_wall_series.sum() + 1e-9
    ask_total = ask_wall_series.sum() + 1e-9
    wall_consumed = float(np.clip(
        (bid_drops + ask_drops) / (bid_total + ask_total) * n_snaps, 0, 1
    ))

    # ── ⑤ صمود الحائط ──
    # الحائط صامد = نفس المستوى يبقى الأكبر عبر اللقطات
    bid_persist = (bid_wall_lvls == dominant_bid_lvl).mean()
    ask_persist = (ask_wall_lvls == dominant_ask_lvl).mean()
    wall_persist = float((bid_persist + ask_persist) / 2.0)

    # ── ⑥ انتقال الحائط ──
    # هل الحائط المهيمن يقترب من السعر (مستوى أقل) أم يبتعد؟
    if n_snaps >= 4:
        early_lvl = np.median(bid_wall_lvls[:mid]) + np.median(ask_wall_lvls[:mid])
        late_lvl  = np.median(bid_wall_lvls[mid:]) + np.median(ask_wall_lvls[mid:])
        # مستوى أقل = أقرب للسعر = +1 ؛ مستوى أعلى = أبعد = -1
        shift = np.clip((early_lvl - late_lvl) / 4.0, -1, 1)
    else:
        shift = 0.0

    return {
        "bid_wall_level": bid_wall_level,
        "ask_wall_level": ask_wall_level,
        "bid_wall_size": float(bid_wall_ratio),
        "ask_wall_size": float(ask_wall_ratio),
        "wall_growth": float(wall_growth),
        "wall_consumed": wall_consumed,
        "wall_persist": wall_persist,
        "wall_shift": float(shift),
    }


# ══════════════════════════════════════════════════════════════════
# المُشغّل — يربط MBP بالشمعات
# ══════════════════════════════════════════════════════════════════

def simulate_wall_depth(
    bars: pd.DataFrame,
    mbp: pd.DataFrame,
    freq: str = "5min",
    window: int = 200,
) -> pd.DataFrame:
    """
    لكل شمعة، يحلل كل لقطات MBP بداخلها ويُخرج 8 مقاييس جدران.

    المخرج (8 أعمدة):
      sim_wall_bid_level  ← مستوى عمق حائط الشراء (0-9)
      sim_wall_ask_level  ← مستوى عمق حائط البيع
      sim_wall_bid_size   ← حجم حائط الشراء نسبياً [0,1]
      sim_wall_ask_size   ← حجم حائط البيع نسبياً [0,1]
      sim_wall_growth     ← نمو [+] / تقلّص [-] الحائط المهيمن
      sim_wall_consumed   ← استهلاك الحائط [0,1]
      sim_wall_persist    ← صمود الحائط [0,1]
      sim_wall_shift      ← انتقال: اقتراب [+] / ابتعاد [-]
    """
    out = bars.copy()
    out["ts_event"] = pd.to_datetime(out["ts_event"])

    m = mbp.copy()
    m["ts_event"] = pd.to_datetime(m["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    m = m.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)

    print(f"🧱 Wall Depth Simulator — {len(m):,} لقطة MBP...")

    # تحقق من الأعمدة
    have_bid = all(c in m.columns for c in BID_SZ)
    have_ask = all(c in m.columns for c in ASK_SZ)
    if not (have_bid and have_ask):
        print("  ⚠️ أعمدة MBP ناقصة — إخراج قيم افتراضية")
        for col in ["sim_wall_bid_level", "sim_wall_ask_level",
                    "sim_wall_bid_size", "sim_wall_ask_size",
                    "sim_wall_growth", "sim_wall_consumed",
                    "sim_wall_persist", "sim_wall_shift"]:
            out[col] = np.float32(0.0)
        return out

    # مصفوفات الحجم/السعر
    bid_sz_all = m[BID_SZ].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    ask_sz_all = m[ASK_SZ].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    bid_px_all = m[BID_PX].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32) \
                 if all(c in m.columns for c in BID_PX) else np.zeros_like(bid_sz_all)
    ask_px_all = m[ASK_PX].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32) \
                 if all(c in m.columns for c in ASK_PX) else np.zeros_like(ask_sz_all)

    mbp_ts = m["ts_event"].to_numpy(dtype="datetime64[ns]")
    bar_ts = out["ts_event"].to_numpy(dtype="datetime64[ns]")
    freq_ns = int(pd.to_timedelta(freq).total_seconds() * 1e9)

    # نتائج خام
    raw = {k: np.zeros(len(out), dtype=np.float32) for k in [
        "bid_wall_level", "ask_wall_level", "bid_wall_size", "ask_wall_size",
        "wall_growth", "wall_consumed", "wall_persist", "wall_shift",
    ]}

    n_analyzed = 0
    for i in range(len(out)):
        t0 = bar_ts[i]
        t1 = t0 + np.timedelta64(freq_ns, "ns")
        lo = int(np.searchsorted(mbp_ts, t0, side="left"))
        hi = int(np.searchsorted(mbp_ts, t1, side="left"))
        if hi <= lo:
            continue
        n_analyzed += 1
        res = _analyze_book_window(
            bid_sz_all[lo:hi], ask_sz_all[lo:hi],
            bid_px_all[lo:hi], ask_px_all[lo:hi],
        )
        for k, v in res.items():
            raw[k][i] = v

    # ── normalization تكيفي ──
    # المستويات تبقى كما هي (0-9 = معلومة مباشرة)
    out["sim_wall_bid_level"] = raw["bid_wall_level"]
    out["sim_wall_ask_level"] = raw["ask_wall_level"]
    # الحجم: rank تكيفي [0,1]
    out["sim_wall_bid_size"] = _rank_causal(
        pd.Series(raw["bid_wall_size"], index=out.index), window
    ).astype(np.float32)
    out["sim_wall_ask_size"] = _rank_causal(
        pd.Series(raw["ask_wall_size"], index=out.index), window
    ).astype(np.float32)
    # نمو/استهلاك/صمود/انتقال: كما هي (محصورة أصلاً)
    out["sim_wall_growth"]   = raw["wall_growth"]
    out["sim_wall_consumed"] = raw["wall_consumed"]
    out["sim_wall_persist"]  = raw["wall_persist"]
    out["sim_wall_shift"]    = raw["wall_shift"]

    print(f"  ✅ حُلِّلت {n_analyzed}/{len(out)} شمعة بعمق دفتر كامل")
    return out


# قائمة المخرجات
WALL_DEPTH_OUTPUTS = [
    "sim_wall_bid_level", "sim_wall_ask_level",
    "sim_wall_bid_size", "sim_wall_ask_size",
    "sim_wall_growth", "sim_wall_consumed",
    "sim_wall_persist", "sim_wall_shift",
]


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Wall Depth Simulator — deep order book walls")
    ap.add_argument("--bars",   required=True, help="parquet الشمعات (sim_features)")
    ap.add_argument("--mbp",    required=True, help="MBP خام (csv/parquet)")
    ap.add_argument("--output", default=None,  help="parquet المخرج")
    ap.add_argument("--freq",   default="5min", help="تردد الشمعة")
    ap.add_argument("--window", type=int, default=200)
    args = ap.parse_args()

    bars = pd.read_parquet(args.bars)
    ext = Path(args.mbp).suffix.lower()
    mbp = pd.read_csv(args.mbp, low_memory=False) if ext == ".csv" \
          else pd.read_parquet(args.mbp)
    print(f"📥 شمعات: {len(bars):,} | MBP: {len(mbp):,}")

    bars = simulate_wall_depth(bars, mbp, freq=args.freq, window=args.window)

    out_path = args.output or str(
        Path(args.bars).with_name(Path(args.bars).stem + "_walls.parquet")
    )
    bars.to_parquet(out_path, index=False)
    print(f"\n💾 محفوظ: {out_path}")
    print(f"   8 مخرجات جدران: {WALL_DEPTH_OUTPUTS}")


if __name__ == "__main__":
    main()
