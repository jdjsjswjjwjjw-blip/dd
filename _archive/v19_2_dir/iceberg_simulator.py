"""
iceberg_simulator.py — كشف الـ Iceberg المؤسسي (هجين 4 طبقات)
══════════════════════════════════════════════════════════════════════

ليس كشف iceberg عادي.
هذا المحاكي مصمَّم لكشف iceberg المؤسسي = الطرف الذي يخفي
حجمه بذكاء ليبني/يصرّف موقعاً ضخماً دون تحريك السعر.

⚠️ مهم — حقيقة السوق:
  المؤسسة *لا* تجدّد بنفس order_id.
  كل order جديد له order_id جديد:
    ① Order_id=123 (5 lot) → Filled
    ② Order_id=456 (6 lot) جديد عند نفس السعر
    ③ Order_id=789 (4 lot) جديد آخر...

  لهذا المحاكي *لا* يعتمد على order_id إطلاقاً.
  الكشف يعتمد على نمط الـ (سعر + زمن + حجم متشابه + action).

التحديات في كشف iceberg المؤسسي:
  - الحجم متغيّر (5, 7, 3, 8) لتمويه الخوارزميات
  - توقيت التجديد عشوائي (200ms - 2s)
  - يتحرك بين مستويات الدفتر
  - متعدد الأرجل (orders في عدة مستويات)
  - يختفي مؤقتاً ثم يعود

الكشف الهجين — 4 طبقات (كلها بدون order_id):
  ① Trade Volume Imbalance
     trades ضد سعر معين ÷ متوسط visible size في ذلك السعر > 3×
  ② Replenishment Detection
     بعد Trade في سعر X، هل ظهر Add جديد في *نفس السعر X*
     خلال 500ms بحجم متشابه (±30%)؟
     = لا يهم order_id الجديد، يهم النمط
  ③ Cross-Level Coordination
     Trade في سعر X + Add في سعر X±tick خلال 100ms
     = نفس المؤسسة تحرّك أوامرها بأرجل متعددة
  ④ Wall Persistence vs Consumption
     wall_persist عالٍ + wall_consumed عالٍ = حائط يصمد
     رغم الاستهلاك = iceberg

المدخل: MBO خام
  ts_event   — توقيت نانوسكند
  action     — A (Add) / C (Cancel) / M (Modify) / T (Trade) / F (Fill)
  side       — B (Bid) / A (Ask)
  price      — سعر الـ order
  size       — حجمه
  [order_id  — موجود لكن غير مستخدم في الكشف]

المخرج (5):
  sim_iceberg_prob       ← احتمال [0,1]
  sim_iceberg_side       ← شرائي [+1] / بيعي [-1] / 0
  sim_iceberg_strength   ← قوة [0,1]
  sim_iceberg_replenish  ← سرعة التجديد [0,1]
  sim_iceberg_stealth    ← مستوى التمويه [0,1]

CLI:
  python iceberg_simulator.py --bars bars.parquet --mbo mbo.csv --output ice.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# ثوابت الكشف
# ══════════════════════════════════════════════════════════════════

REPLENISH_WINDOW_MS = 500    # نافذة كشف التجديد بعد التنفيذ
REPLENISH_TOL = 0.30         # تسامح: الحجم الجديد ضمن ±30% من الأصلي
MIN_TRADES_FOR_ICE = 3       # حد أدنى لـ trades متتالية ضد نفس السعر
HIDDEN_RATIO_THRESHOLD = 3.0 # trades/visible > 3× = طرف مخفي


# ══════════════════════════════════════════════════════════════════
# أدوات
# ══════════════════════════════════════════════════════════════════

def _rank(s: pd.Series, window: int = 200) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mp = max(2, min(window // 10, window))   # min_periods لا يتجاوز window أبداً
    return s.rolling(window, min_periods=mp).rank(pct=True).fillna(0.5)


# ══════════════════════════════════════════════════════════════════
# الطبقة ① — Trade Volume Imbalance
# ══════════════════════════════════════════════════════════════════

def _layer1_volume_imbalance(
    trades_at_price: dict[float, list[tuple[int, float, str]]],
    visible_size_history: dict[float, list[float]],
) -> dict[float, float]:
    """
    لكل سعر:
      hidden_ratio = إجمالي trades / متوسط الـ visible size
      > 3 = iceberg محتمل
    """
    hidden = {}
    for price, trades in trades_at_price.items():
        total_traded = sum(t[1] for t in trades)
        visible_avg = np.mean(visible_size_history.get(price, [1.0])) + 1e-9
        hidden[price] = float(total_traded / visible_avg)
    return hidden


# ══════════════════════════════════════════════════════════════════
# الطبقة ② — Replenishment Detection
# ══════════════════════════════════════════════════════════════════

def _layer2_replenishment(
    events: list[tuple[np.int64, str, str, float, float]],  # (ts_ns, action, side, price, size)
    window_ms: int = REPLENISH_WINDOW_MS,
) -> dict[float, dict]:
    """
    لكل سعر، يكشف:
      - replenish_count: كم مرة عاد الحجم بعد تنفيذ
      - replenish_speed_avg: متوسط زمن التجديد بـ ms
      - size_variance: تباين أحجام التجديد (مؤشر تمويه)
    """
    by_price: dict[float, dict] = {}
    window_ns = window_ms * 1_000_000

    # ترتيب حسب السعر
    by_price_events: dict[float, list] = {}
    for ev in events:
        ts, action, side, price, size = ev
        by_price_events.setdefault(price, []).append(ev)

    for price, evs in by_price_events.items():
        evs_sorted = sorted(evs, key=lambda x: x[0])
        replenish_count = 0
        replenish_speeds = []
        replenish_sizes = []
        last_trade_idx = -1
        last_trade_size = 0.0

        for i, (ts, action, side, _, size) in enumerate(evs_sorted):
            if action in ("T", "F"):  # Trade/Fill
                last_trade_idx = i
                last_trade_size = size
            elif action == "A" and last_trade_idx >= 0:  # Add بعد Trade
                ts_trade = evs_sorted[last_trade_idx][0]
                dt_ns = ts - ts_trade
                if 0 < dt_ns <= window_ns:
                    # تحقق من التشابه (±30%)
                    if last_trade_size > 0:
                        ratio = size / last_trade_size
                        if (1 - REPLENISH_TOL) <= ratio <= (1 + REPLENISH_TOL) * 2:
                            replenish_count += 1
                            replenish_speeds.append(dt_ns / 1_000_000)
                            replenish_sizes.append(size)
                    last_trade_idx = -1  # استخدمناه

        by_price[price] = {
            "count":     replenish_count,
            "avg_ms":    float(np.mean(replenish_speeds)) if replenish_speeds else 0.0,
            "size_cv":   float(np.std(replenish_sizes) / (np.mean(replenish_sizes) + 1e-9))
                         if len(replenish_sizes) > 1 else 0.0,
        }
    return by_price


# ══════════════════════════════════════════════════════════════════
# الطبقة ③ — Cross-Level Coordination
# ══════════════════════════════════════════════════════════════════

def _layer3_cross_level(
    events: list[tuple[np.int64, str, str, float, float]],
    tick_size: float = 0.0001,
) -> float:
    """
    يكشف تنسيقاً بين مستويات الدفتر:
      Trade في price X + Add في price X±tick خلال 100ms
      = نفس المؤسسة تنقل أوامرها
    """
    coord_count = 0
    window_ns = 100 * 1_000_000

    for i, (ts_a, act_a, side_a, px_a, _) in enumerate(events):
        if act_a not in ("T", "F"):
            continue
        # ابحث في نافذة 100ms عن Add في مستوى مجاور
        for j in range(i + 1, min(i + 50, len(events))):
            ts_b, act_b, side_b, px_b, _ = events[j]
            if ts_b - ts_a > window_ns:
                break
            if act_b == "A" and side_b == side_a:
                # هل المستوى مجاور؟
                if abs(px_b - px_a) <= tick_size * 2 and abs(px_b - px_a) > 1e-9:
                    coord_count += 1
                    break
    return float(coord_count)


# ══════════════════════════════════════════════════════════════════
# المُشغّل — لكل شمعة
# ══════════════════════════════════════════════════════════════════

def _analyze_bar(
    mbo_slice: pd.DataFrame,
    wall_persist_val: float,
    wall_consumed_val: float,
) -> dict:
    """
    يحلّل MBO داخل شمعة واحدة عبر الطبقات الأربع.
    """
    if len(mbo_slice) < 10:
        return _empty_result()

    # ── جمع الأحداث ──
    events = []
    trades_at_price: dict[float, list] = {}
    visible_at_price: dict[float, list] = {}
    trades_by_side = {"B": 0.0, "A": 0.0}

    for _, row in mbo_slice.iterrows():
        ts = np.int64(pd.Timestamp(row["ts_event"]).value)
        act = str(row.get("action", "")).upper()
        side = str(row.get("side", "")).upper()
        try:
            price = float(row.get("price", 0))
            size = float(row.get("size", 0))
        except (TypeError, ValueError):
            continue
        if price <= 0 or size <= 0:
            continue
        events.append((ts, act, side, price, size))

        if act in ("T", "F"):
            trades_at_price.setdefault(price, []).append((ts, size, side))
            if side in trades_by_side:
                trades_by_side[side] += size
        elif act == "A":
            visible_at_price.setdefault(price, []).append(size)

    if not trades_at_price:
        return _empty_result()

    # ── الطبقة ① ──
    l1 = _layer1_volume_imbalance(trades_at_price, visible_at_price)
    max_hidden_ratio = max(l1.values()) if l1 else 0.0
    # السعر ذو أعلى hidden ratio = موقع الـ iceberg
    iceberg_price = max(l1, key=l1.get) if l1 else 0.0
    iceberg_side_trades = trades_at_price.get(iceberg_price, [])
    iceberg_side = 0.0
    if iceberg_side_trades:
        sides = [t[2] for t in iceberg_side_trades]
        n_buy = sum(1 for s in sides if s in ("A", "ASK"))
        n_sell = sum(1 for s in sides if s in ("B", "BID"))
        # Trade على ASK = شراء (هاجم العرض) → +1
        # Trade على BID = بيع (هاجم الطلب) → -1
        iceberg_side = 1.0 if n_buy > n_sell else (-1.0 if n_sell > n_buy else 0.0)

    # ── الطبقة ② ──
    l2 = _layer2_replenishment(events)
    total_replenish = sum(d["count"] for d in l2.values())
    avg_replenish_ms = (
        np.mean([d["avg_ms"] for d in l2.values() if d["avg_ms"] > 0])
        if any(d["avg_ms"] > 0 for d in l2.values()) else 1000.0
    )
    avg_size_cv = (
        np.mean([d["size_cv"] for d in l2.values() if d["size_cv"] > 0])
        if any(d["size_cv"] > 0 for d in l2.values()) else 0.0
    )

    # ── الطبقة ③ ──
    l3_coord = _layer3_cross_level(events)

    # ── دمج الطبقات الأربع ──
    # نُطبّع كل مؤشر ثم نمزج
    score_l1 = float(np.clip(max_hidden_ratio / HIDDEN_RATIO_THRESHOLD, 0, 1))
    score_l2 = float(np.clip(total_replenish / 5.0, 0, 1))
    score_l3 = float(np.clip(l3_coord / 10.0, 0, 1))
    score_l4 = float(wall_persist_val * wall_consumed_val * 2.0)  # كلاهما عالٍ
    score_l4 = min(score_l4, 1.0)

    # الاحتمال الكلي = متوسط مرجّح
    prob = (
        0.30 * score_l1 +
        0.30 * score_l2 +
        0.15 * score_l3 +
        0.25 * score_l4
    )

    # سرعة التجديد: أسرع = أعلى (عكس الزمن)
    replenish_speed = float(np.clip(1.0 - (avg_replenish_ms / 500.0), 0, 1))

    # التمويه (stealth): تباين الحجم عالٍ + تنسيق عبر المستويات
    stealth = float(np.clip(
        0.5 * np.clip(avg_size_cv, 0, 1) + 0.5 * score_l3, 0, 1
    ))

    # قوة الـ iceberg: حجم الـ trades الكلي على الجانب المهيمن
    total_iceberg_vol = sum(t[1] for t in iceberg_side_trades) if iceberg_side_trades else 0.0
    total_all_vol = sum(trades_by_side.values()) + 1e-9
    strength = float(np.clip(total_iceberg_vol / total_all_vol, 0, 1))

    return {
        "prob":      prob,
        "side":      iceberg_side,
        "strength":  strength,
        "replenish": replenish_speed,
        "stealth":   stealth,
    }


def _empty_result() -> dict:
    return {"prob": 0.0, "side": 0.0, "strength": 0.0, "replenish": 0.0, "stealth": 0.0}


# ══════════════════════════════════════════════════════════════════
# المُشغّل الرئيسي
# ══════════════════════════════════════════════════════════════════

def simulate_iceberg(
    bars: pd.DataFrame,
    mbo: pd.DataFrame,
    freq: str = "5min",
    window: int = 200,
) -> pd.DataFrame:
    """
    يكشف iceberg المؤسسي في كل شمعة.

    المخرجات:
      sim_iceberg_prob       ← احتمال [0,1]
      sim_iceberg_side       ← شرائي [+1] / بيعي [-1] / 0
      sim_iceberg_strength   ← قوة [0,1]
      sim_iceberg_replenish  ← سرعة التجديد [0,1]
      sim_iceberg_stealth    ← مستوى التمويه [0,1]
    """
    out = bars.copy()
    out["ts_event"] = pd.to_datetime(out["ts_event"])

    m = mbo.copy()
    m["ts_event"] = pd.to_datetime(m["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    m = m.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)

    print(f"🧊 Iceberg Detector — {len(m):,} MBO event...")

    # تحقق من الأعمدة
    needed = ["ts_event", "action", "side", "price", "size"]
    missing = [c for c in needed if c not in m.columns]
    if missing:
        print(f"  ⚠️ أعمدة MBO ناقصة: {missing} — قيم افتراضية")
        for col in ["sim_iceberg_prob", "sim_iceberg_side", "sim_iceberg_strength",
                    "sim_iceberg_replenish", "sim_iceberg_stealth"]:
            out[col] = np.float32(0.0)
        return out

    mbo_ts = m["ts_event"].to_numpy(dtype="datetime64[ns]")
    bar_ts = out["ts_event"].to_numpy(dtype="datetime64[ns]")
    freq_ns = int(pd.to_timedelta(freq).total_seconds() * 1e9)

    # wall_persist و wall_consumed من المحاكي السابق
    wp = pd.to_numeric(out.get("sim_wall_persist", 0.5), errors="coerce").fillna(0.5).to_numpy()
    wc = pd.to_numeric(out.get("sim_wall_consumed", 0.0), errors="coerce").fillna(0.0).to_numpy()

    raw = {k: np.zeros(len(out), dtype=np.float32) for k in
           ["prob", "side", "strength", "replenish", "stealth"]}

    n_analyzed = 0
    for i in range(len(out)):
        t0 = bar_ts[i]
        t1 = t0 + np.timedelta64(freq_ns, "ns")
        lo = int(np.searchsorted(mbo_ts, t0, side="left"))
        hi = int(np.searchsorted(mbo_ts, t1, side="left"))
        if hi <= lo:
            continue
        n_analyzed += 1
        slc = m.iloc[lo:hi]
        res = _analyze_bar(slc, float(wp[i]), float(wc[i]))
        for k, v in res.items():
            raw[k][i] = v

    # ── normalization تكيفي ──
    out["sim_iceberg_prob"] = _rank(
        pd.Series(raw["prob"], index=out.index), window
    ).astype(np.float32)
    out["sim_iceberg_side"]      = raw["side"]
    out["sim_iceberg_strength"]  = _rank(
        pd.Series(raw["strength"], index=out.index), window
    ).astype(np.float32)
    out["sim_iceberg_replenish"] = raw["replenish"]
    out["sim_iceberg_stealth"]   = raw["stealth"]

    print(f"  ✅ حُلِّلت {n_analyzed}/{len(out)} شمعة عبر 4 طبقات كشف")
    return out


# قائمة المخرجات
ICEBERG_OUTPUTS = [
    "sim_iceberg_prob",
    "sim_iceberg_side",
    "sim_iceberg_strength",
    "sim_iceberg_replenish",
    "sim_iceberg_stealth",
]


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Institutional Iceberg Detector")
    ap.add_argument("--bars",   required=True, help="parquet الشمعات (مع wall_persist/consumed)")
    ap.add_argument("--mbo",    required=True, help="MBO خام (csv/parquet)")
    ap.add_argument("--output", default=None,  help="parquet المخرج")
    ap.add_argument("--freq",   default="5min")
    ap.add_argument("--window", type=int, default=200)
    args = ap.parse_args()

    bars = pd.read_parquet(args.bars)
    ext = Path(args.mbo).suffix.lower()
    mbo = pd.read_csv(args.mbo, low_memory=False) if ext == ".csv" \
          else pd.read_parquet(args.mbo)
    print(f"📥 شمعات: {len(bars):,} | MBO: {len(mbo):,}")

    bars = simulate_iceberg(bars, mbo, freq=args.freq, window=args.window)

    out_path = args.output or str(
        Path(args.bars).with_name(Path(args.bars).stem + "_ice.parquet")
    )
    bars.to_parquet(out_path, index=False)
    print(f"\n💾 محفوظ: {out_path}")
    print(f"   5 مخرجات iceberg: {ICEBERG_OUTPUTS}")


if __name__ == "__main__":
    main()
