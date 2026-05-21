"""
session_mapper.py — المستوى ③ من نظام اكتشاف الـ Edge
══════════════════════════════════════════════════════════════════════

الدور: يضع كل شمعة في إطارها الزمني والسعري الصحيح.

يضيف:
  ① zone        — أي جلسة/تداخل (20 zone)
  ② quarter     — الربع داخل الجلسة (Q1/Q2/Q3)
  ③ المستويات   — 10 مستويات مرجعية (يومي/أسبوعي/جلسي)
  ④ dist_*      — المسافة لكل مستوى (بـ ATR)
  ⑤ *_event     — حدث المستوى (touch/break/reject/failed_break/near)

كل الحسابات سببية (لا lookahead):
  - المستويات اليومية/الأسبوعية: من فترة سابقة منتهية
  - المستويات الجلسية: running max/min أثناء الجلسة فقط

الاستخدام:
  from session_mapper import map_sessions
  df = map_sessions(df)

CLI:
  python session_mapper.py --input sim_features.parquet --output mapped.parquet
"""

from __future__ import annotations

import argparse
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — تعريف الـ Zones (20 zone)
# ══════════════════════════════════════════════════════════════════

# الجلسات الأساسية: (بداية, نهاية) UTC
BASE_SESSIONS = {
    "asia":   (time(0, 0),  time(7, 0)),
    "london": (time(7, 0),  time(13, 0)),
    "ny":     (time(16, 0), time(20, 0)),
}

# التداخلات: zone مستقل بخصائصه
OVERLAP_ZONES = {
    "asia_london": (time(7, 0),  time(8, 0)),
    "london_ny":   (time(13, 0), time(16, 0)),
}


def _time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _assign_zone_and_quarter(ts: pd.Series) -> pd.DataFrame:
    """
    لكل طابع زمني يحدد:
      zone        — اسم الجلسة/التداخل
      quarter     — Q1/Q2/Q3 داخل الجلسة
      zone_full   — zone + quarter (e.g. 'london_q2')

    التداخلات لها أولوية: شمعة في london_ny تُعلَّم بالتداخل
    وليس بـ london أو ny.
    """
    minutes = ts.dt.hour * 60 + ts.dt.minute

    zone     = pd.Series("off_session", index=ts.index, dtype="object")
    quarter  = pd.Series("", index=ts.index, dtype="object")

    # ── الجلسات الأساسية ──
    for name, (start, end) in BASE_SESSIONS.items():
        s = _time_to_minutes(start)
        e = _time_to_minutes(end)
        mask = (minutes >= s) & (minutes < e)
        zone.loc[mask] = name
        # تقسيم لأرباع
        span = e - s
        q_size = span / 3.0
        rel = (minutes - s) / q_size
        q = np.where(rel < 1, "q1", np.where(rel < 2, "q2", "q3"))
        quarter.loc[mask] = pd.Series(q, index=ts.index)[mask]

    # ── التداخلات (أولوية أعلى — تكتب فوق الأساسية) ──
    for name, (start, end) in OVERLAP_ZONES.items():
        s = _time_to_minutes(start)
        e = _time_to_minutes(end)
        mask = (minutes >= s) & (minutes < e)
        zone.loc[mask] = name
        span = e - s
        q_size = span / 3.0
        rel = (minutes - s) / q_size
        q = np.where(rel < 1, "q1", np.where(rel < 2, "q2", "q3"))
        quarter.loc[mask] = pd.Series(q, index=ts.index)[mask]

    zone_full = zone.where(quarter == "", zone + "_" + quarter)

    return pd.DataFrame({
        "zone": zone,
        "quarter": quarter,
        "zone_full": zone_full,
    }, index=ts.index)


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — المستويات المرجعية (10)
# ══════════════════════════════════════════════════════════════════

def _add_daily_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    PDH / PDL / PD_50 — من يوم أمس الكامل (سببي).
    PD_50 = منتصف مدى أمس = (PDH + PDL) / 2
    """
    out = df.copy()
    ts  = pd.to_datetime(out["ts_event"])
    day = ts.dt.normalize()

    high = pd.to_numeric(out["high"], errors="coerce")
    low  = pd.to_numeric(out["low"],  errors="coerce")

    daily_hi = pd.DataFrame({"d": day, "h": high}).groupby("d")["h"].max()
    daily_lo = pd.DataFrame({"d": day, "l": low}).groupby("d")["l"].min()

    # shift(1) = يوم أمس
    pdh_map = daily_hi.shift(1).to_dict()
    pdl_map = daily_lo.shift(1).to_dict()

    out["PDH"] = day.map(pdh_map).astype(float)
    out["PDL"] = day.map(pdl_map).astype(float)
    out["PD_50"] = (out["PDH"] + out["PDL"]) / 2.0
    return out


def _add_weekly_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    PWH / PWL / PW_50 — من الأسبوع السابق الكامل (سببي).
    """
    out = df.copy()
    ts  = pd.to_datetime(out["ts_event"])
    # رقم الأسبوع ISO
    week = ts.dt.isocalendar().year.astype(str) + "_" + \
           ts.dt.isocalendar().week.astype(str).str.zfill(2)

    high = pd.to_numeric(out["high"], errors="coerce")
    low  = pd.to_numeric(out["low"],  errors="coerce")

    wk_hi = pd.DataFrame({"w": week, "h": high}).groupby("w")["h"].max()
    wk_lo = pd.DataFrame({"w": week, "l": low}).groupby("w")["l"].min()

    # ترتيب الأسابيع ثم shift
    wk_hi_sorted = wk_hi.sort_index()
    wk_lo_sorted = wk_lo.sort_index()
    pwh_map = wk_hi_sorted.shift(1).to_dict()
    pwl_map = wk_lo_sorted.shift(1).to_dict()

    out["PWH"] = week.map(pwh_map).astype(float)
    out["PWL"] = week.map(pwl_map).astype(float)
    out["PW_50"] = (out["PWH"] + out["PWL"]) / 2.0
    return out


def _add_session_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asia/London/NY High & Low — running max/min أثناء الجلسة (سببي).

    مهم: المستوى يُبنى تدريجياً.
    شمعة في منتصف آسيا ترى فقط قمة آسيا حتى تلك اللحظة.
    بعد انتهاء الجلسة، المستوى يُجمَّد ويُحمَل لباقي اليوم.
    """
    out = df.copy()
    ts  = pd.to_datetime(out["ts_event"])
    day = ts.dt.normalize()
    minutes = ts.dt.hour * 60 + ts.dt.minute

    high = pd.to_numeric(out["high"], errors="coerce")
    low  = pd.to_numeric(out["low"],  errors="coerce")

    for sess, (start, end) in BASE_SESSIONS.items():
        s = _time_to_minutes(start)
        e = _time_to_minutes(end)
        in_sess = (minutes >= s) & (minutes < e)

        hi_col = f"{sess}_H"
        lo_col = f"{sess}_L"
        out[hi_col] = np.nan
        out[lo_col] = np.nan

        # running max/min داخل الجلسة، لكل يوم
        # ⚠️ سببي: نستخدم shift(1) لأن المستوى يجب أن يعكس
        #         "أعلى/أقل سعر *قبل* الشمعة الحالية" — لا يضم الشمعة نفسها
        for d, idx in day.groupby(day).groups.items():
            sess_idx = [i for i in idx if in_sess.loc[i]]
            if not sess_idx:
                continue
            sess_idx = sorted(sess_idx)
            # cummax/cummin ثم shift(1) — السببية الصارمة
            h_run = high.loc[sess_idx].cummax().shift(1)
            l_run = low.loc[sess_idx].cummin().shift(1)
            out.loc[sess_idx, hi_col] = h_run.values
            out.loc[sess_idx, lo_col] = l_run.values

            # بعد انتهاء الجلسة: جمّد آخر قيمة (بعد shift) لباقي اليوم
            after_idx = [i for i in idx if i > sess_idx[-1]]
            if after_idx:
                # القيمة المُجمَّدة = أعلى الجلسة الكاملة (آمن لأنه بعد انتهائها)
                full_max = high.loc[sess_idx].max()
                full_min = low.loc[sess_idx].min()
                out.loc[after_idx, hi_col] = full_max
                out.loc[after_idx, lo_col] = full_min

    return out


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — المسافات وأحداث المستوى
# ══════════════════════════════════════════════════════════════════

LEVEL_NAMES = [
    "PDH", "PDL", "PD_50",
    "PWH", "PWL", "PW_50",
    "asia_H", "asia_L",
    "london_H", "london_L",
    "ny_H", "ny_L",
]


def _add_level_distances_and_events(
    df: pd.DataFrame,
    near_atr: float = 0.5,
    touch_atr: float = 0.15,
) -> pd.DataFrame:
    """
    لكل مستوى من LEVEL_NAMES يضيف:
      dist_to_<level>     — (close - level) / atr   [موجب=فوق المستوى]
      <level>_event       — حدث المستوى لهذه الشمعة

    أحداث المستوى:
      'touch'        — الشمعة لمست المستوى (high/low عبره) والإغلاق قريب
      'break'        — الإغلاق عبر المستوى (كسر مؤكد)
      'reject'       — الشمعة لمست المستوى لكن الإغلاق ارتد بعيداً
      'failed_break' — الشمعة الماضية كسرت، هذه رجعت = كسر فاشل
      'near'         — قريب من المستوى (ضمن near_atr) دون لمس
      ''             — بعيد عن المستوى
    """
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    high  = pd.to_numeric(out["high"],  errors="coerce")
    low   = pd.to_numeric(out["low"],   errors="coerce")
    # ATR fallback: لو atr_14 موجود استخدمه، وإلا احسب من المدى
    if "atr_14" in out.columns:
        atr = pd.to_numeric(out["atr_14"], errors="coerce")
        if atr.isna().all():
            atr = (high - low).rolling(14, min_periods=1).mean()
    else:
        atr = (high - low).rolling(14, min_periods=1).mean()
    atr = atr.clip(lower=1e-9)

    for lvl in LEVEL_NAMES:
        if lvl not in out.columns:
            continue
        level = pd.to_numeric(out[lvl], errors="coerce")

        # ── المسافة ──
        dist = (close - level) / atr
        out[f"dist_to_{lvl}"] = dist.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-30, 30)

        # ── أحداث المستوى ──
        # هل الشمعة لمست المستوى؟ (high فوقه و low تحته، أو قريب جداً)
        crossed   = (high >= level) & (low <= level)
        close_above = close > level
        close_below = close < level

        # touch: عبرت المستوى والإغلاق قريب منه
        is_touch = crossed & (dist.abs() <= touch_atr)

        # break: الإغلاق عبر المستوى بوضوح + الشمعة الماضية كانت الجانب الآخر
        prev_close = close.shift(1)
        broke_up   = (prev_close <= level) & (close > level) & (dist.abs() > touch_atr)
        broke_down = (prev_close >= level) & (close < level) & (dist.abs() > touch_atr)
        is_break   = broke_up | broke_down

        # reject: لمست المستوى لكن الإغلاق ارتد بعيداً
        rejected_up   = crossed & close_below & ((level - close) / atr > touch_atr)
        rejected_down = crossed & close_above & ((close - level) / atr > touch_atr)
        is_reject = rejected_up | rejected_down

        # failed_break: الشمعة الماضية كسرت، هذه رجعت للجانب الأصلي
        prev_broke_up   = broke_up.shift(1).fillna(False)
        prev_broke_down = broke_down.shift(1).fillna(False)
        failed = (
            (prev_broke_up & (close < level)) |
            (prev_broke_down & (close > level))
        )

        # near: قريب دون لمس
        is_near = (~crossed) & (dist.abs() <= near_atr)

        # دمج الأحداث — الأولوية: failed > break > reject > touch > near
        event = pd.Series("", index=out.index, dtype="object")
        event = event.mask(is_near,   "near")
        event = event.mask(is_touch,  "touch")
        event = event.mask(is_reject, "reject")
        event = event.mask(is_break,  "break")
        event = event.mask(failed,    "failed_break")

        out[f"{lvl}_event"] = event

    return out


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — المُشغّل الكامل
# ══════════════════════════════════════════════════════════════════

def map_sessions(
    df: pd.DataFrame,
    near_atr: float = 0.5,
    touch_atr: float = 0.15,
) -> pd.DataFrame:
    """
    يضيف كل أعمدة الإطار الزمني والسعري.

    المخرجات الجديدة:
      zone, quarter, zone_full
      12 مستوى (PDH..ny_L)
      12 × dist_to_<level>
      12 × <level>_event
    """
    df = df.sort_values("ts_event").reset_index(drop=True)
    ts = pd.to_datetime(df["ts_event"])

    print("🗺️  Session Mapper — بناء الإطار...")

    # ① zones + quarters
    zq = _assign_zone_and_quarter(ts)
    df = pd.concat([df, zq], axis=1)
    n_zones = df["zone_full"].nunique()
    print(f"  ✅ Zones: {n_zones} منطقة زمنية فريدة")

    # ② المستويات
    df = _add_daily_levels(df)
    df = _add_weekly_levels(df)
    df = _add_session_levels(df)
    n_levels = sum(1 for c in LEVEL_NAMES if c in df.columns)
    print(f"  ✅ Levels: {n_levels} مستوى مرجعي")

    # ③ المسافات والأحداث
    df = _add_level_distances_and_events(df, near_atr=near_atr, touch_atr=touch_atr)

    # تقرير الأحداث
    total_events = 0
    for lvl in LEVEL_NAMES:
        ev_col = f"{lvl}_event"
        if ev_col in df.columns:
            total_events += (df[ev_col] != "").sum()
    print(f"  ✅ Level events: {total_events:,} حدث (touch/break/reject/failed/near)")

    return df


# قائمة الأعمدة المُنتَجة — يستخدمها edge_scanner
def get_mapper_columns() -> dict:
    return {
        "zone_cols":  ["zone", "quarter", "zone_full"],
        "level_cols": list(LEVEL_NAMES),
        "dist_cols":  [f"dist_to_{l}" for l in LEVEL_NAMES],
        "event_cols": [f"{l}_event" for l in LEVEL_NAMES],
    }


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Session Mapper — Edge System Layer 3")
    ap.add_argument("--input",  required=True, help="parquet المدخل (sim_features)")
    ap.add_argument("--output", default=None,  help="parquet المخرج")
    ap.add_argument("--near-atr",  type=float, default=0.5,
                    help="عتبة 'near' بوحدات ATR")
    ap.add_argument("--touch-atr", type=float, default=0.15,
                    help="عتبة 'touch' بوحدات ATR")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    print(f"📥 {len(df):,} rows | {len(df.columns)} cols")

    df = map_sessions(df, near_atr=args.near_atr, touch_atr=args.touch_atr)

    out_path = args.output or str(
        Path(args.input).with_name(Path(args.input).stem + "_mapped.parquet")
    )
    df.to_parquet(out_path, index=False)
    print(f"\n💾 محفوظ: {out_path}")
    print(f"   إجمالي الأعمدة: {len(df.columns)}")


if __name__ == "__main__":
    main()
