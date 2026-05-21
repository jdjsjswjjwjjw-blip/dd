#!/usr/bin/env python3
"""
diagnose_label_timeout_daytrade.py

تشخيص صفوف timeout (path_outcome=4) تحت is_event، بدون تعديل prepare_day_trading.

1) تقرير ملاحظ (Observed): إن وُجدت path_outcome و neutral_reason في الـ parquet.
2) محاكاة مضادة (Counterfactual): يعيد حساب نفس منطق First-Barrier كـ baseline_regime_only
   مع شبكة من:
     - مضاعف max_bars لكل regime (لا يغيّر أرقام TP/SL من REGIME_TP_SL)
     - إضافة ساعات لنهاية كل جلسة في خريطة الجلسات (اختبار «قطع الجلسة مبكرًا»)
     - kalman_event_floor

3) تقاطع (Intersection):
     - للـ timeouts المخزنة: تقاطع neutral_reason المخزن × هل كان مسموحًا بالطرفين عند الدخول
       (both_sides_allowed = أفق حقيقي محتمل مقابل veto جانب).
     - مقارنة صفًا بصف بين الليبل المخزن ومحاكاة مرجعية (نفس max_bars/session/kalman).

متطلبات parquet للمحاكاة:
  ts_event, close, high, low, atr_14, is_event, event_score,
  regime_label, session (مستحسن), kalman_direction, event_direction, open (اختياري)

مثال:
  py -3 diagnose_label_timeout_daytrade.py --parquet PATH \\
      --max-bars-mult-grid "1.0,1.5,2.0" \\
      --session-extend-hours-grid "0,2" \\
      --kalman-floor-grid "0.7,0.85" \\
      --out-intersection-prefix "E:\\\\...\\\\daytrade\\\\label_timeout_ix"

يُنشئ ملفين: *_timeout_entry_crosstab.csv و *_sim_vs_obs_events.csv
"""

from __future__ import annotations

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd

_SESSION_END_HOUR_BASE: dict[str, int] = {
    "asia": 8,
    "london": 16,
    "overlap": 16,
    "ny": 21,
}

NEUTRAL_REASON_NONE = 0
NEUTRAL_REASON_TIMEOUT = 1
NEUTRAL_REASON_LONG_SL = 2
NEUTRAL_REASON_SHORT_SL = 3
NEUTRAL_REASON_KALMAN = 4
NEUTRAL_REASON_WEAK_EVENT = 5

DIR_LONG = 0
DIR_SHORT = 1
DIR_NEUTRAL = 2

_NEUTRAL_NAMES = {
    NEUTRAL_REASON_NONE: "none",
    NEUTRAL_REASON_TIMEOUT: "timeout_horizon",
    NEUTRAL_REASON_LONG_SL: "long_sl",
    NEUTRAL_REASON_SHORT_SL: "short_sl",
    NEUTRAL_REASON_KALMAN: "kalman_veto",
    NEUTRAL_REASON_WEAK_EVENT: "weak_event_non_row",
}

_PATH_NAMES = {
    0: "long_tp",
    1: "short_tp",
    2: "long_sl",
    3: "short_sl",
    4: "timeout",
    5: "weak_long",
    6: "weak_short",
}

try:
    from regime_config import REGIME_TP_SL, REGIME_MAX_BARS
except ImportError:
    REGIME_TP_SL = {"trending": (2.0, 1.0), "ranging": (0.8, 0.6), "volatile": (1.5, 1.5)}
    REGIME_MAX_BARS = {"trending": 12, "ranging": 6, "volatile": 3}


def _session_hours_with_extend(extend_hours: int) -> dict[str, int]:
    return {k: min(int(v) + int(extend_hours), 23) for k, v in _SESSION_END_HOUR_BASE.items()}


def _parse_float_grid(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_grid(s: str) -> list[int]:
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def direction_allow_masks(df: pd.DataFrame, kalman_event_floor: float) -> tuple[np.ndarray, np.ndarray]:
    """
    نفس قواعد السماح بالـ long/short عند الدخلة لصف is_event في prepare_day_trading.
    للصفوف غير الأحداث يُعاد True, True (لا يُستخدم في التقارير).
    """
    n = len(df)
    allow_long = np.ones(n, dtype=bool)
    allow_short = np.ones(n, dtype=bool)
    has_event = "is_event" in df.columns
    is_ev_arr = df["is_event"].to_numpy(dtype=np.int8) if has_event else np.zeros(n, dtype=np.int8)
    ev_score_src = df["event_score"] if "event_score" in df.columns else pd.Series(0.0, index=df.index)
    kalman_src = df["kalman_direction"] if "kalman_direction" in df.columns else pd.Series(0, index=df.index)
    event_dir_src = df["event_direction"] if "event_direction" in df.columns else pd.Series(0, index=df.index)
    event_score_arr = pd.to_numeric(ev_score_src, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    kalman_dir_arr = pd.to_numeric(kalman_src, errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    event_dir_arr = pd.to_numeric(event_dir_src, errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    kf = float(kalman_event_floor)

    for i in range(n):
        if not is_ev_arr[i]:
            continue
        event_score_i = float(event_score_arr[i])
        kalman_dir_i = int(kalman_dir_arr[i])
        event_dir_i = int(event_dir_arr[i])
        al = True
        ash = True
        if event_dir_i > 0:
            ash = False
        elif event_dir_i < 0:
            al = False
        else:
            if kalman_dir_i == 1 and event_score_i < kf:
                ash = False
            elif kalman_dir_i == -1 and event_score_i < kf:
                al = False
        allow_long[i] = al
        allow_short[i] = ash
    return allow_long, allow_short


def veto_source_at_entry(event_dir_i: int, kalman_dir_i: int, event_score_i: float, kalman_floor: float) -> str:
    """
    تصنيف سبب منع أحد الطرفين عند شريطة الدخول (نفس ترتيب prepare_day_trading).
    إذا لم يُمنع طرف returns both_sides_allowed.
    """
    kf = float(kalman_floor)
    if event_dir_i > 0:
        return "hard_event_direction_blocks_short"
    if event_dir_i < 0:
        return "hard_event_direction_blocks_long"
    if kalman_dir_i == 1 and event_score_i < kf:
        return "kalman_event_floor_blocks_short"
    if kalman_dir_i == -1 and event_score_i < kf:
        return "kalman_event_floor_blocks_long"
    return "both_sides_allowed"


def print_timeout_intersection_observed(df: pd.DataFrame, *, kalman_floor: float) -> pd.DataFrame | None:
    """تقاطع: timeouts المخزنة × تصنيف الدخول (طرفان مسموحان vs جانب ممنوع). يعيد جدول Crosstab أو None."""
    need = {"is_event", "path_outcome", "neutral_reason"}
    if need - set(df.columns):
        print("\n[intersection/observed] skip — missing columns for stored labels")
        return None

    ie = pd.to_numeric(df["is_event"], errors="coerce").fillna(0).astype(np.int8).to_numpy() == 1
    po = pd.to_numeric(df["path_outcome"], errors="coerce").fillna(4).astype(np.int8).to_numpy()
    nr = pd.to_numeric(df["neutral_reason"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    to_mask = ie & (po == 4)
    n_to = int(to_mask.sum())
    if n_to == 0:
        print("\n[intersection/observed] no stored timeouts among is_event — empty crosstab")
        return None

    allow_long, allow_short = direction_allow_masks(df, kalman_floor)
    both_allowed = allow_long & allow_short
    entry_kind = np.where(both_allowed[to_mask], "both_sides_allowed", "one_side_blocked_veto")
    stored_kind = np.array([_NEUTRAL_NAMES.get(int(x), str(int(x))) for x in nr[to_mask]])

    ct = pd.crosstab(
        pd.Series(stored_kind, name="stored_neutral_reason"),
        pd.Series(entry_kind, name="entry_allowance_at_bar_i"),
        margins=True,
    )
    print("\n[intersection/observed] Timeout rows (path_outcome=4 & is_event): stored_reason x entry_allowance")
    print(f"  (allow_long/allow_short recomputed with kalman_event_floor={kalman_floor})")
    print(ct.to_string())
    return ct


def build_sim_vs_observed_comparison(
    df: pd.DataFrame,
    *,
    max_bars_mult: float,
    session_extend_hours: int,
    kalman_event_floor: float,
    compare_kalman_for_allow: float,
) -> pd.DataFrame | None:
    """صفًا بصف: أحداث فقط — الليبل المخزن مقابل محاكاة مرجعية."""
    if "path_outcome" not in df.columns:
        print("\n[intersection/sim_vs_observed] skip — no path_outcome in parquet")
        return None

    sh = _session_hours_with_extend(session_extend_hours)
    sim = run_barrier_baseline(
        df,
        max_bars_mult=max_bars_mult,
        session_end_hours=sh,
        kalman_event_floor=kalman_event_floor,
    )

    ie = pd.to_numeric(df["is_event"], errors="coerce").fillna(0).astype(np.int8).to_numpy() == 1
    allow_long, allow_short = direction_allow_masks(df, compare_kalman_for_allow)
    both_allowed = allow_long & allow_short

    obs_po = pd.to_numeric(df["path_outcome"], errors="coerce").fillna(-1).astype(np.int16).to_numpy()
    obs_nr = pd.to_numeric(df["neutral_reason"], errors="coerce").fillna(-1).astype(np.int16).to_numpy() if "neutral_reason" in df.columns else np.full(len(df), -1, dtype=np.int16)

    ev_score_src = df["event_score"] if "event_score" in df.columns else pd.Series(0.0, index=df.index)
    kalman_src = df["kalman_direction"] if "kalman_direction" in df.columns else pd.Series(0, index=df.index)
    event_dir_src = df["event_direction"] if "event_direction" in df.columns else pd.Series(0, index=df.index)
    event_score_arr = pd.to_numeric(ev_score_src, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    kalman_dir_arr = pd.to_numeric(kalman_src, errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    event_dir_arr = pd.to_numeric(event_dir_src, errors="coerce").fillna(0).to_numpy(dtype=np.int8)

    rows = []
    idx_ev = np.flatnonzero(ie)
    for i in idx_ev:
        spo = int(sim.iloc[i]["path_outcome"])
        snr = int(sim.iloc[i]["neutral_reason"])
        opo = int(obs_po[i])
        onr = int(obs_nr[i])
        edi = int(event_dir_arr[i])
        kdi = int(kalman_dir_arr[i])
        esi = float(event_score_arr[i])
        vsrc = veto_source_at_entry(edi, kdi, esi, compare_kalman_for_allow)
        rows.append(
            {
                "row_idx": i,
                "ts_event": df["ts_event"].iloc[i],
                "event_direction": edi,
                "kalman_direction": kdi,
                "event_score": esi,
                "veto_source_at_entry": vsrc,
                "both_sides_allowed_entry": bool(both_allowed[i]),
                "obs_path_outcome": opo,
                "obs_path_name": _PATH_NAMES.get(opo, str(opo)),
                "sim_path_outcome": spo,
                "sim_path_name": _PATH_NAMES.get(spo, str(spo)),
                "obs_neutral_reason": onr,
                "obs_neutral_name": _NEUTRAL_NAMES.get(onr, str(onr)) if onr >= 0 else "missing",
                "sim_neutral_reason": snr,
                "sim_neutral_name": _NEUTRAL_NAMES.get(snr, str(snr)),
                "agree_path": opo == spo,
                "agree_timeout_flag": (opo == 4) == (spo == 4),
            }
        )

    cmp_df = pd.DataFrame(rows)
    n = len(cmp_df)
    if n == 0:
        return cmp_df

    print("\n[intersection/sim_vs_observed] Event rows only - agreement with reference simulation")
    print(
        f"  ref_sim: max_bars_mult={max_bars_mult}, session_extend_hours={session_extend_hours}, "
        f"kalman_event_floor={kalman_event_floor}"
    )
    print(f"  both_sides_allowed uses kalman_floor={compare_kalman_for_allow} (for column both_sides_allowed_entry)")
    print(f"  agree_path (full path_outcome): {float(cmp_df['agree_path'].mean()):.2%} ({int(cmp_df['agree_path'].sum())}/{n})")
    print(f"  agree_timeout_flag:           {float(cmp_df['agree_timeout_flag'].mean()):.2%}")

    print("\n[veto_source_at_entry] Among all is_event rows:")
    for name, c in cmp_df["veto_source_at_entry"].value_counts().items():
        print(f"    {name}: {int(c)}")

    obs_to = cmp_df["obs_path_outcome"] == 4
    if obs_to.any():
        print("\n[veto_source_at_entry] Among rows where OBSERVED path_outcome==timeout only:")
        for name, c in cmp_df.loc[obs_to, "veto_source_at_entry"].value_counts().items():
            print(f"    {name}: {int(c)}")

    # Sub-analysis: stored timeouts vs sim
    if obs_to.any():
        sub = cmp_df.loc[obs_to]
        print("  Among rows where OBSERVED path_outcome==timeout:")
        print(f"    sim also timeout: {float(sub['sim_path_outcome'].eq(4).mean()):.2%}")
        print("    sim_path_name counts:")
        vc = sub["sim_path_name"].value_counts()
        for name, c in vc.head(10).items():
            print(f"      {name}: {int(c)}")

    ct2 = pd.crosstab(
        cmp_df["obs_path_name"],
        cmp_df["sim_path_name"],
        margins=True,
    )
    print("\n  Crosstab observed_path x simulated_path (events only):")
    print(ct2.to_string())

    return cmp_df


def run_barrier_baseline(
    df: pd.DataFrame,
    *,
    max_bars_mult: float = 1.0,
    session_end_hours: dict[str, int] | None = None,
    kalman_event_floor: float = 0.70,
    default_tp_mult: float = 1.5,
    default_sl_mult: float = 1.0,
    default_max_bars: int = 6,
    min_atr: float = 0.0003,
    weak_event_to_directional: bool = False,
    weak_event_min_move_atr: float = 0.35,
) -> pd.DataFrame:
    """يعيد أعمدة bias_label, path_outcome, neutral_reason, trade_duration (منطق prepare baseline)."""
    df = df.copy().sort_values("ts_event").reset_index(drop=True)
    n = len(df)
    sess_hours = session_end_hours if session_end_hours is not None else dict(_SESSION_END_HOUR_BASE)

    bias_label = np.full(n, DIR_NEUTRAL, dtype=np.int8)
    path_outcome = np.full(n, 4, dtype=np.int8)
    neutral_reason = np.full(n, NEUTRAL_REASON_NONE, dtype=np.int8)
    trade_duration = np.zeros(n, dtype=np.int16)
    signal_quality = np.zeros(n, dtype=np.int8)
    forward_return = np.zeros(n, dtype=np.float32)

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=np.float64)
    open_ = (
        pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=np.float64)
        if "open" in df.columns
        else close.copy()
    )
    atr = pd.to_numeric(df["atr_14"], errors="coerce").to_numpy(dtype=np.float64)

    has_regime = "regime_label" in df.columns
    has_session = "session" in df.columns
    has_event = "is_event" in df.columns

    regime_arr = df["regime_label"].astype(str).to_numpy() if has_regime else None
    session_arr = df["session"].astype(str).to_numpy() if has_session else None
    is_ev_arr = df["is_event"].to_numpy(dtype=np.int8) if has_event else np.ones(n, dtype=np.int8)
    ev_score_src = df["event_score"] if "event_score" in df.columns else pd.Series(0.0, index=df.index)
    kalman_src = df["kalman_direction"] if "kalman_direction" in df.columns else pd.Series(0, index=df.index)
    event_dir_src = df["event_direction"] if "event_direction" in df.columns else pd.Series(0, index=df.index)
    event_score_arr = pd.to_numeric(ev_score_src, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    kalman_dir_arr = pd.to_numeric(kalman_src, errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    event_dir_arr = pd.to_numeric(event_dir_src, errors="coerce").fillna(0).to_numpy(dtype=np.int8)

    london_ok = np.zeros(n, dtype=bool)
    for col in ("is_london", "is_overlap"):
        if col in df.columns:
            london_ok |= pd.to_numeric(df[col], errors="coerce").fillna(0).astype(bool).to_numpy()

    ts_arr = df["ts_event"].to_numpy()
    mb_mult = float(max(max_bars_mult, 0.05))

    def _session_end(i: int, max_bars_r: int) -> int:
        cap = min(i + max_bars_r, n - 1)
        if session_arr is None:
            return cap
        sess = session_arr[i]
        end_hour = sess_hours.get(sess, 21)
        for k in range(i + 1, cap + 1):
            ts = ts_arr[k]
            try:
                hour = pd.Timestamp(ts).hour
            except Exception:
                continue
            if hour >= end_hour:
                return k - 1
        return cap

    for i in range(n):
        regime_i = str(regime_arr[i]) if has_regime else "ranging"
        base_mb = int(REGIME_MAX_BARS.get(regime_i, default_max_bars))
        max_bars_r = max(1, int(round(base_mb * mb_mult)))
        tp_mult, sl_mult = REGIME_TP_SL.get(regime_i, (default_tp_mult, default_sl_mult))

        atr_i = max(float(atr[i]), float(min_atr))
        entry = float(close[i])
        sess_end = _session_end(i, max_bars_r)
        fwd_idx = min(i + max_bars_r, n - 1)
        forward_return[i] = float((close[fwd_idx] - entry) / max(entry, 1e-8))

        if not is_ev_arr[i]:
            if weak_event_to_directional:
                move_px = float(close[sess_end] - entry) if sess_end >= i else 0.0
                move_thr = float(max(weak_event_min_move_atr, 0.0)) * atr_i
                trade_duration[i] = max(0, sess_end - i)
                if move_px >= move_thr:
                    bias_label[i] = DIR_LONG
                    path_outcome[i] = 5
                    signal_quality[i] = 1 if london_ok[i] else 0
                    neutral_reason[i] = NEUTRAL_REASON_NONE
                elif move_px <= -move_thr:
                    bias_label[i] = DIR_SHORT
                    path_outcome[i] = 6
                    signal_quality[i] = 1 if london_ok[i] else 0
                    neutral_reason[i] = NEUTRAL_REASON_NONE
                else:
                    neutral_reason[i] = NEUTRAL_REASON_WEAK_EVENT
            else:
                neutral_reason[i] = NEUTRAL_REASON_WEAK_EVENT
            continue

        event_score_i = float(event_score_arr[i])
        kalman_dir_i = int(kalman_dir_arr[i])
        event_dir_i = int(event_dir_arr[i])

        allow_long = True
        allow_short = True
        if event_dir_i > 0:
            allow_short = False
        elif event_dir_i < 0:
            allow_long = False
        else:
            if kalman_dir_i == 1 and event_score_i < float(kalman_event_floor):
                allow_short = False
            elif kalman_dir_i == -1 and event_score_i < float(kalman_event_floor):
                allow_long = False

        tp_dist = float(tp_mult) * atr_i
        sl_dist = float(sl_mult) * atr_i

        tp_long = entry + tp_dist
        sl_long = entry - sl_dist
        tp_short = entry - tp_dist
        sl_short = entry + sl_dist

        first_ev: tuple[str, int] | None = None

        for j in range(i + 1, sess_end + 1):
            h = float(high[j])
            l = float(low[j])
            o = float(open_[j]) if np.isfinite(open_[j]) else 0.5 * (h + l)

            lt = h >= tp_long
            ls = l <= sl_long
            st = l <= tp_short
            ss = h >= sl_short

            if not allow_long:
                lt = ls = False
            if not allow_short:
                st = ss = False

            picked: str | None = None
            if lt and ls:
                d_tp = abs(tp_long - o)
                d_sl = abs(sl_long - o)
                picked = "long_tp" if d_tp <= d_sl else "long_sl"
            elif lt:
                picked = "long_tp"
            elif ls:
                picked = "long_sl"

            if picked is None:
                if st and ss:
                    d_tp = abs(tp_short - o)
                    d_sl = abs(sl_short - o)
                    picked = "short_tp" if d_tp <= d_sl else "short_sl"
                elif st:
                    picked = "short_tp"
                elif ss:
                    picked = "short_sl"

            if picked is not None:
                first_ev = (picked, j)
                break

        if first_ev is None:
            bias_label[i] = DIR_NEUTRAL
            path_outcome[i] = 4
            trade_duration[i] = max(0, sess_end - i)
            neutral_reason[i] = NEUTRAL_REASON_KALMAN if ((not allow_long) or (not allow_short)) else NEUTRAL_REASON_TIMEOUT
        else:
            ev_type, ev_j = first_ev
            trade_duration[i] = ev_j - i
            sq = 2 if london_ok[i] else 1

            if ev_type == "long_tp":
                bias_label[i] = DIR_LONG
                path_outcome[i] = 0
                signal_quality[i] = sq
            elif ev_type == "short_tp":
                bias_label[i] = DIR_SHORT
                path_outcome[i] = 1
                signal_quality[i] = sq
            elif ev_type == "long_sl":
                bias_label[i] = DIR_NEUTRAL
                path_outcome[i] = 2
                neutral_reason[i] = NEUTRAL_REASON_LONG_SL
            elif ev_type == "short_sl":
                bias_label[i] = DIR_NEUTRAL
                path_outcome[i] = 3
                neutral_reason[i] = NEUTRAL_REASON_SHORT_SL

    out = pd.DataFrame(
        {
            "bias_label": bias_label,
            "path_outcome": path_outcome,
            "neutral_reason": neutral_reason,
            "trade_duration": trade_duration,
            "signal_quality": signal_quality,
            "forward_return": forward_return,
        }
    )
    return out


def _metrics_among_events(df_src: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    is_ev = pd.to_numeric(df_src["is_event"], errors="coerce").fillna(0).astype(np.int8).to_numpy() == 1
    po = labels["path_outcome"].to_numpy()
    nr = labels["neutral_reason"].to_numpy()
    n_ev = int(is_ev.sum())
    if n_ev == 0:
        return {"n_is_event": 0}

    ev_po = po[is_ev]
    ev_nr = nr[is_ev]
    ev_td = labels["trade_duration"].to_numpy()[is_ev]
    timeout_m = ev_po == 4
    n_to = int(timeout_m.sum())

    return {
        "n_is_event": n_ev,
        "among_events_timeout_share": float(np.mean(timeout_m)),
        "among_events_tp_share": float(np.mean((ev_po == 0) | (ev_po == 1))),
        "among_events_sl_share": float(np.mean((ev_po == 2) | (ev_po == 3))),
        "among_events_timeout_then_kalman_share": float(np.mean(timeout_m & (ev_nr == NEUTRAL_REASON_KALMAN))),
        "among_events_timeout_then_horizon_share": float(np.mean(timeout_m & (ev_nr == NEUTRAL_REASON_TIMEOUT))),
        "among_events_mean_trade_dur": float(labels["trade_duration"].to_numpy()[is_ev].mean()),
        "among_events_mean_trade_dur_if_timeout": float(ev_td[timeout_m].mean()) if n_to else 0.0,
    }


def print_observed(df: pd.DataFrame) -> None:
    need = {"is_event", "path_outcome", "neutral_reason"}
    miss = need - set(df.columns)
    if miss:
        print(f"[observed] skip — parquet missing columns: {sorted(miss)}")
        return

    ie = pd.to_numeric(df["is_event"], errors="coerce").fillna(0).astype(np.int8).to_numpy() == 1
    po = pd.to_numeric(df["path_outcome"], errors="coerce").fillna(4).astype(np.int8).to_numpy()
    nr = pd.to_numeric(df["neutral_reason"], errors="coerce").fillna(0).astype(np.int8).to_numpy()

    n_ev = int(ie.sum())
    to_mask = ie & (po == 4)
    n_to = int(to_mask.sum())

    print("\n[observed] Stored labels in parquet")
    print(f"  is_event count: {n_ev:,}")
    print(f"  timeout (path_outcome=4) among events: {n_to:,} ({n_to / max(n_ev, 1):.2%})")

    if n_to > 0:
        sub = df.loc[to_mask]
        vc_nr = pd.Series(nr[to_mask]).value_counts().sort_index()
        print("  neutral_reason among timeouts (1=horizon, 4=kalman veto):")
        for k, c in vc_nr.items():
            print(f"    {_NEUTRAL_NAMES.get(int(k), str(k))}: {int(c)}")

        if "regime_label" in df.columns:
            print("  regime_label (timeouts):")
            for k, c in sub["regime_label"].astype(str).value_counts().head(8).items():
                print(f"    {k}: {int(c)}")
        if "session" in df.columns:
            print("  session (timeouts):")
            for k, c in sub["session"].astype(str).value_counts().head(8).items():
                print(f"    {k}: {int(c)}")
        if "event_score" in df.columns:
            es_to = pd.to_numeric(sub["event_score"], errors="coerce")
            es_ev = pd.to_numeric(df.loc[ie, "event_score"], errors="coerce")
            print(f"  event_score timeouts: min/mean/max = {es_to.min():.3f} / {es_to.mean():.3f} / {es_to.max():.3f}")
            print(f"  event_score all events: min/mean/max = {es_ev.min():.3f} / {es_ev.mean():.3f} / {es_ev.max():.3f}")
        td = pd.to_numeric(sub["trade_duration"], errors="coerce") if "trade_duration" in df.columns else None
        if td is not None and len(td):
            print(f"  trade_duration (timeouts): mean={td.mean():.2f} median={td.median():.2f} max={td.max():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose timeout rows among is_event (observed + counterfactual sweep)")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-csv", default=None, help="Write sweep rows to CSV")
    ap.add_argument("--min-atr", type=float, default=0.0003)
    ap.add_argument("--observed-only", action="store_true", help="Only print observed breakdown from parquet")
    ap.add_argument("--skip-observed", action="store_true")
    ap.add_argument(
        "--max-bars-mult-grid",
        default="1.0,1.5,2.0",
        help="Multiply REGIME_MAX_BARS per regime (TP/SL unchanged)",
    )
    ap.add_argument(
        "--session-extend-hours-grid",
        default="0,2",
        help="Add N hours to every session end hour (capped at 23)",
    )
    ap.add_argument("--kalman-floor-grid", default="0.7,0.85")
    ap.add_argument(
        "--compare-kalman-floor",
        type=float,
        default=0.70,
        help="Kalman floor لإعادة حساب allow_long/allow_short في تقاطع الـ observed timeout",
    )
    ap.add_argument(
        "--ref-max-bars-mult",
        type=float,
        default=1.0,
        help="محاكاة مرجعية لمقارنة الليبل المخزن (تقاطع sim_vs_observed)",
    )
    ap.add_argument("--ref-session-extend-hours", type=int, default=0)
    ap.add_argument("--ref-kalman-floor", type=float, default=0.70)
    ap.add_argument("--skip-intersection", action="store_true", help="Do not print/save intersection blocks")
    ap.add_argument(
        "--out-intersection-prefix",
        default=None,
        help="If set, writes {prefix}_timeout_entry_crosstab.csv and {prefix}_sim_vs_obs_events.csv",
    )
    args = ap.parse_args()

    pq = os.path.abspath(args.parquet)
    if "..." in pq:
        raise SystemExit("Pass full parquet path (no ... placeholder).")
    if not os.path.isfile(pq):
        raise SystemExit(f"parquet not found: {pq}")

    sim_need = {
        "ts_event",
        "close",
        "high",
        "low",
        "atr_14",
        "is_event",
        "event_score",
    }
    df = pd.read_parquet(pq).sort_values("ts_event").reset_index(drop=True)
    miss_sim = sorted(sim_need - set(df.columns))
    if miss_sim:
        raise SystemExit(f"parquet missing columns for simulation: {miss_sim}")

    print("=" * 72)
    print("diagnose_label_timeout_daytrade")
    print(f"  parquet={pq}")
    print(f"  rows={len(df):,}")
    print(f"  REGIME_MAX_BARS={dict(REGIME_MAX_BARS)}")
    print(f"  base session end hours={_SESSION_END_HOUR_BASE}")
    print("=" * 72)

    if not args.skip_observed:
        print_observed(df)

    if not args.skip_intersection:
        ct_obs = print_timeout_intersection_observed(df, kalman_floor=args.compare_kalman_floor)
        cmp_tbl = build_sim_vs_observed_comparison(
            df,
            max_bars_mult=args.ref_max_bars_mult,
            session_extend_hours=args.ref_session_extend_hours,
            kalman_event_floor=args.ref_kalman_floor,
            compare_kalman_for_allow=args.compare_kalman_floor,
        )
        prefix = args.out_intersection_prefix
        if prefix:
            base = os.path.abspath(prefix)
            if ct_obs is not None:
                p1 = base + "_timeout_entry_crosstab.csv"
                ct_obs.to_csv(p1, encoding="utf-8-sig")
                print(f"\nWrote {p1}")
            if cmp_tbl is not None and len(cmp_tbl):
                p2 = base + "_sim_vs_obs_events.csv"
                cmp_tbl.to_csv(p2, index=False, encoding="utf-8-sig")
                print(f"Wrote {p2}")

    if args.observed_only:
        return

    mults = _parse_float_grid(args.max_bars_mult_grid)
    extends = _parse_int_grid(args.session_extend_hours_grid)
    kalman_fs = _parse_float_grid(args.kalman_floor_grid)
    n_combo = len(mults) * len(extends) * len(kalman_fs)
    if n_combo > 20_000:
        raise SystemExit(f"Too many combinations ({n_combo}); narrow grids.")

    rows_out: list[dict[str, Any]] = []
    print("\n[counterfactual] sweep (same TP/SL as regime_config; only horizon/session/kalman change)")
    for mb in mults:
        for ext in extends:
            sh = _session_hours_with_extend(ext)
            for kf in kalman_fs:
                lab = run_barrier_baseline(
                    df,
                    max_bars_mult=mb,
                    session_end_hours=sh,
                    kalman_event_floor=kf,
                    min_atr=args.min_atr,
                )
                m = _metrics_among_events(df, lab)
                row = {
                    "max_bars_mult": mb,
                    "session_extend_hours": ext,
                    "kalman_event_floor": kf,
                    **m,
                }
                rows_out.append(row)

    pdf = pd.DataFrame(rows_out)
    pdf = pdf.sort_values(["among_events_timeout_share", "among_events_tp_share"], ascending=[True, False])
    show_cols = [
        "max_bars_mult",
        "session_extend_hours",
        "kalman_event_floor",
        "among_events_timeout_share",
        "among_events_timeout_then_horizon_share",
        "among_events_timeout_then_kalman_share",
        "among_events_tp_share",
        "among_events_sl_share",
        "among_events_mean_trade_dur_if_timeout",
    ]
    show_cols = [c for c in show_cols if c in pdf.columns]
    print(pdf[show_cols].head(20).to_string(index=False))

    out_csv = args.out_csv
    if out_csv:
        pdf.to_csv(os.path.abspath(out_csv), index=False, encoding="utf-8-sig")
        print(f"\nWrote sweep CSV -> {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    main()
