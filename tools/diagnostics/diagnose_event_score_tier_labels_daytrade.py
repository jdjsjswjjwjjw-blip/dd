#!/usr/bin/env python3
"""
diagnose_event_score_tier_labels_daytrade.py

يقارن ليبلات First-Barrier (نفس منطق label_by_outcome في prepare_day_trading.py)
مرّتين على نفس parquet:

  • baseline — TP/SL و max_bars حسب regime فقط (كما في الإنتاج الحالي).
  • tiered — لصفوف is_event==1 فقط: يستبدل (tp_mult, sl_mult, max_bars) حسب شرائح
    event_score (أهداف أوسع وأفق أطول للأحداث «القوية»).

لا يعدّل ملفات الإنتاج؛ يطبع ملخصًا ويكتب CSV اختياريًا.

متطلبات باركيه DAYTRADE المُصنَّف مسبقًا:
  ts_event, close, high, low, atr_14, is_event, event_score
  + regime_label, session (مستحسن), kalman_direction, event_direction, open (اختياري)

استخدام:
  py -3 diagnose_event_score_tier_labels_daytrade.py --parquet PATH [--out-csv PATH]
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

_SESSION_END_HOUR: dict[str, int] = {
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

try:
    from regime_config import REGIME_TP_SL, REGIME_MAX_BARS
except ImportError:
    REGIME_TP_SL = {"trending": (2.0, 1.0), "ranging": (0.8, 0.6), "volatile": (1.5, 1.5)}
    REGIME_MAX_BARS = {"trending": 12, "ranging": 6, "volatile": 3}


def _parse_tp_sl_mb(s: str) -> tuple[float, float, int]:
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected tp,sl,max_bars comma-separated, got: {s!r}")
    return float(parts[0]), float(parts[1]), int(float(parts[2]))


def run_first_barrier_like_prepare(
    df: pd.DataFrame,
    *,
    use_event_score_tiers: bool,
    default_tp_mult: float = 1.5,
    default_sl_mult: float = 1.0,
    default_max_bars: int = 6,
    min_atr: float = 0.0003,
    kalman_event_floor: float = 0.70,
    weak_event_to_directional: bool = False,
    weak_event_min_move_atr: float = 0.35,
    score_floor_high: float = 0.7,
    score_floor_mid: float = 0.5,
    strong_tpl: tuple[float, float, int] = (2.0, 1.0, 24),
    mid_tpl: tuple[float, float, int] = (1.5, 1.0, 12),
    weak_tpl: tuple[float, float, int] = (1.0, 1.0, 6),
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    يعيد (جدول ملخص صفّي خفيف، مصفوفة tier لكل صف: -1 غير حدث أو baseline، 0 ضعيف، 1 وسط، 2 قوي).
    """
    df = df.copy().sort_values("ts_event").reset_index(drop=True)
    n = len(df)

    bias_label = np.full(n, DIR_NEUTRAL, dtype=np.int8)
    path_outcome = np.full(n, 4, dtype=np.int8)
    neutral_reason = np.full(n, NEUTRAL_REASON_NONE, dtype=np.int8)
    trade_duration = np.zeros(n, dtype=np.int16)
    signal_quality = np.zeros(n, dtype=np.int8)
    forward_return = np.zeros(n, dtype=np.float32)
    tier_i = np.full(n, -1, dtype=np.int16)

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

    def _session_end(i: int, max_bars_r: int) -> int:
        cap = min(i + max_bars_r, n - 1)
        if session_arr is None:
            return cap
        sess = session_arr[i]
        end_hour = _SESSION_END_HOUR.get(sess, 21)
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
        base_max_bars = int(REGIME_MAX_BARS.get(regime_i, default_max_bars))
        tp_mult_base, sl_mult_base = REGIME_TP_SL.get(regime_i, (default_tp_mult, default_sl_mult))

        if use_event_score_tiers and bool(is_ev_arr[i]):
            es = float(event_score_arr[i])
            if es >= score_floor_high:
                tp_mult, sl_mult, max_bars_r = strong_tpl
                tier_i[i] = 2
            elif es >= score_floor_mid:
                tp_mult, sl_mult, max_bars_r = mid_tpl
                tier_i[i] = 1
            else:
                tp_mult, sl_mult, max_bars_r = weak_tpl
                tier_i[i] = 0
            max_bars_r = int(max_bars_r)
        else:
            tp_mult, sl_mult = tp_mult_base, sl_mult_base
            max_bars_r = base_max_bars

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
    is_ev = pd.to_numeric(df["is_event"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    out["event_flag"] = (
        (is_ev == 1)
        & (out["bias_label"].isin([DIR_LONG, DIR_SHORT]))
        & (out["signal_quality"] > 0)
    ).astype(np.int8)
    out["train_event_flag"] = ((is_ev == 1) & (out["bias_label"] != DIR_NEUTRAL)).astype(np.int8)

    return out, tier_i


def _summarize(mode: str, df_src: pd.DataFrame, labels: pd.DataFrame, tier_i: np.ndarray) -> dict[str, Any]:
    is_ev = pd.to_numeric(df_src["is_event"], errors="coerce").fillna(0).astype(np.int8).to_numpy() == 1
    n = len(df_src)
    n_ev = int(is_ev.sum())

    def _among_ev(mask: np.ndarray) -> float:
        sub = mask & is_ev
        return float(sub.sum() / max(n_ev, 1))

    po = labels["path_outcome"].to_numpy()
    bl = labels["bias_label"].to_numpy()

    row: dict[str, Any] = {
        "mode": mode,
        "n_rows": n,
        "n_is_event": n_ev,
        "tier_strong_share_among_events": float(np.mean(tier_i[is_ev] == 2)) if n_ev else 0.0,
        "tier_mid_share_among_events": float(np.mean(tier_i[is_ev] == 1)) if n_ev else 0.0,
        "tier_weak_share_among_events": float(np.mean(tier_i[is_ev] == 0)) if n_ev else 0.0,
        "long_share_all": float(np.mean(bl == DIR_LONG)),
        "short_share_all": float(np.mean(bl == DIR_SHORT)),
        "neutral_share_all": float(np.mean(bl == DIR_NEUTRAL)),
        "event_flag_rate_all": float(labels["event_flag"].mean()),
        "train_event_flag_rate_all": float(labels["train_event_flag"].mean()),
        "among_events_tp_hit_share": _among_ev((po == 0) | (po == 1)),
        "among_events_timeout_share": _among_ev(po == 4),
        "among_events_sl_share": _among_ev((po == 2) | (po == 3)),
        "among_events_directional_share": _among_ev((bl == DIR_LONG) | (bl == DIR_SHORT)),
        "among_events_mean_trade_duration": float(labels.loc[is_ev, "trade_duration"].mean()) if n_ev else 0.0,
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baseline vs event_score-tiered TP/SL/horizon labeling on parquet")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-csv", default=None, help="Write comparison summary (2 rows) to CSV")
    ap.add_argument("--min-atr", type=float, default=0.0003)
    ap.add_argument("--kalman-event-floor", type=float, default=0.70)
    ap.add_argument("--score-floor-high", type=float, default=0.70, help="event_score >= this → strong tier")
    ap.add_argument("--score-floor-mid", type=float, default=0.50, help="event_score >= this (and < high) → mid tier")
    ap.add_argument("--strong-tp-sl-maxbars", type=str, default="2.0,1.0,24", help="tp,sl,max_bars")
    ap.add_argument("--mid-tp-sl-maxbars", type=str, default="1.5,1.0,12")
    ap.add_argument("--weak-tp-sl-maxbars", type=str, default="1.0,1.0,6")
    args = ap.parse_args()

    pq = os.path.abspath(args.parquet)
    if "..." in pq:
        raise SystemExit("Pass full parquet path (no ... placeholder).")
    if not os.path.isfile(pq):
        raise SystemExit(f"parquet not found: {pq}")

    need = {"ts_event", "close", "high", "low", "atr_14", "is_event", "event_score"}
    df = pd.read_parquet(pq).sort_values("ts_event").reset_index(drop=True)
    miss = sorted(need - set(df.columns))
    if miss:
        raise SystemExit(f"parquet missing columns: {miss}")

    strong_tpl = _parse_tp_sl_mb(args.strong_tp_sl_maxbars)
    mid_tpl = _parse_tp_sl_mb(args.mid_tp_sl_maxbars)
    weak_tpl = _parse_tp_sl_mb(args.weak_tp_sl_maxbars)

    if args.score_floor_mid > args.score_floor_high:
        raise SystemExit("--score-floor-mid must be <= --score-floor-high")

    base_labels, tier_base = run_first_barrier_like_prepare(
        df,
        use_event_score_tiers=False,
        min_atr=args.min_atr,
        kalman_event_floor=args.kalman_event_floor,
        score_floor_high=args.score_floor_high,
        score_floor_mid=args.score_floor_mid,
        strong_tpl=strong_tpl,
        mid_tpl=mid_tpl,
        weak_tpl=weak_tpl,
    )
    tier_labels, tier_ids = run_first_barrier_like_prepare(
        df,
        use_event_score_tiers=True,
        min_atr=args.min_atr,
        kalman_event_floor=args.kalman_event_floor,
        score_floor_high=args.score_floor_high,
        score_floor_mid=args.score_floor_mid,
        strong_tpl=strong_tpl,
        mid_tpl=mid_tpl,
        weak_tpl=weak_tpl,
    )

    s_base = _summarize("baseline_regime_only", df, base_labels, tier_base)
    s_tier = _summarize("tiered_event_score", df, tier_labels, tier_ids)

    pdf = pd.DataFrame([s_base, s_tier])

    print("=" * 72)
    print("diagnose_event_score_tier_labels_daytrade")
    print(f"  parquet={pq}")
    print(f"  rows={len(df):,}  REGIME_TP_SL keys={list(REGIME_TP_SL.keys())}")
    print(f"  tier floors: strong >= {args.score_floor_high} | mid >= {args.score_floor_mid} | else weak")
    print(f"  strong {strong_tpl} | mid {mid_tpl} | weak {weak_tpl}")
    print("=" * 72)
    print("\nAmong is_event rows - tier mix (tiered: tiers 0=weak,1=mid,2=strong; baseline row tiers are N/A):")
    ev_mask = pd.to_numeric(df["is_event"], errors="coerce").fillna(0).astype(np.int8).to_numpy() == 1
    if ev_mask.any():
        es = pd.to_numeric(df["event_score"], errors="coerce").fillna(0.0).to_numpy()[ev_mask]
        print(f"  event_score min/mean/max on events: {es.min():.3f} / {es.mean():.3f} / {es.max():.3f}")
        n_ev = int(es.size)
        n_str = int(np.sum(es >= args.score_floor_high))
        n_mid = int(np.sum((es >= args.score_floor_mid) & (es < args.score_floor_high)))
        n_wk = int(np.sum(es < args.score_floor_mid))
        print(f"  score-tier counts (among events): strong={n_str} mid={n_mid} weak={n_wk} (of {n_ev})")
    print(pdf.to_string(index=False))

    print("\nKey deltas (tiered minus baseline), among all rows:")
    for k in (
        "event_flag_rate_all",
        "train_event_flag_rate_all",
        "among_events_tp_hit_share",
        "among_events_timeout_share",
        "among_events_sl_share",
        "among_events_directional_share",
    ):
        d = float(s_tier[k]) - float(s_base[k])
        print(f"  {k}: {d:+.4f}")

    out_csv = args.out_csv
    if out_csv:
        pdf.to_csv(os.path.abspath(out_csv), index=False, encoding="utf-8-sig")
        print(f"\nWrote summary CSV -> {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    main()
