#!/usr/bin/env python3
"""
diagnose_event_direction_impact_daytrade.py

تجربة تشخيصية: أثر event_direction على ليبل First-Barrier (timeout / TP / SL)
بدون إعادة تشغيل prepare_day_trading.

1) يعيد حساب مسار التصويت نفسه تقريبًا كما في add_event_direction (prepare_day_trading.py)
   ويطبع ملخصًا لصفوف الـ timeout الملاحظة (إن وُجد path_outcome في parquet).

3) تشخيص عميق [vote_diagnosis]: بين صفوف is_event فقط
   - تقاطع strict_dir × stored_event_direction
   - تقاطع vote_sum × stored_event_direction
   - مصدر الحسم النهائي للاتجاه: strict_majority | fallback_score | weak_vote_only
   - صفوف هشة fragile: strict_dir==0 و vote_sum في {-1,+1} (تصويت غير صارم + هامش ضيق)

4) تجربة ثالثة اختيارية: event_direction=0 و kalman_event_floor=0 (تعطيل فعلي لفرع منع Kalman عندما الحدث محايد اتجاهًا)

استخدام:
  py -3 diagnose_event_direction_impact_daytrade.py --parquet PATH [--out-csv PATH]
  py -3 diagnose_event_direction_impact_daytrade.py --parquet PATH --out-diagnosis-prefix E:\\\\tmp\\\\ed_dx

معاملات:
  --no-kalman-zero-experiment   لا تشغّل التجربة الثالثة (صفر ed + kalman_floor=0)
  --skip-deep-diagnosis        لا تطبع/تحفظ تشخيص التصويت العميق
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd

# استيراد محاكاة الحواجز من السكربت المجاور
from diagnose_label_timeout_daytrade import (
    _metrics_among_events,
    _session_hours_with_extend,
    run_barrier_baseline,
    veto_source_at_entry,
)


def trace_event_direction_vote(df: pd.DataFrame) -> pd.DataFrame:
    """
    يطابق منطق add_event_direction في prepare_day_trading.py (نسخة تشخيصية).
    يعيد أعمدة تفسيرية صفًا بصف.
    """
    n = len(df)
    cvd_signed = pd.to_numeric(
        df.get("bar_cvd_delta", df.get("cvd_velocity_signed", 0.0)),
        errors="coerce",
    ).fillna(0.0).astype(np.float64)

    if "obi_direction" in df.columns:
        obi_dir = pd.to_numeric(df["obi_direction"], errors="coerce").fillna(0).astype(np.int8)
    else:
        obi_raw = pd.to_numeric(df.get("obi", df.get("order_flow_imbalance", 0.0)), errors="coerce").fillna(0.0)
        obi_dir = pd.Series(np.sign(obi_raw.to_numpy(dtype=np.float64)).astype(np.int8), index=df.index, dtype=np.int8)

    kalman_dir = pd.to_numeric(df.get("kalman_direction", 0), errors="coerce").fillna(0).astype(np.int8)
    cvd_arr = cvd_signed.to_numpy(dtype=np.float64)
    obi_arr = obi_dir.to_numpy(dtype=np.int8)
    kalman_arr = kalman_dir.to_numpy(dtype=np.int8)

    has_vwap_roll = "vwap_dist_1h_roll" in df.columns
    if has_vwap_roll:
        vwap_h = pd.to_numeric(df["vwap_dist_1h_roll"], errors="coerce").fillna(0.0).astype(np.float64)
        vwap_vote = np.sign(vwap_h).astype(np.int8)
    else:
        vwap_h = np.zeros(n, dtype=np.float64)
        vwap_vote = np.zeros(n, dtype=np.int8)

    vote_need = 3 if has_vwap_roll else 2
    cvd_sign = np.sign(cvd_arr).astype(np.int8)
    obi_sign = np.sign(obi_arr).astype(np.int8)
    kal_sign = np.sign(kalman_arr).astype(np.int8)
    vote = (cvd_sign + obi_sign + kal_sign + vwap_vote).astype(np.int16)

    strict_dir = np.where(vote >= vote_need, 1, np.where(vote <= -vote_need, -1, 0)).astype(np.int8)

    if "obi" in df.columns:
        obi_raw = pd.to_numeric(df["obi"], errors="coerce").fillna(0.0).astype(np.float64)
    elif "order_flow_imbalance" in df.columns:
        obi_raw = pd.to_numeric(df["order_flow_imbalance"], errors="coerce").fillna(0.0).astype(np.float64)
    else:
        obi_raw = pd.Series(obi_arr.astype(np.float64), index=df.index, dtype=np.float64)

    cvd_scale = pd.Series(np.abs(cvd_arr), index=df.index).rolling(50, min_periods=5).median()
    cvd_norm = pd.Series(cvd_arr, index=df.index) / cvd_scale.replace(0.0, np.nan)
    cvd_norm = cvd_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    obi_norm = obi_raw.clip(-1.0, 1.0)

    if has_vwap_roll:
        vwap_scale = pd.Series(np.abs(vwap_h), index=df.index).rolling(50, min_periods=5).median()
        vwap_norm = pd.Series(vwap_h, index=df.index) / vwap_scale.replace(0.0, np.nan)
        vwap_norm = vwap_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        fallback_score = (
            0.38 * cvd_norm.to_numpy(dtype=np.float64)
            + 0.30 * obi_norm.to_numpy(dtype=np.float64)
            + 0.17 * np.sign(kalman_arr).astype(np.float64)
            + 0.15 * vwap_norm.to_numpy(dtype=np.float64)
        )
    else:
        fallback_score = (
            0.45 * cvd_norm.to_numpy(dtype=np.float64)
            + 0.35 * obi_norm.to_numpy(dtype=np.float64)
            + 0.20 * np.sign(kalman_arr).astype(np.float64)
        )

    fallback_dir = np.where(fallback_score > 0.10, 1, np.where(fallback_score < -0.10, -1, 0)).astype(np.int8)
    weak_vote_dir = np.where(vote > 0, 1, np.where(vote < 0, -1, 0)).astype(np.int8)
    traced = np.where(
        strict_dir != 0,
        strict_dir,
        np.where(fallback_dir != 0, fallback_dir, weak_vote_dir),
    ).astype(np.int8)

    stored = pd.to_numeric(df["event_direction"], errors="coerce").fillna(0).astype(np.int8).to_numpy() if "event_direction" in df.columns else traced.copy()

    return pd.DataFrame(
        {
            "cvd_sign": cvd_sign.astype(np.int8),
            "obi_sign": obi_sign.astype(np.int8),
            "kalman_sign": kal_sign.astype(np.int8),
            "vwap_sign": vwap_vote.astype(np.int8),
            "vote_sum": vote.astype(np.int16),
            "vote_need": np.full(n, vote_need, dtype=np.int8),
            "strict_dir_before_fallback": strict_dir,
            "fallback_score": fallback_score.astype(np.float32),
            "fallback_dir": fallback_dir,
            "weak_vote_dir": weak_vote_dir,
            "traced_event_direction": traced,
            "stored_event_direction": stored,
            "trace_matches_stored": traced == stored,
        },
        index=df.index,
    )


def _resolution_label(strict_dir: np.ndarray, fallback_dir: np.ndarray, weak_vote_dir: np.ndarray) -> np.ndarray:
    """أين حُسم traced_event_direction بعد غياب أغلبية صارمة."""
    out = np.empty(len(strict_dir), dtype=object)
    for i in range(len(strict_dir)):
        if int(strict_dir[i]) != 0:
            out[i] = "strict_majority"
        elif int(fallback_dir[i]) != 0:
            out[i] = "fallback_score"
        else:
            out[i] = "weak_vote_only"
    return out


def print_and_export_vote_deep_diagnosis(
    df: pd.DataFrame,
    tr: pd.DataFrame,
    *,
    out_diagnosis_prefix: str | None,
) -> None:
    """تشخيص تصويت event_direction على كل صفوف is_event."""
    ie = pd.to_numeric(df["is_event"], errors="coerce").fillna(0).astype(np.int8).to_numpy() == 1
    if not ie.any():
        print("\n[vote_diagnosis] no is_event rows — skip")
        return

    sub_tr = tr.loc[ie].copy()
    sub_df = df.loc[ie]
    n_ev = int(ie.sum())

    strict = sub_tr["strict_dir_before_fallback"].to_numpy(dtype=np.int16)
    vote_sum = sub_tr["vote_sum"].to_numpy(dtype=np.int16)
    fb = sub_tr["fallback_dir"].to_numpy(dtype=np.int16)
    wv = sub_tr["weak_vote_dir"].to_numpy(dtype=np.int16)
    stored = sub_tr["stored_event_direction"].to_numpy(dtype=np.int16)

    res_lbl = _resolution_label(
        sub_tr["strict_dir_before_fallback"].to_numpy(dtype=np.int8),
        sub_tr["fallback_dir"].to_numpy(dtype=np.int8),
        sub_tr["weak_vote_dir"].to_numpy(dtype=np.int8),
    )
    fragile = (strict == 0) & (np.abs(vote_sum) == 1)

    print("\n[vote_diagnosis] Among is_event rows only")
    print(f"  n_events={n_ev:,}")

    ct_strict_stored = pd.crosstab(
        pd.Series(strict, name="strict_dir_before_fallback"),
        pd.Series(stored, name="stored_event_direction"),
        margins=True,
    )
    print("\n  Crosstab strict_dir_before_fallback x stored_event_direction:")
    print(ct_strict_stored.to_string())

    ct_vote_stored = pd.crosstab(
        pd.Series(vote_sum, name="vote_sum"),
        pd.Series(stored, name="stored_event_direction"),
        margins=True,
    )
    print("\n  Crosstab vote_sum x stored_event_direction:")
    print(ct_vote_stored.to_string())

    vc_res = pd.Series(res_lbl).value_counts()
    print("\n  Final direction resolution source (matches add_event_direction chain):")
    for k, v in vc_res.items():
        print(f"    {k}: {int(v)} ({int(v) / max(n_ev, 1):.2%})")

    n_frag = int(np.sum(fragile))
    print(f"\n  Fragile votes (strict_dir==0 AND |vote_sum|==1): {n_frag} ({n_frag / max(n_ev, 1):.2%} of events)")
    if n_frag > 0 and "path_outcome" in df.columns:
        po_ev = pd.to_numeric(sub_df["path_outcome"], errors="coerce").fillna(-1).astype(np.int8).to_numpy()
        po_frag = po_ev[fragile]
        po_rest = po_ev[~fragile]
        print("    observed path_outcome mix among fragile vs rest (events):")
        for name, arr in ("fragile", po_frag), ("rest", po_rest):
            n_a = len(arr)
            if n_a == 0:
                continue
            tp_ = float(np.mean((arr == 0) | (arr == 1)))
            sl_ = float(np.mean((arr == 2) | (arr == 3)))
            to_ = float(np.mean(arr == 4))
            print(f"      {name}: n={n_a} tp_share={tp_:.2%} sl_share={sl_:.2%} timeout_share={to_:.2%}")

    if out_diagnosis_prefix:
        base = os.path.abspath(out_diagnosis_prefix)
        p1 = base + "_strict_x_stored_event_direction.csv"
        ct_strict_stored.to_csv(p1, encoding="utf-8-sig")
        p2 = base + "_vote_sum_x_stored_event_direction.csv"
        ct_vote_stored.to_csv(p2, encoding="utf-8-sig")
        print(f"\n[vote_diagnosis] Wrote {p1}")
        print(f"[vote_diagnosis] Wrote {p2}")

        if "ts_event" in sub_df.columns:
            fragile_df = pd.concat(
                [sub_df.loc[fragile, ["ts_event"]].reset_index(drop=True), sub_tr.loc[fragile].reset_index(drop=True)],
                axis=1,
            )
        else:
            fragile_df = sub_tr.loc[fragile].reset_index(drop=True)
        if "path_outcome" in sub_df.columns:
            fragile_df = pd.concat(
                [fragile_df, sub_df.loc[fragile, ["path_outcome"]].reset_index(drop=True)],
                axis=1,
            )
        p3 = base + "_fragile_vote_margin_events.csv"
        fragile_df.to_csv(p3, index=False, encoding="utf-8-sig")
        print(f"[vote_diagnosis] Wrote {p3} ({len(fragile_df)} fragile rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose impact of event_direction on barrier labels")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-csv", default=None, help="Save experiment comparison rows")
    ap.add_argument("--kalman-floor", type=float, default=0.70)
    ap.add_argument("--max-bars-mult", type=float, default=1.0)
    ap.add_argument("--session-extend-hours", type=int, default=0)
    ap.add_argument("--skip-vote-trace", action="store_true")
    ap.add_argument("--skip-deep-diagnosis", action="store_true", help="Skip vote crosstabs / fragile analysis")
    ap.add_argument(
        "--no-kalman-zero-experiment",
        action="store_true",
        help="Do not run counterfactual: zero event_direction + kalman_floor=0",
    )
    ap.add_argument(
        "--out-diagnosis-prefix",
        default=None,
        help="Write vote diagnosis CSVs: {prefix}_strict_x_stored..., _vote_sum_x..., _fragile_vote_margin_events.csv",
    )
    args = ap.parse_args()

    pq = os.path.abspath(args.parquet)
    if "..." in pq:
        raise SystemExit("Use full parquet path (no ... placeholder).")
    if not os.path.isfile(pq):
        raise SystemExit(f"parquet not found: {pq}")

    df = pd.read_parquet(pq).sort_values("ts_event").reset_index(drop=True)

    print("=" * 72)
    print("diagnose_event_direction_impact_daytrade")
    print(f"  parquet={pq}")
    print(f"  rows={len(df):,}")
    print("=" * 72)

    tr: pd.DataFrame | None = None
    if "event_direction" in df.columns:
        tr = trace_event_direction_vote(df)

    if tr is not None and not args.skip_vote_trace:
        match_rate = float(np.mean(tr["trace_matches_stored"].to_numpy()))
        print(f"\n[vote_trace] traced vs stored event_direction match rate: {match_rate:.2%}")
        ie = pd.to_numeric(df["is_event"], errors="coerce").fillna(0).astype(np.int8).to_numpy() == 1
        if ie.any() and "path_outcome" in df.columns:
            po = pd.to_numeric(df["path_outcome"], errors="coerce").fillna(-1).astype(np.int8).to_numpy()
            to_ev = ie & (po == 4)
            if to_ev.any():
                print(f"\n[vote_trace] Observed timeouts among events ({int(to_ev.sum())} rows):")
                cols_show = [
                    "vote_sum",
                    "vote_need",
                    "strict_dir_before_fallback",
                    "fallback_score",
                    "fallback_dir",
                    "weak_vote_dir",
                    "traced_event_direction",
                    "stored_event_direction",
                ]
                sub = pd.concat([df.loc[to_ev, ["ts_event"]], tr.loc[to_ev, cols_show]], axis=1)
                print(sub.to_string(index=False))
            else:
                print("\n[vote_trace] No observed timeouts among is_event — skip timeout detail table.")

    if tr is not None and not args.skip_deep_diagnosis:
        print_and_export_vote_deep_diagnosis(df, tr, out_diagnosis_prefix=args.out_diagnosis_prefix)

    sh = _session_hours_with_extend(args.session_extend_hours)

    exp_specs: list[tuple[str, bool, float]] = [
        ("baseline_stored_event_direction", False, args.kalman_floor),
        ("counterfactual_event_direction_all_zero", True, args.kalman_floor),
    ]
    if not args.no_kalman_zero_experiment:
        exp_specs.append(("counterfactual_zero_ed_kalman_floor_0", True, 0.0))

    experiments: list[dict[str, Any]] = []
    for label, zero_ed, kf in exp_specs:
        dfx = df.copy()
        if zero_ed:
            dfx["event_direction"] = np.int8(0)
        lab = run_barrier_baseline(
            dfx,
            max_bars_mult=args.max_bars_mult,
            session_end_hours=sh,
            kalman_event_floor=kf,
        )
        m = _metrics_among_events(df, lab)
        experiments.append(
            {
                "experiment": label,
                "zeroed_all_event_direction": zero_ed,
                "kalman_event_floor_used": kf,
                **m,
            }
        )

        ie = pd.to_numeric(df["is_event"], errors="coerce").fillna(0).astype(np.int8).to_numpy() == 1
        po = lab["path_outcome"].to_numpy()
        to_ix = np.flatnonzero(ie & (po == 4))
        if len(to_ix):
            es = pd.to_numeric(dfx["event_score"], errors="coerce").fillna(0.0).to_numpy()
            kd = pd.to_numeric(dfx["kalman_direction"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
            ed = pd.to_numeric(dfx["event_direction"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
            reasons = [veto_source_at_entry(int(ed[i]), int(kd[i]), float(es[i]), kf) for i in to_ix]
            vc = pd.Series(reasons).value_counts()
            print(f"\n[veto_source after sim] {label} (kf={kf}) - simulated timeouts among events ({len(to_ix)}):")
            for name, c in vc.items():
                print(f"    {name}: {int(c)}")

    pdf = pd.DataFrame(experiments)
    print("\n[experiment] Barrier outcomes among is_event")
    print(
        f"  default kalman_floor={args.kalman_floor}, max_bars_mult={args.max_bars_mult}, "
        f"session_extend_hours={args.session_extend_hours}"
    )
    show = [
        "experiment",
        "zeroed_all_event_direction",
        "kalman_event_floor_used",
        "among_events_timeout_share",
        "among_events_timeout_then_horizon_share",
        "among_events_timeout_then_kalman_share",
        "among_events_tp_share",
        "among_events_sl_share",
    ]
    show = [c for c in show if c in pdf.columns]
    print(pdf[show].to_string(index=False))

    base_to = float(pdf.loc[pdf["experiment"].eq("baseline_stored_event_direction"), "among_events_timeout_share"].iloc[0])
    for _, row in pdf.iterrows():
        if row["experiment"] == "baseline_stored_event_direction":
            continue
        ctr_to = float(row["among_events_timeout_share"])
        print(f"\n  Delta timeout_share ({row['experiment']} - baseline): {ctr_to - base_to:+.4f}")

    if args.out_csv:
        pdf.to_csv(os.path.abspath(args.out_csv), index=False, encoding="utf-8-sig")
        print(f"\nWrote -> {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
