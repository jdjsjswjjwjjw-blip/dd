#!/usr/bin/env python3
"""
diagnose_hybrid_gate_tier_daytrade.py

تشخيص «هجين»: يجرّب شبكة معلمات Event Gate (كما في diagnose_event_gate_daytrade.py)،
ثم لكل تركيبة يعيد بناء is_event و event_score من المحاكاة، ويقارن ليبلات First-Barrier:

  • baseline_regime — TP/SL/max_bars حسب regime فقط (كما في الإنتاج).
  • tiered_event_score — بين صفوف الحدث فقط: TP/SL/horizon حسب شرائح event_score المحاكى.

لا يعدّل prepare_day_trading ولا يكتب باركيه؛ يخرج CSV واحداً للتحليل.

متطلبات parquet (إلى جانب أعمدة البوابة):
  ts_event, close, high, low, atr_14, regime_label (مستحسن), kalman_direction, event_direction,
  hawkes_intensity, absorption_intensity, kyle_lambda (+ cvd_direction_pct أو cvd_momentum)

استخدام (PowerShell، سطر واحد أو backtick):
  py -3 diagnose_hybrid_gate_tier_daytrade.py ^
    --parquet "E:/path/day_trading_features_xxx.parquet" ^
    --scale-grid "1.08" --shift-grid "0.01" --min-rate-grid "0.12" ^
    --hawkes-z-cut-grid "1.0,1.2,1.4,1.6" ^
    --absorb-z-cut-grid "1.0,1.2,1.4" ^
    --kyle-z-cut-grid "0.5,0.75,1.0" ^
    --cvd-align-cut-grid "0.6,0.7,0.75" ^
    --out-csv "E:/path/hybrid_gate_tier_sweep.csv"
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd

import diagnose_event_score_tier_labels_daytrade as tier_lab
from diagnose_event_gate_daytrade import (
    EVENT_LOB_COVERAGE_CUT,
    EVENT_MBO_COVERAGE_CUT,
    _mbo_bar_coverage_from_tick_series,
    _parse_float_grid,
    simulate_event_gate,
)


def _add_gate_tier_argparse(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--parquet", required=True, help="day_trading_features*.parquet")
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Sweep CSV (default: next to parquet, suffix _hybrid_gate_tier_sweep.csv)",
    )
    ap.add_argument("--scale-grid", default="1.0")
    ap.add_argument("--shift-grid", default="0.0")
    ap.add_argument("--min-rate-grid", default="0.12")
    ap.add_argument("--hawkes-z-cut-grid", default="1.0")
    ap.add_argument("--absorb-z-cut-grid", default="1.0")
    ap.add_argument("--kyle-z-cut-grid", default="0.5")
    ap.add_argument("--cvd-align-cut-grid", default="0.6")
    ap.add_argument(
        "--mbp-cov-cut-grid",
        default=str(EVENT_LOB_COVERAGE_CUT),
        help="Cuts on mbp_roll_lob_coverage",
    )
    ap.add_argument(
        "--mbo-cov-cut-grid",
        default=str(EVENT_MBO_COVERAGE_CUT),
        help="Cuts on mbo_bar_coverage",
    )
    ap.add_argument("--max-combo", type=int, default=100_000, help="Abort if Cartesian product exceeds this")
    # Tier labeling (نفس diagnose_event_score_tier_labels_daytrade)
    ap.add_argument("--min-atr", type=float, default=0.0003)
    ap.add_argument("--kalman-event-floor", type=float, default=0.70)
    ap.add_argument("--score-floor-high", type=float, default=0.70)
    ap.add_argument("--score-floor-mid", type=float, default=0.50)
    ap.add_argument("--strong-tp-sl-maxbars", type=str, default="2.0,1.0,24")
    ap.add_argument("--mid-tp-sl-maxbars", type=str, default="1.5,1.0,12")
    ap.add_argument("--weak-tp-sl-maxbars", type=str, default="1.0,1.0,6")
    ap.add_argument(
        "--sort-by",
        default="delta_among_events_tp_hit_share",
        help="Column name to sort descending (CSV + printed head)",
    )
    ap.add_argument("--head", type=int, default=20, help="Rows to print after sort")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep Event Gate × compare tiered vs regime-only labeling (DayTrade parquet)"
    )
    _add_gate_tier_argparse(ap)
    args = ap.parse_args()

    pq = os.path.abspath(args.parquet)
    if "..." in pq:
        raise SystemExit("Pass full parquet path (no ... placeholder).")
    if not os.path.isfile(pq):
        raise SystemExit(f"parquet not found: {pq}")

    need_gate = {"hawkes_intensity", "absorption_intensity", "kyle_lambda"}
    need_tier = {"ts_event", "close", "high", "low", "atr_14"}
    df = pd.read_parquet(pq).sort_values("ts_event").reset_index(drop=True)
    miss = sorted((need_gate | need_tier) - set(df.columns))
    if miss:
        raise SystemExit(f"parquet missing columns: {miss}")

    if "tick_count" in df.columns:
        df = df.copy()
        df["mbo_bar_coverage"] = _mbo_bar_coverage_from_tick_series(df["tick_count"])
    elif "num_trades" in df.columns:
        df = df.copy()
        df["mbo_bar_coverage"] = _mbo_bar_coverage_from_tick_series(df["num_trades"])
    elif "mbo_bar_coverage" not in df.columns:
        df = df.copy()
        df["mbo_bar_coverage"] = np.float32(0.0)

    if args.score_floor_mid > args.score_floor_high:
        raise SystemExit("--score-floor-mid must be <= --score-floor-high")

    strong_tpl = tier_lab._parse_tp_sl_mb(args.strong_tp_sl_maxbars)
    mid_tpl = tier_lab._parse_tp_sl_mb(args.mid_tp_sl_maxbars)
    weak_tpl = tier_lab._parse_tp_sl_mb(args.weak_tp_sl_maxbars)

    scales = _parse_float_grid(args.scale_grid)
    shifts = _parse_float_grid(args.shift_grid)
    rates = _parse_float_grid(args.min_rate_grid)
    hawkes_cuts = _parse_float_grid(args.hawkes_z_cut_grid)
    absorb_cuts = _parse_float_grid(args.absorb_z_cut_grid)
    kyle_cuts = _parse_float_grid(args.kyle_z_cut_grid)
    cvd_cuts = _parse_float_grid(args.cvd_align_cut_grid)
    cov_cuts = _parse_float_grid(args.mbp_cov_cut_grid)
    mbo_cuts = _parse_float_grid(args.mbo_cov_cut_grid)

    n_combo = (
        len(scales)
        * len(shifts)
        * len(rates)
        * len(hawkes_cuts)
        * len(absorb_cuts)
        * len(kyle_cuts)
        * len(cvd_cuts)
        * len(cov_cuts)
        * len(mbo_cuts)
    )
    if n_combo > int(args.max_combo):
        raise SystemExit(f"Too many combinations ({n_combo:,} > --max-combo {args.max_combo}); narrow grids.")

    parquet_ev = None
    if "is_event" in df.columns:
        parquet_ev = pd.to_numeric(df["is_event"], errors="coerce").fillna(0).astype(np.int8).to_numpy()

    tier_kw: dict[str, Any] = {
        "min_atr": float(args.min_atr),
        "kalman_event_floor": float(args.kalman_event_floor),
        "score_floor_high": float(args.score_floor_high),
        "score_floor_mid": float(args.score_floor_mid),
        "strong_tpl": strong_tpl,
        "mid_tpl": mid_tpl,
        "weak_tpl": weak_tpl,
    }

    rows_out: list[dict[str, Any]] = []

    print("=" * 72)
    print("diagnose_hybrid_gate_tier_daytrade")
    print(f"  parquet={pq}")
    print(f"  rows={len(df):,}")
    if parquet_ev is not None:
        print(f"  parquet is_event rate={float(np.mean(parquet_ev == 1)):.1%}")
    print(f"  gate sweep combos={n_combo:,}")
    print(
        f"  tier floors: strong >= {args.score_floor_high} | mid >= {args.score_floor_mid} | else weak"
    )
    print(f"  strong {strong_tpl} | mid {mid_tpl} | weak {weak_tpl}")
    print("=" * 72)

    for hz in hawkes_cuts:
        for az in absorb_cuts:
            for kz in kyle_cuts:
                for cv in cvd_cuts:
                    for cov_c in cov_cuts:
                        for mbo_c in mbo_cuts:
                            for sc in scales:
                                for sh in shifts:
                                    for mr in rates:
                                        r = simulate_event_gate(
                                            df,
                                            threshold_scale=sc,
                                            threshold_shift=sh,
                                            min_event_rate=mr,
                                            hawkes_z_cut=hz,
                                            absorb_z_cut=az,
                                            kyle_z_cut=kz,
                                            cvd_align_cut=cv,
                                            mbp_cov_cut=cov_c,
                                            mbo_cov_cut=mbo_c,
                                        )
                                        df_w = df.copy()
                                        df_w["is_event"] = r["event_mask"].astype(np.int8)
                                        df_w["event_score"] = r["stored_score"]

                                        base_labels, tier_base = tier_lab.run_first_barrier_like_prepare(
                                            df_w,
                                            use_event_score_tiers=False,
                                            **tier_kw,
                                        )
                                        tier_labels, tier_ids = tier_lab.run_first_barrier_like_prepare(
                                            df_w,
                                            use_event_score_tiers=True,
                                            **tier_kw,
                                        )

                                        s_base = tier_lab._summarize(
                                            "baseline_regime_only", df_w, base_labels, tier_base
                                        )
                                        s_tier = tier_lab._summarize(
                                            "tiered_event_score", df_w, tier_labels, tier_ids
                                        )

                                        row: dict[str, Any] = {
                                            "hawkes_z_cut": hz,
                                            "absorb_z_cut": az,
                                            "kyle_z_cut": kz,
                                            "cvd_align_cut": cv,
                                            "mbp_cov_cut": cov_c,
                                            "mbo_cov_cut": mbo_c,
                                            "threshold_scale": sc,
                                            "threshold_shift": sh,
                                            "min_event_rate": mr,
                                            "sim_event_rate": float(np.mean(r["event_mask"])),
                                            "sim_adaptive_added": int(r["adaptive_added"]),
                                            "sim_stored_score_mean": float(r["stored_score_mean"]),
                                            "sim_thr_mean": float(r["threshold_mean"]),
                                        }
                                        if parquet_ev is not None:
                                            row["agreement_vs_parquet_is_event"] = float(
                                                np.mean((parquet_ev == 1) == r["event_mask"])
                                            )

                                        for prefix, sd in (("base_", s_base), ("tier_", s_tier)):
                                            for k, v in sd.items():
                                                if k == "mode":
                                                    continue
                                                row[prefix + k] = v

                                        keys_delta = (
                                            "among_events_tp_hit_share",
                                            "among_events_timeout_share",
                                            "among_events_sl_share",
                                            "among_events_directional_share",
                                            "event_flag_rate_all",
                                            "train_event_flag_rate_all",
                                            "among_events_mean_trade_duration",
                                        )
                                        for k in keys_delta:
                                            row["delta_" + k] = float(s_tier[k]) - float(s_base[k])

                                        rows_out.append(row)

    pdf = pd.DataFrame(rows_out)
    sort_col = str(args.sort_by)
    if sort_col not in pdf.columns:
        raise SystemExit(f"--sort-by column not in results: {sort_col!r}")
    pdf = pdf.sort_values(sort_col, ascending=False)

    head_n = max(0, int(args.head))
    if head_n > 0:
        print(f"\nTop {head_n} by {sort_col} (tiered minus baseline where delta_*):")
        show_cols = [
            c
            for c in (
                "hawkes_z_cut",
                "absorb_z_cut",
                "kyle_z_cut",
                "cvd_align_cut",
                "mbp_cov_cut",
                "mbo_cov_cut",
                "threshold_scale",
                "threshold_shift",
                "min_event_rate",
                "sim_event_rate",
                "sim_adaptive_added",
                sort_col,
                "delta_among_events_tp_hit_share",
                "delta_among_events_sl_share",
                "delta_among_events_timeout_share",
                "agreement_vs_parquet_is_event",
            )
            if c in pdf.columns
        ]
        print(pdf[show_cols].head(head_n).to_string(index=False))

    out_csv = args.out_csv or (os.path.splitext(pq)[0] + "_hybrid_gate_tier_sweep.csv")
    out_abs = os.path.abspath(out_csv)
    pdf.to_csv(out_abs, index=False, encoding="utf-8-sig")
    print(f"\nWrote sweep CSV -> {out_abs}  (rows={len(pdf):,})")


if __name__ == "__main__":
    main()
