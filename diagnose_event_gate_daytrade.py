#!/usr/bin/env python3
"""
diagnose_event_gate_daytrade.py — تشخيص قوي لـ Event Gate على باركيه DayTrade جاهز

يعيد تنفيذ منطق detect_microstructure_events على أعمدة الشمعات في parquet، ثم يجرّب شبكة من:
  threshold_scale / threshold_shift / min_event_rate
  واختياريًا عتبات الشروط الثنائية hawkes_z / absorb_z / kyle_z / cvd_align (قيم افتراضية = prepare_day_trading).

  • عمود session (asia/london/overlap/ny): جدول --session-breakdown يطبّق أوّل قيمة من كل شبكة
  • mbp_roll_lob_coverage / mbo_bar_coverage: مرحلتان وزنيتان في event_score (regime_config + أعمدة parquet أو tick_count)

لا يعيد بناء الليبلات الصلبة — يقيس تأثير البوابة على:
  • نسبة الأحداث
  • adaptive_added (عدد الصفوف المضافة للوصول لـ min_event_rate)
  • تقاطع مع bias_label الموجود في الملف (كم اتجاهي حاليًا كان تحت البوابة الجديدة)

استخدام (سطر واحد — يعمل في PowerShell و CMD):
  py -3 diagnose_event_gate_daytrade.py --parquet "E:/.../day_trading_features_xxx.parquet" --out-csv "E:/.../event_gate_sweep.csv"

PowerShell: لا تستخدم ^ (هذا لـ cmd.exe فقط؛ وإلا يُمرَّر ^ إلى بايثون فيظهر unrecognized arguments).
  استمرار سطر في PowerShell بالرمز backtick عند نهاية السطر، أو اكتب الأمر في سطر واحد، أو شغّل من cmd.

  py -3 diagnose_event_gate_daytrade.py `
    --parquet "E:/.../day_trading_features_xxx.parquet" `
    --session-breakdown --out-session-csv "E:/.../event_gate_by_session.csv"

cmd.exe (استمرار بـ ^):
  py -3 diagnose_event_gate_daytrade.py ^
    --parquet "E:/.../day_trading_features_xxx.parquet" ^
    --out-csv "E:/.../event_gate_sweep.csv"

شبكة مخصصة (cmd):
  py -3 diagnose_event_gate_daytrade.py --parquet "E:/.../file.parquet" ^
    --scale-grid "1.0,0.85,0.7" --shift-grid "0,-0.05,-0.1" --min-rate-grid "0.12,0.30,0.45" ^
    --mbp-cov-cut-grid "0.45,0.50,0.55" ^
    --mbo-cov-cut-grid "0.45,0.50,0.55" ^
    --session-breakdown --out-session-csv "E:/.../event_gate_by_session.csv"

عتبات القطع داخل event_score (افتراضيات مثل prepare_day_trading):
  --hawkes-z-cut-grid   hawkes_z > cut   (افتراضي 1.0)
  --absorb-z-cut-grid   absorb_z > cut   (افتراضي 1.0)
  --kyle-z-cut-grid     kyle_z > cut     (افتراضي 0.5)
  --cvd-align-cut-grid  cvd_align > cut  (افتراضي 0.6)
  --mbp-cov-cut-grid    mbp_roll_lob_coverage > cut (افتراضي EVENT_LOB_COVERAGE_CUT)
  --mbo-cov-cut-grid    mbo_bar_coverage > cut (افتراضي EVENT_MBO_COVERAGE_CUT)

تحذير: كل قيمة إضافية تضرب حجم الجدول ضرب كارتيزي مع باقي الشبكات.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd

# نفس fallback في prepare_day_trading.py إذا لم يُحمَّل regime_config
REGIME_EVENT_THRESHOLD = {"trending": 0.60, "ranging": 0.60, "volatile": 0.75}
EVENT_ZSCORE_WINDOW = 100
EVENT_ZSCORE_MIN_PERIODS = 20
EVENT_LOB_COVERAGE_CUT = 0.50
EVENT_MBO_COVERAGE_CUT = 0.50
EVENT_SCORE_WEIGHTS = {
    "hawkes_z_above_1": 0.26,
    "absorb_z_above_1": 0.26,
    "kyle_z_above_05": 0.165,
    "cvd_align_above_06": 0.165,
    "mbp_roll_cov_above_cut": 0.075,
    "mbo_tick_cov_above_cut": 0.075,
}

DIR_LONG = 0
DIR_SHORT = 1
DIR_NEUTRAL = 2


try:
    from regime_config import (
        REGIME_EVENT_THRESHOLD as RC_THRESH,
        EVENT_ZSCORE_WINDOW as RC_ZW,
        EVENT_ZSCORE_MIN_PERIODS as RC_ZMIN,
        EVENT_SCORE_WEIGHTS as RC_W,
        EVENT_LOB_COVERAGE_CUT as RC_LOB_CUT,
        EVENT_MBO_COVERAGE_CUT as RC_MBO_CUT,
    )

    REGIME_EVENT_THRESHOLD = dict(RC_THRESH)
    EVENT_ZSCORE_WINDOW = int(RC_ZW)
    EVENT_ZSCORE_MIN_PERIODS = int(RC_ZMIN)
    EVENT_SCORE_WEIGHTS = dict(RC_W)
    EVENT_LOB_COVERAGE_CUT = float(RC_LOB_CUT)
    EVENT_MBO_COVERAGE_CUT = float(RC_MBO_CUT)
except ImportError:
    pass


def _zscore(series: pd.Series) -> pd.Series:
    roll = series.rolling(EVENT_ZSCORE_WINDOW, min_periods=EVENT_ZSCORE_MIN_PERIODS)
    return ((series - roll.mean()) / (roll.std() + 1e-9)).fillna(0.0)


def _mbo_bar_coverage_from_tick_series(tc: pd.Series) -> np.ndarray:
    """موازٍ لـ prepare_day_trading._mbo_bar_coverage_from_tick_series."""
    t = pd.to_numeric(tc, errors="coerce").fillna(0.0).astype(np.float64)
    med = t.rolling(EVENT_ZSCORE_WINDOW, min_periods=EVENT_ZSCORE_MIN_PERIODS).median()
    den = np.maximum(med.to_numpy(dtype=np.float64), 1.0)
    den = np.where(np.isfinite(den), den, np.maximum(t.to_numpy(dtype=np.float64), 1.0))
    return np.minimum(t.to_numpy(dtype=np.float64) / den, 1.0).astype(np.float32)


def simulate_event_gate(
    df: pd.DataFrame,
    *,
    threshold_scale: float,
    threshold_shift: float,
    min_event_rate: float,
    hawkes_z_cut: float = 1.0,
    absorb_z_cut: float = 1.0,
    kyle_z_cut: float = 0.5,
    cvd_align_cut: float = 0.6,
    mbp_cov_cut: float | None = None,
    mbo_cov_cut: float | None = None,
) -> dict[str, Any]:
    """
    مطابق لمنطق detect_microstructure_events في prepare_day_trading.py (نسخة تشخيصية)،
    مع السماح بتغيير عتبات الشروط الثنائية في event_score وعتبات تغطية MBP/MBO.
    """
    hawkes_z = _zscore(
        pd.to_numeric(df.get("hawkes_intensity", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    )
    absorb_z = _zscore(
        pd.to_numeric(df.get("absorption_intensity", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    )
    kyle_z = _zscore(pd.to_numeric(df.get("kyle_lambda", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0))

    if "cvd_direction_pct" in df.columns:
        cvd_align = pd.to_numeric(df["cvd_direction_pct"], errors="coerce").fillna(0.5)
    elif "cvd_momentum" in df.columns:
        cm = pd.to_numeric(df["cvd_momentum"], errors="coerce").fillna(0.0)
        cvd_align = (cm.abs() / (cm.abs().rolling(50, min_periods=5).max() + 1e-9)).clip(0.0, 1.0)
    else:
        cvd_align = pd.Series(0.5, index=df.index)

    cov_raw = pd.to_numeric(
        df.get("mbp_roll_lob_coverage", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).fillna(0.0)
    cov = cov_raw.clip(lower=0.0, upper=1.0)

    mbo_cov_raw = pd.to_numeric(
        df.get("mbo_bar_coverage", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).fillna(0.0)
    mbo_cov = mbo_cov_raw.clip(lower=0.0, upper=1.0)

    hz = float(hawkes_z_cut)
    az = float(absorb_z_cut)
    kz = float(kyle_z_cut)
    cv = float(cvd_align_cut)
    cov_c = float(EVENT_LOB_COVERAGE_CUT if mbp_cov_cut is None else mbp_cov_cut)
    mbo_c = float(EVENT_MBO_COVERAGE_CUT if mbo_cov_cut is None else mbo_cov_cut)

    w = EVENT_SCORE_WEIGHTS
    cov_w = float(w.get("mbp_roll_cov_above_cut", 0.0))
    mbo_w = float(w.get("mbo_tick_cov_above_cut", 0.0))

    event_score = (
        (hawkes_z > hz).astype(np.float32) * w["hawkes_z_above_1"]
        + (absorb_z > az).astype(np.float32) * w["absorb_z_above_1"]
        + (kyle_z > kz).astype(np.float32) * w["kyle_z_above_05"]
        + (cvd_align > cv).astype(np.float32) * w["cvd_align_above_06"]
    ).astype(np.float32)
    if cov_w > 0.0:
        event_score = event_score + (cov > cov_c).astype(np.float32) * np.float32(cov_w)
    if mbo_w > 0.0:
        event_score = event_score + (mbo_cov > mbo_c).astype(np.float32) * np.float32(mbo_w)

    continuous_score = (
        hawkes_z.clip(lower=0.0, upper=3.0).astype(np.float32) / np.float32(3.0) * w["hawkes_z_above_1"]
        + absorb_z.clip(lower=0.0, upper=3.0).astype(np.float32) / np.float32(3.0) * w["absorb_z_above_1"]
        + kyle_z.clip(lower=0.0, upper=2.0).astype(np.float32) / np.float32(2.0) * w["kyle_z_above_05"]
        + cvd_align.clip(lower=0.0, upper=1.0).astype(np.float32) * w["cvd_align_above_06"]
    ).astype(np.float32)
    if cov_w > 0.0:
        continuous_score = continuous_score + cov.astype(np.float32) * np.float32(cov_w)
    if mbo_w > 0.0:
        continuous_score = continuous_score + mbo_cov.astype(np.float32) * np.float32(mbo_w)

    if "regime_label" in df.columns:
        regime_s = df["regime_label"].astype(str)
        base_threshold = regime_s.map(REGIME_EVENT_THRESHOLD).fillna(0.60).astype(np.float32)
    else:
        base_threshold = pd.Series(0.60, index=df.index, dtype=np.float32)

    scale = float(max(threshold_scale, 0.01))
    shift = float(threshold_shift)
    threshold = (base_threshold.astype(np.float64) * scale + shift).clip(0.05, 0.95).astype(np.float32)

    pass_raw_before_adaptive = np.array(event_score >= threshold, dtype=bool)
    event_mask = pass_raw_before_adaptive.copy()
    adaptive_added = 0
    target_rate = float(np.clip(min_event_rate, 0.0, 0.60))
    n = len(df)
    if target_rate > 0.0 and n:
        target_events = int(np.ceil(n * target_rate))
        current_events = int(event_mask.sum())
        if current_events < target_events:
            rank_score = np.maximum(
                event_score.to_numpy(dtype=np.float32),
                continuous_score.to_numpy(dtype=np.float32),
            )
            if "regime_label" in df.columns:
                candidate_mask = np.array(df["regime_label"].astype(str).ne("low_liquidity"), dtype=bool)
            else:
                candidate_mask = np.ones(n, dtype=bool)
            candidate_mask &= np.isfinite(rank_score) & (rank_score > 0.0)
            candidate_mask &= ~event_mask
            need = max(target_events - current_events, 0)
            if need > 0 and bool(np.any(candidate_mask)):
                candidate_idx = np.flatnonzero(candidate_mask)
                order = candidate_idx[np.argsort(rank_score[candidate_idx])[::-1]]
                chosen = order[:need]
                event_mask[chosen] = True
                adaptive_added = int(len(chosen))

    stored_score = np.maximum(
        event_score.to_numpy(dtype=np.float32),
        continuous_score.to_numpy(dtype=np.float32),
    )

    return {
        "event_mask": event_mask,
        "pass_raw_before_adaptive": pass_raw_before_adaptive,
        "stored_score": stored_score.astype(np.float32),
        "threshold_mean": float(np.mean(threshold.to_numpy(dtype=np.float64))),
        "event_score_discrete_mean": float(np.mean(event_score.to_numpy())),
        "stored_score_mean": float(np.mean(stored_score)),
        "adaptive_added": adaptive_added,
        "target_rate_applied": target_rate,
        "scale": scale,
        "shift": shift,
        "hawkes_z_cut": hz,
        "absorb_z_cut": az,
        "kyle_z_cut": kz,
        "cvd_align_cut": cv,
        "mbp_cov_cut_applied": cov_c,
        "mbo_cov_cut_applied": mbo_c,
    }


def session_event_gate_breakdown(
    df: pd.DataFrame,
    *,
    event_mask: np.ndarray,
    pass_raw_before_adaptive: np.ndarray,
    parquet_ev: np.ndarray | None,
    directional: np.ndarray,
) -> pd.DataFrame:
    """جدول تشخيص زمني حسب عمود session (asia / london / overlap / ny / ...)."""
    if "session" not in df.columns:
        return pd.DataFrame()

    sess = df["session"].astype(str)
    rows: list[dict[str, Any]] = []
    for s in sorted(sess.unique()):
        msk = (sess == s).to_numpy()
        n = int(msk.sum())
        if n == 0:
            continue
        row: dict[str, Any] = {
            "session": s,
            "n_rows": n,
            "sim_event_rate_after_adaptive": float(np.mean(event_mask[msk])),
            "sim_pass_raw_share_before_adaptive": float(np.mean(pass_raw_before_adaptive[msk])),
        }
        if "mbp_roll_lob_coverage" in df.columns:
            row["mean_mbp_roll_lob_coverage"] = float(
                pd.to_numeric(df.loc[msk, "mbp_roll_lob_coverage"], errors="coerce").fillna(0.0).mean()
            )
        if "mbo_bar_coverage" in df.columns:
            row["mean_mbo_bar_coverage"] = float(
                pd.to_numeric(df.loc[msk, "mbo_bar_coverage"], errors="coerce").fillna(0.0).mean()
            )
        if parquet_ev is not None:
            row["parquet_is_event_share"] = float(np.mean(parquet_ev[msk] == 1))
            row["agreement_vs_parquet_is_event"] = float(np.mean((parquet_ev[msk] == 1) == event_mask[msk]))
        row["directional_miss_gate_share"] = float(np.mean(directional[msk] & (~event_mask[msk])))
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_float_grid(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep DayTrade Event Gate parameters on existing parquet")
    ap.add_argument("--parquet", required=True, help="day_trading_features*.parquet (sorted by ts_event)")
    ap.add_argument("--out-csv", default=None, help="Write sweep results CSV")
    ap.add_argument("--scale-grid", default="1.0,0.90,0.85,0.75,0.65,0.55")
    ap.add_argument("--shift-grid", default="0.0,-0.03,-0.06,-0.10")
    ap.add_argument("--min-rate-grid", default="0.12,0.20,0.30,0.40,0.50")
    ap.add_argument(
        "--hawkes-z-cut-grid",
        default="1.0",
        help="Comma-separated cuts for hawkes_z > cut (default 1.0; raise = stricter discrete flag)",
    )
    ap.add_argument("--absorb-z-cut-grid", default="1.0", help="absorb_z > cut (default 1.0)")
    ap.add_argument("--kyle-z-cut-grid", default="0.5", help="kyle_z > cut (default 0.5)")
    ap.add_argument("--cvd-align-cut-grid", default="0.6", help="cvd_align > cut (default 0.6)")
    ap.add_argument(
        "--mbp-cov-cut-grid",
        default=str(EVENT_LOB_COVERAGE_CUT),
        help="Comma-separated cuts on mbp_roll_lob_coverage (default from regime EVENT_LOB_COVERAGE_CUT)",
    )
    ap.add_argument(
        "--mbo-cov-cut-grid",
        default=str(EVENT_MBO_COVERAGE_CUT),
        help="Comma-separated cuts on mbo_bar_coverage (default from regime EVENT_MBO_COVERAGE_CUT)",
    )
    ap.add_argument(
        "--session-breakdown",
        action="store_true",
        help="Per-session diagnostic table using the FIRST value from each sweep grid",
    )
    ap.add_argument("--out-session-csv", default=None, help="Write session breakdown table to CSV")
    args = ap.parse_args()

    pq = os.path.abspath(args.parquet)
    if "..." in pq:
        raise SystemExit("Pass full parquet path (no ... placeholder).")
    if not os.path.isfile(pq):
        raise SystemExit(f"parquet not found: {pq}")

    df = pd.read_parquet(pq).sort_values("ts_event").reset_index(drop=True)
    need = {"hawkes_intensity", "absorption_intensity", "kyle_lambda"}
    miss = sorted(need - set(df.columns))
    if miss:
        raise SystemExit(f"parquet missing columns needed to rebuild Event Gate: {miss}")

    bias = (
        pd.to_numeric(df.get("bias_label", DIR_NEUTRAL), errors="coerce").fillna(DIR_NEUTRAL).astype(np.int16).to_numpy()
    )
    parquet_ev = None
    if "is_event" in df.columns:
        parquet_ev = pd.to_numeric(df["is_event"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    directional = (bias == DIR_LONG) | (bias == DIR_SHORT)

    if "tick_count" in df.columns:
        df = df.copy()
        df["mbo_bar_coverage"] = _mbo_bar_coverage_from_tick_series(df["tick_count"])
    elif "num_trades" in df.columns:
        df = df.copy()
        df["mbo_bar_coverage"] = _mbo_bar_coverage_from_tick_series(df["num_trades"])
    elif "mbo_bar_coverage" not in df.columns:
        df = df.copy()
        df["mbo_bar_coverage"] = np.float32(0.0)

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
    if n_combo > 50_000:
        raise SystemExit(f"Too many sweep combinations ({n_combo:,}); narrow grids.")

    rows_out: list[dict[str, Any]] = []
    print("=" * 72)
    print("diagnose_event_gate_daytrade")
    print(f"  rows={len(df):,}  parquet={pq}")
    if parquet_ev is not None:
        print(f"  parquet is_event rate={float(np.mean(parquet_ev == 1)):.1%}")
    print(f"  directional bias_share={float(np.mean(directional)):.1%}")
    print(
        f"  lob_lane: mbp_roll_cov_weight={float(EVENT_SCORE_WEIGHTS.get('mbp_roll_cov_above_cut', 0.0))} "
        f"(EVENT_LOB_COVERAGE_CUT base={EVENT_LOB_COVERAGE_CUT})"
    )
    print(
        f"  mbo_lane: mbo_tick_cov_weight={float(EVENT_SCORE_WEIGHTS.get('mbo_tick_cov_above_cut', 0.0))} "
        f"(EVENT_MBO_COVERAGE_CUT base={EVENT_MBO_COVERAGE_CUT})"
    )
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
                                        m = r["event_mask"]
                                        ev_rate = float(np.mean(m))
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
                                            "target_rate_clipped": r["target_rate_applied"],
                                            "event_rate": ev_rate,
                                            "adaptive_added": r["adaptive_added"],
                                            "thr_mean": r["threshold_mean"],
                                            "event_score_disc_mean": r["event_score_discrete_mean"],
                                            "stored_score_mean": r["stored_score_mean"],
                                            "directional_and_event": float(np.mean(directional & m)),
                                            "directional_given_event": float(np.sum(directional & m) / max(int(m.sum()), 1)),
                                            "neutral_parquet_share": float(np.mean(bias == DIR_NEUTRAL)),
                                            "directional_miss_gate_share": float(np.mean(directional & (~m))),
                                        }
                                        if parquet_ev is not None:
                                            agree = float(np.mean((parquet_ev == 1) == m))
                                            row["agreement_vs_parquet_is_event"] = agree
                                        rows_out.append(row)

    pdf = pd.DataFrame(rows_out)
    pdf = pdf.sort_values(["event_rate", "directional_miss_gate_share"], ascending=[False, True])

    print("\nTop configs by event_rate (then fewer directional rows blocked by gate):")
    cols = [
        "hawkes_z_cut",
        "absorb_z_cut",
        "kyle_z_cut",
        "cvd_align_cut",
        "mbp_cov_cut",
        "mbo_cov_cut",
        "threshold_scale",
        "threshold_shift",
        "min_event_rate",
        "event_rate",
        "adaptive_added",
        "directional_miss_gate_share",
        "agreement_vs_parquet_is_event",
    ]
    cols_show = [c for c in cols if c in pdf.columns]
    print(pdf[cols_show].head(15).to_string(index=False))

    out_csv = args.out_csv or os.path.splitext(pq)[0] + "_event_gate_sweep.csv"
    pdf.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nWrote CSV -> {os.path.abspath(out_csv)}")

    if args.session_breakdown:
        hz0, az0, kz0, cv0 = hawkes_cuts[0], absorb_cuts[0], kyle_cuts[0], cvd_cuts[0]
        cov0, mbo0 = cov_cuts[0], mbo_cuts[0]
        sc0, sh0, mr0 = scales[0], shifts[0], rates[0]
        r0 = simulate_event_gate(
            df,
            threshold_scale=sc0,
            threshold_shift=sh0,
            min_event_rate=mr0,
            hawkes_z_cut=hz0,
            absorb_z_cut=az0,
            kyle_z_cut=kz0,
            cvd_align_cut=cv0,
            mbp_cov_cut=cov0,
            mbo_cov_cut=mbo0,
        )
        print("\n[session_breakdown] Using FIRST grid combo from each axis:")
        print(
            f"  hawkes>{hz0} absorb>{az0} kyle>{kz0} cvd>{cv0} mbp_cov>{cov0} mbo_cov>{mbo0} | "
            f"scale={sc0} shift={sh0} min_rate={mr0}"
        )
        sb = session_event_gate_breakdown(
            df,
            event_mask=r0["event_mask"],
            pass_raw_before_adaptive=r0["pass_raw_before_adaptive"],
            parquet_ev=parquet_ev,
            directional=directional,
        )
        if sb.empty:
            print("  (no 'session' column in parquet — skip)")
        else:
            print(sb.to_string(index=False))
            if args.out_session_csv:
                session_path = os.path.abspath(args.out_session_csv)
                sb.to_csv(session_path, index=False, encoding="utf-8-sig")
                print(f"\nWrote session CSV -> {session_path}")

    print(
        "\nNote: directional_miss_gate_share = share of rows that are LONG/SHORT in parquet "
        "but would be is_event=0 under this gate — lower is better if you trust parquet bias."
    )


if __name__ == "__main__":
    main()
