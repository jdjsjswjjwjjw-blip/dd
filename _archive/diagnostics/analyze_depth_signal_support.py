#!/usr/bin/env python3
"""
analyze_depth_signal_support.py — تطابق عمق السوق مع الإشارات + مدة استمرار الدعم

يقرأ parquet DayTrade ويبني لكل شمعة مقياسًا لـ «هل عمق السوق يدعم اتجاه الصفقة؟»
ثم لصفوف الإشارة (افتراضيًا: train_event_flag ∧ اتجاهي):
  • bars_support_before : عدد الشموع المتتالية التي كانت العمق يؤيد نفس الاتجاه حتى الشمعة السابقة للإشارة
  • bars_support_at_signal : هل الشمعة عند الإشارة نفسها مدعومة (0/1)
  • bars_support_after   : عدد الشموع المتتالية بعد الإشارة التي ظل فيها العمق يؤيد الاتجاه
  • bars_not_hostile_*   : سلسلة متصلة بدون «انعكاس قوي» ضد الصفقة على obi/lob (--anti-reversal-thr، افتراضيًا = obi_thr)
  • bars_calm_depth_*    : إن وُجد --depth-step-eps: ما سبق + خطوة صغيرة |Δobi|,|Δlob|≤ eps بين شمعتين متتاليتين

يتطلب أعمدة شائعة في DayTrade (يُستخدم ما يتوفر): obi, lob_imbalance, bid_wall_strength,
ask_wall_strength، واختياريًا mbp_bar_coverage لتقييم جودة لقطة الكتاب.

مخرجات:
  • PNG: Close + إشارات + لوحات OBI / LOB imbalance / جدار العرض-الطلب / تغطية MBP
  • CSV: صف لكل إشارة بالمدد أعلاه + مقاييس مساعدة

مثال (عتبات افتراضية balanced):

  py -3 analyze_depth_signal_support.py ^
    --parquet "E:\\...\\day_trading_features_mbo2_refinery_latest.parquet" ^
    --manifest "E:\\...\\day_trading_manifest_mbo2_refinery_latest.json" ^
    --out-png "E:\\...\\depth_signal_support.png" ^
    --out-csv "E:\\...\\depth_signal_support_rows.csv"

شدّ الدعم (ملف strict):

  py -3 analyze_depth_signal_support.py ^
    --parquet "..." --manifest "..." ^
    --support-profile strict ^
    --out-png "E:\\...\\depth_signal_support_strict.png" ^
    --out-csv "E:\\...\\depth_signal_support_strict_rows.csv"

strict ≈ obi≥0.15، lob≥0.12، فرق الجدار≥0.22، mbp_bar_coverage≥0.45، mbp_roll_lob_coverage≥0.50 (إن وُجد العمود).
يمكن تجاوز أي قيمة بـ --obi-thr / --lob-thr / --wall-margin / --require-mbp-coverage / --require-roll-lob-coverage.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("❌ مطلوب matplotlib — pip install matplotlib") from e

DIR_LONG = 0
DIR_SHORT = 1
DIR_NEUTRAL = 2

# عتبات جاهزة — يمكن تجاوزها بـ --obi-thr / --lob-thr / ...
SUPPORT_PROFILES: dict[str, dict[str, Any]] = {
    "balanced": {
        "obi_thr": 0.05,
        "lob_thr": 0.05,
        "wall_margin": 0.05,
        "require_mbp_coverage": None,
        "require_roll_lob_coverage": None,
    },
    "strict": {
        "obi_thr": 0.15,
        "lob_thr": 0.12,
        "wall_margin": 0.22,
        "require_mbp_coverage": 0.45,
        "require_roll_lob_coverage": 0.50,
    },
    "relaxed": {
        "obi_thr": 0.02,
        "lob_thr": 0.02,
        "wall_margin": 0.02,
        "require_mbp_coverage": None,
        "require_roll_lob_coverage": None,
    },
}


def _num(s: pd.Series | np.ndarray, index: pd.Index) -> pd.Series:
    return pd.to_numeric(pd.Series(s, index=index), errors="coerce").fillna(0.0)


def depth_supports_direction(
    df: pd.DataFrame,
    *,
    obi_thr: float,
    lob_thr: float,
    wall_margin: float,
    require_mbp_cov: float | None,
    require_roll_cov: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    يُرجع:
      support_long[i], support_short[i] — منطق منفصل حتى نستخدمه مع bias الإشارة.
    LONG مدعوم إذا: obi>thr و lob>thr و (bid_wall > ask_wall + margin)
    SHORT معكوس.
    """
    idx = df.index
    obi = _num(df.get("obi", 0), idx).to_numpy(dtype=np.float64)
    if "lob_imbalance" in df.columns:
        lob = _num(df["lob_imbalance"], idx).to_numpy(dtype=np.float64)
    else:
        lob = obi.copy()
    bw = _num(df.get("bid_wall_strength", 0), idx).to_numpy(dtype=np.float64)
    aw = _num(df.get("ask_wall_strength", 0), idx).to_numpy(dtype=np.float64)

    wall_ok_long = (bw > aw + float(wall_margin))
    wall_ok_short = (aw > bw + float(wall_margin))

    base_long = (obi > float(obi_thr)) & (lob > float(lob_thr)) & wall_ok_long
    base_short = (obi < -float(obi_thr)) & (lob < -float(lob_thr)) & wall_ok_short

    if require_mbp_cov is not None and "mbp_bar_coverage" in df.columns:
        cov = _num(df["mbp_bar_coverage"], idx).to_numpy(dtype=np.float64)
        m = cov >= float(require_mbp_cov)
        base_long &= m
        base_short &= m

    if require_roll_cov is not None and "mbp_roll_lob_coverage" in df.columns:
        rc = _num(df["mbp_roll_lob_coverage"], idx).to_numpy(dtype=np.float64)
        m2 = rc >= float(require_roll_cov)
        base_long &= m2
        base_short &= m2

    return base_long.astype(np.bool_), base_short.astype(np.bool_)


def depth_not_hostile_to_trade(
    df: pd.DataFrame,
    *,
    anti_rev_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    الشمعة «ليست عدائية» لاتجاه افتراضي:
      LONG  — لا انعكاس بيع قوي: obi ≥ -R و lob ≥ -R
      SHORT — لا انعكاس شراء قوي: obi ≤ +R و lob ≤ +R
    (R = anti_rev_thr). لا يستخدم الجدار/MBP هنا — تركيز على عدم قلب الـ imbalance ضد الصفقة.
    """
    idx = df.index
    obi = _num(df.get("obi", 0), idx).to_numpy(dtype=np.float64)
    if "lob_imbalance" in df.columns:
        lob = _num(df["lob_imbalance"], idx).to_numpy(dtype=np.float64)
    else:
        lob = obi.copy()
    r = float(anti_rev_thr)
    long_ok = (obi >= -r) & (lob >= -r)
    short_ok = (obi <= r) & (lob <= r)
    return long_ok.astype(np.bool_), short_ok.astype(np.bool_)


def depth_step_stable(
    df: pd.DataFrame,
    *,
    step_eps: float,
) -> np.ndarray:
    """لكل شمعة j≥1: |Δobi|,|Δlob|≤ eps مقابل الشمعة السابقة؛ الشمعة 0 تُعتبر مستقرة."""
    idx = df.index
    obi = _num(df.get("obi", 0), idx).to_numpy(dtype=np.float64)
    if "lob_imbalance" in df.columns:
        lob = _num(df["lob_imbalance"], idx).to_numpy(dtype=np.float64)
    else:
        lob = obi.copy()
    n = len(df)
    out = np.ones(n, dtype=np.bool_)
    eps = float(step_eps)
    for j in range(1, n):
        out[j] = bool(
            (abs(float(obi[j]) - float(obi[j - 1])) <= eps)
            and (abs(float(lob[j]) - float(lob[j - 1])) <= eps)
        )
    return out


def _quantile_line(name: str, s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return f"  {name}: (empty / all NaN)"
    qs = s.quantile([0.0, 0.05, 0.5, 0.95, 1.0])
    return (
        f"  {name}: n={len(s):,} | min={float(qs.iloc[0]):.5f} | p5={float(qs.iloc[1]):.5f} | "
        f"p50={float(qs.iloc[2]):.5f} | p95={float(qs.iloc[3]):.5f} | max={float(qs.iloc[4]):.5f}"
    )


def print_depth_column_stats(
    df: pd.DataFrame,
    sig_ix: np.ndarray,
    *,
    obi_thr: float,
    lob_thr: float,
    wall_margin: float,
) -> None:
    """
    يؤكد أن أعمدة obi / lob / الجدار مملوءة في الملف، ويعرض توزيعها على كل الشموع المحمّلة
    مقابل مواقع الإشارة فقط (التشخيص ≠ مرور بوابات الاتجاه).
    """
    idx = df.index
    obi = _num(df.get("obi", 0), idx)
    lob = _num(df["lob_imbalance"], idx) if "lob_imbalance" in df.columns else obi.copy()
    bw = _num(df.get("bid_wall_strength", 0), idx)
    aw = _num(df.get("ask_wall_strength", 0), idx)
    wall_delta = bw - aw

    print("\n📈 Depth column stats (raw columns in loaded range — not directional gates)")
    print(_quantile_line("obi", obi))
    print(_quantile_line("lob_imbalance", lob))
    print(_quantile_line("bid_wall - ask_wall", wall_delta))

    n = len(df)
    if n:
        print("\n  Share of bars |metric| > threshold (whole loaded sample, direction-agnostic):")
        print(f"    |obi|  > {obi_thr:g} : {(obi.abs() > float(obi_thr)).mean():.1%}")
        print(f"    |lob|  > {lob_thr:g} : {(lob.abs() > float(lob_thr)).mean():.1%}")
        print(f"    |Δwall| > {wall_margin:g}: {(wall_delta.abs() > float(wall_margin)).mean():.1%}")

    if len(sig_ix):
        print(f"\n  Same metrics at signal rows only (n={len(sig_ix)}):")
        print(_quantile_line("obi", obi.iloc[sig_ix]))
        print(_quantile_line("lob_imbalance", lob.iloc[sig_ix]))
        print(_quantile_line("bid_wall - ask_wall", wall_delta.iloc[sig_ix]))
    print()


def row_depth_gate_components(
    df: pd.DataFrame,
    pos: int,
    bias_val: int,
    *,
    obi_thr: float,
    lob_thr: float,
    wall_margin: float,
    mbp_req: float | None,
    roll_req: float | None,
) -> dict[str, bool]:
    """نفس منطق depth_supports_direction لكن كقيم منفصلة على صف واحد (للتشخيص)."""
    ix = df.index
    obi_v = float(_num(df.get("obi", 0), ix).iloc[pos])
    if "lob_imbalance" in df.columns:
        lob_v = float(_num(df["lob_imbalance"], ix).iloc[pos])
    else:
        lob_v = obi_v
    bw = float(_num(df.get("bid_wall_strength", 0), ix).iloc[pos])
    aw = float(_num(df.get("ask_wall_strength", 0), ix).iloc[pos])

    if int(bias_val) == DIR_LONG:
        pass_obi = obi_v > float(obi_thr)
        pass_lob = lob_v > float(lob_thr)
        pass_wall = bw > aw + float(wall_margin)
    elif int(bias_val) == DIR_SHORT:
        pass_obi = obi_v < -float(obi_thr)
        pass_lob = lob_v < -float(lob_thr)
        pass_wall = aw > bw + float(wall_margin)
    else:
        return {
            "pass_obi": False,
            "pass_lob": False,
            "pass_wall": False,
            "pass_mbp_bar": False,
            "pass_roll_lob": False,
        }

    if mbp_req is not None and "mbp_bar_coverage" in df.columns:
        cov = float(_num(df["mbp_bar_coverage"], ix).iloc[pos])
        pass_mbp = cov >= float(mbp_req)
    else:
        pass_mbp = True

    if roll_req is not None and "mbp_roll_lob_coverage" in df.columns:
        rc = float(_num(df["mbp_roll_lob_coverage"], ix).iloc[pos])
        pass_roll = rc >= float(roll_req)
    else:
        pass_roll = True

    return {
        "pass_obi": bool(pass_obi),
        "pass_lob": bool(pass_lob),
        "pass_wall": bool(pass_wall),
        "pass_mbp_bar": bool(pass_mbp),
        "pass_roll_lob": bool(pass_roll),
    }


def consecutive_true_forward(mask: np.ndarray, start_ix: int, cap: int) -> int:
    c = 0
    n = len(mask)
    last = min(int(start_ix) + int(cap), n)
    for j in range(int(start_ix), last):
        if mask[j]:
            c += 1
        else:
            break
    return c


def run_streak_before(mask: np.ndarray, end_ix: int) -> tuple[int, int]:
    """
    end_ix = موقع الإشارة.
    bars قبل الإشارة بدون احتساب شمعة الإشارة: نعد من end_ix-1 للخلف.
    عند الإشارة: mask[end_ix].
    """
    at_sig = int(mask[int(end_ix)]) if 0 <= end_ix < len(mask) else 0
    before = 0
    for j in range(int(end_ix) - 1, -1, -1):
        if mask[j]:
            before += 1
        else:
            break
    return before, at_sig


def run_streak_after(mask: np.ndarray, start_ix: int, cap: int) -> int:
    """يبدأ من الشمعة التالية للإشارة."""
    return consecutive_true_forward(mask, int(start_ix) + 1, cap)


def main() -> None:
    ap = argparse.ArgumentParser(description="Depth vs signal alignment + support persistence")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--manifest", default=None, help="يقرأ freq إن وُجد")
    ap.add_argument("--freq", default=None)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument(
        "--support-profile",
        choices=tuple(SUPPORT_PROFILES.keys()),
        default="balanced",
        help="balanced | strict (عتبات أشد + تغطية MBP/ROLL) | relaxed",
    )
    ap.add_argument(
        "--obi-thr",
        type=float,
        default=None,
        help="يتجاوز الملف الشخصي؛ إن لُغي يُؤخذ من --support-profile",
    )
    ap.add_argument("--lob-thr", type=float, default=None, help="يتجاوز الملف الشخصي")
    ap.add_argument("--wall-margin", type=float, default=None, help="يتجاوز الملف الشخصي")
    ap.add_argument(
        "--require-mbp-coverage",
        type=float,
        default=None,
        help="≥ هذا القدر من mbp_bar_coverage؛ None = استخدم قيمة الملف الشخصي (strict يفرض تلقائيًا)",
    )
    ap.add_argument(
        "--require-roll-lob-coverage",
        type=float,
        default=None,
        help="≥ mbp_roll_lob_coverage إن وُجد العمود؛ None = من الملف الشخصي",
    )
    ap.add_argument("--forward-cap-bars", type=int, default=48, help="أقصى شموع للأمام لقياس استمرار الدعم")
    ap.add_argument(
        "--signal-mask",
        choices=("train_event_directional", "all_directional"),
        default="train_event_directional",
        help="صفوف الإشارة: train_event_flag∧اتجاهي أو كل الصفوف الاتجاهية",
    )
    ap.add_argument(
        "--print-column-stats",
        action="store_true",
        help="اطبع quantiles لـ obi / lob / فرق الجدار على كل الشموع المحمّلة مقابل صفوف الإشارة",
    )
    ap.add_argument(
        "--anti-reversal-thr",
        type=float,
        default=None,
        help="عتبة انعكاس ضد الصفقة على obi/lob (طرّ LONG إذا obi أو lob < −R؛ SHORT العكس). None = نفس obi المحلول",
    )
    ap.add_argument(
        "--depth-step-eps",
        type=float,
        default=None,
        help="إن وُجد: يضيف مقاييس «ثبات خطوة» — بين شمعتين متتاليتين |Δobi|,|Δlob|≤ هذا مع عدم الانعكاس",
    )
    args = ap.parse_args()

    def _no_placeholder_paths(paths: dict[str, str | None]) -> None:
        for label, raw in paths.items():
            if not raw:
                continue
            s = str(raw).strip()
            if "..." in s:
                raise SystemExit(
                    f"❌ مسار {label} يحتوي على «...» — هذا مثال في التوثيق وليس مسارًا حقيقيًا.\n"
                    "   انسخ المسار الكامل من مستكشف الملفات، مثلًا:\n"
                    "   E:\\QuantSystem-master (3) (2)\\QuantSystem-master\\pipeline_mbo2_refinery_latest\\daytrade_run\\"
                    "day_trading_features_mbo2_refinery_latest.parquet"
                )

    _no_placeholder_paths(
        {
            "parquet": args.parquet,
            "manifest": args.manifest,
            "out-png": args.out_png,
            "out-csv": args.out_csv,
        }
    )

    prof = SUPPORT_PROFILES[str(args.support_profile)]
    obi_thr = float(args.obi_thr if args.obi_thr is not None else prof["obi_thr"])
    lob_thr = float(args.lob_thr if args.lob_thr is not None else prof["lob_thr"])
    wall_margin = float(args.wall_margin if args.wall_margin is not None else prof["wall_margin"])
    mbp_req = args.require_mbp_coverage if args.require_mbp_coverage is not None else prof["require_mbp_coverage"]
    roll_req = (
        args.require_roll_lob_coverage
        if args.require_roll_lob_coverage is not None
        else prof["require_roll_lob_coverage"]
    )
    anti_r = float(args.anti_reversal_thr if args.anti_reversal_thr is not None else obi_thr)
    step_eps = args.depth_step_eps

    pq = os.path.abspath(args.parquet)
    if not os.path.isfile(pq):
        raise SystemExit(f"❌ parquet not found: {pq}")

    freq = args.freq or "5min"
    if args.manifest and os.path.isfile(args.manifest):
        import json

        with open(args.manifest, encoding="utf-8") as f:
            freq = json.load(f).get("freq") or freq

    df = pd.read_parquet(pq)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.sort_values("ts_event").reset_index(drop=True)

    bias = pd.to_numeric(df.get("bias_label", DIR_NEUTRAL), errors="coerce").fillna(DIR_NEUTRAL).astype(np.int8).to_numpy()
    te = pd.to_numeric(df.get("train_event_flag", 0), errors="coerce").fillna(0).astype(np.int8).to_numpy()

    long_sup, short_sup = depth_supports_direction(
        df,
        obi_thr=obi_thr,
        lob_thr=lob_thr,
        wall_margin=wall_margin,
        require_mbp_cov=mbp_req,
        require_roll_cov=roll_req,
    )

    long_nh, short_nh = depth_not_hostile_to_trade(df, anti_rev_thr=anti_r)
    calm_long = calm_short = None
    if step_eps is not None:
        stab = depth_step_stable(df, step_eps=float(step_eps))
        calm_long = long_nh & stab
        calm_short = short_nh & stab

    if args.signal_mask == "train_event_directional":
        sig_mask = (te == 1) & np.isin(bias, [DIR_LONG, DIR_SHORT])
    else:
        sig_mask = np.isin(bias, [DIR_LONG, DIR_SHORT])

    sig_ix = np.flatnonzero(sig_mask)

    if args.print_column_stats:
        print_depth_column_stats(
            df,
            sig_ix,
            obi_thr=obi_thr,
            lob_thr=lob_thr,
            wall_margin=wall_margin,
        )

    rows: list[dict[str, Any]] = []

    for i in sig_ix:
        b = int(bias[i])
        if b == DIR_NEUTRAL:
            continue
        mdir = long_sup if b == DIR_LONG else short_sup
        before, at_sig = run_streak_before(mdir, i)
        after = run_streak_after(mdir, i, args.forward_cap_bars)

        m_nh = long_nh if b == DIR_LONG else short_nh
        nh_before, nh_at = run_streak_before(m_nh, i)
        nh_after = run_streak_after(m_nh, i, args.forward_cap_bars)

        row_out: dict[str, Any] = {}
        if calm_long is not None and calm_short is not None:
            m_calm = calm_long if b == DIR_LONG else calm_short
            cb_before, cb_at = run_streak_before(m_calm, i)
            cb_after = run_streak_after(m_calm, i, args.forward_cap_bars)
            row_out.update(
                {
                    "bars_calm_depth_before": cb_before,
                    "bars_calm_depth_at_signal": int(cb_at),
                    "bars_calm_depth_after": cb_after,
                }
            )

        gates = row_depth_gate_components(
            df,
            i,
            b,
            obi_thr=obi_thr,
            lob_thr=lob_thr,
            wall_margin=wall_margin,
            mbp_req=mbp_req,
            roll_req=roll_req,
        )
        pass_all = all(gates.values())
        rows.append(
            {
                "ts_event": df["ts_event"].iloc[i],
                "bias_label": b,
                "bias_name": "LONG" if b == DIR_LONG else "SHORT",
                "train_event_flag": int(te[i]),
                "bars_depth_support_strict_before": before,
                "bars_depth_support_at_signal": int(at_sig),
                "bars_depth_support_strict_after": after,
                "bars_not_hostile_before": nh_before,
                "bars_not_hostile_at_signal": int(nh_at),
                "bars_not_hostile_after": nh_after,
                **row_out,
                "pass_obi": int(gates["pass_obi"]),
                "pass_lob": int(gates["pass_lob"]),
                "pass_wall": int(gates["pass_wall"]),
                "pass_mbp_bar": int(gates["pass_mbp_bar"]),
                "pass_roll_lob": int(gates["pass_roll_lob"]),
                "pass_all_gates": int(pass_all),
                "obi": float(_num(df.get("obi", 0), df.index).iloc[i]),
                "lob_imbalance": float(_num(df.get("lob_imbalance", 0), df.index).iloc[i]),
                "bid_wall_strength": float(_num(df.get("bid_wall_strength", 0), df.index).iloc[i]),
                "ask_wall_strength": float(_num(df.get("ask_wall_strength", 0), df.index).iloc[i]),
                "mbp_bar_coverage": float(_num(df.get("mbp_bar_coverage", np.nan), df.index).iloc[i])
                if "mbp_bar_coverage" in df.columns
                else np.nan,
                "mbp_roll_lob_coverage": float(_num(df.get("mbp_roll_lob_coverage", np.nan), df.index).iloc[i])
                if "mbp_roll_lob_coverage" in df.columns
                else np.nan,
                "event_score": float(_num(df.get("event_score", np.nan), df.index).iloc[i])
                if "event_score" in df.columns
                else np.nan,
            }
        )

    out_csv = args.out_csv or os.path.splitext(args.out_png)[0] + "_rows.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ── Figure ──────────────────────────────────────────────────────────────
    ts = df["ts_event"]
    close = pd.to_numeric(df["close"], errors="coerce")

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), dpi=120, sharex=True)
    ax0, ax1, ax2, ax3 = axes

    ax0.plot(ts, close, color="#263238", linewidth=1.1, label="close")
    long_sig = sig_mask & (bias == DIR_LONG)
    short_sig = sig_mask & (bias == DIR_SHORT)
    ax0.scatter(ts[long_sig], close[long_sig], color="#2e7d32", s=90, marker="^", zorder=5, label="signal LONG")
    ax0.scatter(ts[short_sig], close[short_sig], color="#c62828", s=90, marker="v", zorder=5, label="signal SHORT")
    ax0.set_ylabel("close")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(True, alpha=0.2)
    mbp_s = f"{mbp_req:.2f}" if mbp_req is not None else "off"
    roll_s = f"{roll_req:.2f}" if roll_req is not None else "off"
    ax0.set_title(
        f"profile={args.support_profile} | obi≥{obi_thr} lob≥{lob_thr} wallΔ≥{wall_margin} "
        f"| mbp_bar≥{mbp_s} roll≥{roll_s} | freq={freq}",
        fontsize=10,
    )

    obi_plot = _num(df.get("obi", 0), df.index)
    ax1.plot(ts, obi_plot, color="#1565c0", linewidth=1.0, label="obi")
    ax1.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax1.axhline(obi_thr, color="green", linewidth=0.6, linestyle=":", alpha=0.7)
    ax1.axhline(-obi_thr, color="red", linewidth=0.6, linestyle=":", alpha=0.7)
    ax1.set_ylabel("OBI")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)

    lob_plot = _num(df["lob_imbalance"], df.index) if "lob_imbalance" in df.columns else obi_plot.copy()
    ax2.plot(ts, lob_plot, color="#6a1b9a", linewidth=1.0, label="lob_imbalance")
    ax2.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax2.axhline(lob_thr, color="green", linewidth=0.6, linestyle=":", alpha=0.7)
    ax2.axhline(-lob_thr, color="red", linewidth=0.6, linestyle=":", alpha=0.7)
    ax2.set_ylabel("LOB imb.")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.2)

    bw = _num(df.get("bid_wall_strength", 0), df.index)
    aw = _num(df.get("ask_wall_strength", 0), df.index)
    ax3.plot(ts, bw, color="#2e7d32", linewidth=1.0, label="bid_wall")
    ax3.plot(ts, aw, color="#c62828", linewidth=1.0, label="ask_wall")
    if "mbp_bar_coverage" in df.columns:
        cov = _num(df["mbp_bar_coverage"], df.index)
        ax3b = ax3.twinx()
        ax3b.fill_between(ts, 0.0, cov, color="#90caf9", alpha=0.25, label="mbp_bar_coverage")
        ax3b.set_ylim(0.0, 1.05)
        ax3b.set_ylabel("MBP cov")
        ax3b.legend(loc="upper right", fontsize=8)

    ax3.set_ylabel("wall strength")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    fig.autofmt_xdate()
    fig.tight_layout()
    out_png = os.path.abspath(args.out_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # ملخص في الطرفية
    pdf = pd.DataFrame(rows)
    print("=" * 72)
    print("📊 Depth–signal support summary")
    print("=" * 72)
    print(f"  Bars total        : {len(df):,}")
    print(f"  support-profile    : {args.support_profile}")
    print(f"  Resolved thresholds: obi={obi_thr} | lob={lob_thr} | wall_margin={wall_margin}")
    print(f"                       mbp_bar_cov≥{mbp_s} | roll_lob_cov≥{roll_s}")
    print(f"  Signal rows used  : {len(rows):,} | mask={args.signal_mask}")
    if len(pdf):
        print(
            f"  Mean support bars BEFORE signal : {pdf['bars_depth_support_strict_before'].mean():.2f} "
            f"(median {pdf['bars_depth_support_strict_before'].median():.0f})"
        )
        print(
            f"  Mean support bars AFTER signal  : {pdf['bars_depth_support_strict_after'].mean():.2f} "
            f"(median {pdf['bars_depth_support_strict_after'].median():.0f}) "
            f"(cap forward={args.forward_cap_bars})"
        )
        print(f"  Share depth OK AT signal bar    : {pdf['bars_depth_support_at_signal'].mean():.1%}")
        if "bars_not_hostile_before" in pdf.columns:
            print(
                f"\n  🧭 بدون انعكاس قوي ضد الصفقة (obi/lob؛ R={anti_r:g}) — متوسط الشموع المتصلة:"
            )
            print(
                f"      قبل الإشارة : {pdf['bars_not_hostile_before'].mean():.2f} "
                f"(median {pdf['bars_not_hostile_before'].median():.0f})"
            )
            print(
                f"      بعد الإشارة : {pdf['bars_not_hostile_after'].mean():.2f} "
                f"(median {pdf['bars_not_hostile_after'].median():.0f}) "
                f"| عند الإشارة OK: {pdf['bars_not_hostile_at_signal'].mean():.1%}"
            )
        if step_eps is not None and "bars_calm_depth_before" in pdf.columns:
            print(
                f"\n  🝘 ثبات خطوة + بدون انعكاس (|Δobi|,|Δlob|≤{float(step_eps):g}) — متوسط الشموع المتصلة:"
            )
            print(
                f"      قبل الإشارة : {pdf['bars_calm_depth_before'].mean():.2f} "
                f"(median {pdf['bars_calm_depth_before'].median():.0f})"
            )
            print(
                f"      بعد الإشارة : {pdf['bars_calm_depth_after'].mean():.2f} "
                f"(median {pdf['bars_calm_depth_after'].median():.0f}) "
                f"| عند الإشارة OK: {pdf['bars_calm_depth_at_signal'].mean():.1%}"
            )
    if len(pdf) and "pass_obi" in pdf.columns:
        print("\n  🔍 تفكيك الشروط على صفوف الإشارة فقط (كل عمود = نسبة المرور):")
        for col in ("pass_obi", "pass_lob", "pass_wall", "pass_mbp_bar", "pass_roll_lob", "pass_all_gates"):
            if col in pdf.columns:
                print(f"      {col:16s}: {float(pd.to_numeric(pdf[col], errors='coerce').mean()):.1%}")
        worst = None
        worst_r = 2.0
        for col, label in [
            ("pass_obi", "OBI"),
            ("pass_lob", "LOB imbalance"),
            ("pass_wall", "جدار الشراء/البيع"),
            ("pass_mbp_bar", "mbp_bar_coverage"),
            ("pass_roll_lob", "mbp_roll_lob_coverage"),
        ]:
            if col not in pdf.columns:
                continue
            r = float(pd.to_numeric(pdf[col], errors="coerce").mean())
            if r < worst_r:
                worst_r = r
                worst = label
        if worst is not None and "pass_all_gates" in pdf.columns and float(pdf["pass_all_gates"].mean()) < 0.01:
            print(
                f"\n  💡 أقل شرط تحققًا: «{worst}» — AND على كل البوابات؛ إن كانت obi/lob/wall كلها ~0٪ "
                "والـ CSV يُظهر قيمًا غير صفرية، فالخلل في العتبات/اتجاه الصف مقابل المقياس وليس «غياب العمود». "
                "جرّب --support-profile balanced أو --print-column-stats لمقارنة التوزيع على كل الشموع."
            )
    print(f"\n  PNG → {out_png}")
    print(f"  CSV → {os.path.abspath(out_csv)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
