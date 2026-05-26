"""
تحليل أنماط ما قبل التدريب الكامل — Rolling IC، IC Decay، Regime IC.

الموضع في خط الأنابيب (بعد أن تحدد الفيتشرز التي «تضيف قيمة»):
  ① سبق هذا الملف: ``alpha_validation.py --rigorous`` → قائمة مختصرة (VALID_EDGE / مرشّحون).
  ② هذا الملف: كشف الأنماط على Parquet موسّع وبـ ``--features`` = تلك القائمة فقط —
     أفق العائد (IC Decay)، سلوك النظام (Regime IC)، هل الإشارة حيّة في كل نافذة (Rolling IC).
  ③ ثبات الأنماط: راجع ``pct_same_sign_as_full`` ونوافذ Rolling IC؛ وللثبات الإحصائي الأولي
     استخدم ما زال ``test_stability`` داخل المسار الصارم في alpha_validation؛ كرّر ① على البيانات
     الطويلة إذا احتجت تأكيدًا بعد التوسيع.

يُفضّل تشغيله على بيانات موسّعة (تاريخ طويل) وبعد فلتر الجلسة إن لزم.

يعتمد على alpha_validation لـ Spearman الآمن وتنظيف الأزواج.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_validation import (
    _clean_pair,
    _resolve_ts_series,
    _spearman_safe,
    filter_session,
    load_feature_list,
)


def _sort_by_ts(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_event" not in df.columns:
        raise ValueError("يلزم عمود ts_event.")
    return df.sort_values("ts_event").reset_index(drop=True)


def _forward_cum_ret(r: np.ndarray, h: int, *, start_offset: int = 1) -> np.ndarray:
    """
    في الفهرس i: مجموع h خطوة أمامية ابتداءً من i+start_offset.
    start_offset=1: لا يُدخل عائد الشمعة الحالية (يتماشى مع فيتشر يُعرف عند إغلاق i).
    """
    n = len(r)
    out = np.full(n, np.nan, dtype=float)
    lo = int(start_offset)
    hi = lo + int(h)
    for i in range(n):
        if i + hi > n:
            break
        out[i] = float(np.nansum(r[i + lo : i + hi]))
    return out


def rolling_ic_summary(
    df: pd.DataFrame,
    feature: str,
    forward_col: str,
    *,
    window_bars: int,
    min_window_rows: int | None = None,
) -> dict[str, Any]:
    """
    IC سبيرمان على نوافذ rolling متجاورة زمنيًا (بدون خلط المستقبل داخل النافذة).
    """
    d = _sort_by_ts(df)
    sub = _clean_pair(d, feature, forward_col)
    if len(sub) < window_bars:
        return {
            "n_windows": 0,
            "rolling_ic_mean": np.nan,
            "rolling_ic_std": np.nan,
            "rolling_ic_min": np.nan,
            "rolling_ic_max": np.nan,
            "pct_same_sign_as_full": np.nan,
            "full_sample_ic": np.nan,
            "reason": "صفوف غير كافية",
        }

    x = sub[feature].to_numpy(dtype=float)
    y = sub[forward_col].to_numpy(dtype=float)
    mp = int(min_window_rows) if min_window_rows is not None else max(window_bars // 4, 15)

    ic_full, _ = _spearman_safe(x, y)
    ic_full = float(ic_full) if ic_full == ic_full else np.nan

    window_ics: list[float] = []
    for end in range(mp, len(sub) + 1):
        start = max(0, end - window_bars)
        sl = slice(start, end)
        if end - start < mp:
            continue
        ic, _ = _spearman_safe(x[sl], y[sl])
        if ic == ic and not np.isnan(ic):
            window_ics.append(float(ic))

    if len(window_ics) < 2:
        return {
            "n_windows": len(window_ics),
            "rolling_ic_mean": np.nan,
            "rolling_ic_std": np.nan,
            "rolling_ic_min": np.nan,
            "rolling_ic_max": np.nan,
            "pct_same_sign_as_full": np.nan,
            "full_sample_ic": round(ic_full, 6) if ic_full == ic_full else np.nan,
            "reason": "نوافذ غير كافية",
        }

    arr = np.array(window_ics, dtype=float)
    ref_sign = np.sign(ic_full) if ic_full == ic_full and ic_full != 0 else np.nan
    if ref_sign == ref_sign and ref_sign != 0:
        same = float(np.mean(np.sign(arr) == ref_sign))
    else:
        same = float(np.mean(arr > 0))  # لا IC كلية — نعتمد نسبة النوافذ الموجبة

    return {
        "n_windows": len(window_ics),
        "rolling_ic_mean": round(float(np.mean(arr)), 6),
        "rolling_ic_std": round(float(np.std(arr, ddof=1)), 6) if len(arr) > 1 else 0.0,
        "rolling_ic_min": round(float(np.min(arr)), 6),
        "rolling_ic_max": round(float(np.max(arr)), 6),
        "pct_same_sign_as_full": round(same, 4),
        "full_sample_ic": round(ic_full, 6) if ic_full == ic_full else np.nan,
        "reason": "",
    }


def ic_decay_table(
    df: pd.DataFrame,
    feature: str,
    forward_col: str,
    *,
    max_horizon: int,
    eval_tail_frac: float,
    forward_start_offset: int = 1,
) -> pd.DataFrame:
    """
    صف لكل أفق h: IC بين الفيتشر ومجموع العائد الأمامي لـ h شمعة (عائد 1-bar من forward_col).
    """
    d = _sort_by_ts(df)
    cut = int(len(d) * (1.0 - float(eval_tail_frac)))
    cut = max(cut, 1)
    eval_slice = d.iloc[cut:].copy()

    sub0 = _clean_pair(eval_slice, feature, forward_col)
    if len(sub0) < 30:
        return pd.DataFrame(
            columns=[
                "horizon_bars",
                "ic",
                "p_value",
                "n_valid",
            ]
        )

    r_full = pd.to_numeric(d[forward_col], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []

    for h in range(1, int(max_horizon) + 1):
        tgt = _forward_cum_ret(r_full, h, start_offset=forward_start_offset)
        tmp = d.copy()
        tmp["_tgt_h"] = tgt
        sub = _clean_pair(tmp, feature, "_tgt_h")
        # تقييد تقييم الـ OOS tail على الصفوف التي تقع ضمن فترة التقييم (زمنيًا)
        es_min = eval_slice["ts_event"].min()
        sub = sub[sub["ts_event"] >= es_min]
        if len(sub) < 25:
            rows.append({"horizon_bars": h, "ic": np.nan, "p_value": np.nan, "n_valid": len(sub)})
            continue
        ic, pv = _spearman_safe(
            sub[feature].to_numpy(dtype=float),
            sub["_tgt_h"].to_numpy(dtype=float),
        )
        ic_f = float(ic) if ic == ic else np.nan
        pv_f = float(pv) if pv == pv else np.nan
        rows.append(
            {
                "horizon_bars": h,
                "ic": round(ic_f, 6) if ic_f == ic_f else np.nan,
                "p_value": round(pv_f, 6) if pv_f == pv_f else np.nan,
                "n_valid": len(sub),
            }
        )

    return pd.DataFrame(rows)


def assign_market_regimes(
    df: pd.DataFrame,
    *,
    forward_col: str,
    lookback: int = 20,
    quantile_window: int = 120,
    close_col: str | None = "close",
) -> pd.Series:
    """
    تسمية سببية تقريبًا: volatile إذا تذبذب عالٍ؛ trending إذا انحراف تراكمي كبير؛ وإلا ranging.

    إن وُجد close يُفضّل pct_change؛ وإلا يُستخدم نفس عمود العائد الأمامي كبديل خام.
    """
    d = _sort_by_ts(df)
    lb = max(3, int(lookback))
    wq = max(lb * 4, int(quantile_window))

    if close_col and close_col in d.columns:
        px = pd.to_numeric(d[close_col], errors="coerce").astype(float)
        r = px.pct_change()
    else:
        r = pd.to_numeric(d[forward_col], errors="coerce").astype(float)

    vol = r.rolling(lb).std()
    drift = r.rolling(lb).sum().abs()

    vol_hi = vol.rolling(wq, min_periods=lb * 3).quantile(0.66)
    drift_hi = drift.rolling(wq, min_periods=lb * 3).quantile(0.66)

    reg = pd.Series(np.nan, index=d.index, dtype=object)
    m_vol = vol >= vol_hi
    m_tr = (~m_vol) & (drift >= drift_hi)
    reg = reg.mask(~m_vol & ~m_tr, "ranging")
    reg = reg.mask(m_tr, "trending")
    reg = reg.mask(m_vol, "volatile")
    return reg


def regime_ic_summary(
    df: pd.DataFrame,
    feature: str,
    forward_col: str,
    *,
    regimes: pd.Series,
    eval_tail_frac: float,
    min_rows: int = 40,
) -> pd.DataFrame:
    """جدول IC لكل نظام على ذيل التقييم الزمني فقط."""
    d = _sort_by_ts(df).reset_index(drop=True)
    reg = regimes.reindex(d.index).reset_index(drop=True)
    cut = int(len(d) * (1.0 - float(eval_tail_frac)))
    cut = max(cut, 0)
    tail = d.iloc[cut:].copy()
    tail["_reg"] = reg.iloc[cut:].to_numpy()

    out_rows: list[dict[str, Any]] = []
    for label in ("trending", "ranging", "volatile"):
        part = tail[tail["_reg"] == label]
        sub = _clean_pair(part, feature, forward_col)
        n = len(sub)
        if n < min_rows:
            out_rows.append(
                {
                    "regime": label,
                    "n_rows": n,
                    "ic": np.nan,
                    "p_value": np.nan,
                    "passed_min_rows": False,
                }
            )
            continue
        ic, pv = _spearman_safe(
            sub[feature].to_numpy(dtype=float),
            sub[forward_col].to_numpy(dtype=float),
        )
        out_rows.append(
            {
                "regime": label,
                "n_rows": n,
                "ic": round(float(ic), 6) if ic == ic else np.nan,
                "p_value": round(float(pv), 6) if pv == pv else np.nan,
                "passed_min_rows": True,
            }
        )
    return pd.DataFrame(out_rows)


def run_pattern_analysis(
    df: pd.DataFrame,
    features: list[str],
    forward_col: str,
    *,
    rolling_window_bars: int = 500,
    max_horizon: int = 48,
    eval_tail_frac: float = 0.30,
    regime_lookback: int = 20,
    regime_quantile_window: int = 120,
    close_col: str | None = "close",
    forward_start_offset: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    يعيد (rolling_summary, decay_long, regime_long) كلها بصيغة طويلة + عمود feature.
    """
    df = _sort_by_ts(df)
    if forward_col not in df.columns:
        raise ValueError(f"عمود العائد غير موجود: {forward_col}")

    regimes = assign_market_regimes(
        df,
        forward_col=forward_col,
        lookback=regime_lookback,
        quantile_window=regime_quantile_window,
        close_col=close_col if (close_col is not None and close_col in df.columns) else None,
    )

    roll_rows: list[dict[str, Any]] = []
    decay_parts: list[pd.DataFrame] = []
    regime_parts: list[pd.DataFrame] = []

    for feat in features:
        if feat not in df.columns:
            continue
        dt = ic_decay_table(
            df,
            feat,
            forward_col,
            max_horizon=max_horizon,
            eval_tail_frac=eval_tail_frac,
            forward_start_offset=forward_start_offset,
        )
        half_life: float | int | None = None
        if not dt.empty and "ic" in dt.columns:
            s = dt.dropna(subset=["ic"])
            r1 = s[s["horizon_bars"] == 1]
            if len(r1) > 0:
                ic1 = float(r1["ic"].iloc[0])
                if ic1 == ic1 and abs(ic1) > 1e-12:
                    thr = 0.5 * abs(ic1)
                    crossed = s[s["horizon_bars"] > 1]
                    crossed = crossed[pd.to_numeric(crossed["ic"], errors="coerce").abs() <= thr]
                    if len(crossed) > 0:
                        half_life = int(crossed.iloc[0]["horizon_bars"])

        rs = rolling_ic_summary(
            df,
            feat,
            forward_col,
            window_bars=rolling_window_bars,
        )
        rs["feature"] = feat
        rs["ic_decay_half_life_bars"] = half_life if half_life is not None else np.nan
        roll_rows.append(rs)

        if not dt.empty:
            dt["feature"] = feat
            decay_parts.append(dt)

        rg = regime_ic_summary(
            df,
            feat,
            forward_col,
            regimes=regimes,
            eval_tail_frac=eval_tail_frac,
        )
        rg["feature"] = feat
        regime_parts.append(rg)

    rolling_df = pd.DataFrame(roll_rows)
    decay_long = pd.concat(decay_parts, ignore_index=True) if decay_parts else pd.DataFrame()
    regime_long = pd.concat(regime_parts, ignore_index=True) if regime_parts else pd.DataFrame()

    return rolling_df, decay_long, regime_long


def main() -> None:
    ap = argparse.ArgumentParser(description="Pattern Analysis: Rolling IC, IC Decay, Regime IC")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output-prefix", default=None, help="بادئة ملفات CSV (افتراضي: اسم parquet)")
    ap.add_argument("--session", default="asia")
    ap.add_argument("--session-profile", default="daytrade_default")
    ap.add_argument("--forward-col", default="fwd_ret_clean")
    ap.add_argument("--features", default=None)
    ap.add_argument("--rolling-window", type=int, default=500)
    ap.add_argument("--max-horizon", type=int, default=48)
    ap.add_argument("--eval-tail-frac", type=float, default=0.30)
    ap.add_argument("--regime-lookback", type=int, default=20)
    ap.add_argument("--regime-q-window", type=int, default=120)
    ap.add_argument("--forward-offset", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    df = df.copy()
    df["_alpha_ts"] = _resolve_ts_series(df)
    if "ts_event" not in df.columns:
        df["ts_event"] = df["_alpha_ts"]

    df_s = filter_session(df, args.session, args.session_profile)
    if args.forward_col not in df_s.columns:
        print(f"العمود {args.forward_col} غير موجود.", file=sys.stderr)
        sys.exit(2)

    feats = [f for f in load_feature_list(args.features) if f in df_s.columns]
    if not feats:
        print("لا فيتشر صالحة.", file=sys.stderr)
        sys.exit(2)

    stem = Path(args.parquet).stem
    prefix = args.output_prefix or str(Path(args.parquet).with_name(f"{stem}_pattern"))

    rdf, dlong, reglong = run_pattern_analysis(
        df_s,
        feats,
        args.forward_col,
        rolling_window_bars=args.rolling_window,
        max_horizon=args.max_horizon,
        eval_tail_frac=args.eval_tail_frac,
        regime_lookback=args.regime_lookback,
        regime_quantile_window=args.regime_q_window,
        forward_start_offset=args.forward_offset,
    )

    p_roll = f"{prefix}_rolling_ic.csv"
    p_decay = f"{prefix}_ic_decay.csv"
    p_reg = f"{prefix}_regime_ic.csv"
    rdf.to_csv(p_roll, index=False)
    dlong.to_csv(p_decay, index=False)
    reglong.to_csv(p_reg, index=False)

    print(f"Rows filtered: {len(df_s):,} | features: {len(feats)}")
    print(f"Saved: {p_roll}")
    print(f"Saved: {p_decay}")
    print(f"Saved: {p_reg}")
    print("\nRolling IC (أعلى 15 حسب |rolling_ic_mean|):")
    if not rdf.empty and "rolling_ic_mean" in rdf.columns:
        rdf["_abs"] = pd.to_numeric(rdf["rolling_ic_mean"], errors="coerce").abs()
        print(rdf.sort_values("_abs", ascending=False).head(15).drop(columns=["_abs"]).to_string(index=False))


if __name__ == "__main__":
    main()
