"""
قياس Edge على مستوى جلسة آسيا (صف واحد لكل يوم UTC ضمن نافذة آسيا).

مساران:
  • افتراضي: IC على كل الجلسات + qcut للـ quintile spread ⇒ فيه lookahead على تعريف البUCKET.
  • --rigorous: نفس منطق ``AlphaValidatorRigorous`` (OOS، كوانتايل متوسع، استقرار على Train، FDR)
    على جدول الجلسات — أقرب لـ alpha_validation الصارم لكن الهدف = عائد الجلسة.

تحذير: عدد أيام آسيا أقل بكثير من عدد الشموع ⇒ المعايير أضعف إحصائيًا؛ وسّع التاريخ.

مثال صارم:
  python asia_session_edge.py --parquet FEATURES.parquet --rigorous --ic-period M
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from alpha_validation import (
    DEFAULT_MICRO_FEATURES,
    AlphaValidatorRigorous,
    filter_session,
    load_feature_list,
    _resolve_ts_series,
)


def _aggregate_within_session(g: pd.DataFrame, col: str) -> tuple[float, str]:
    """إرجاع (قيمة مجمّعة، اسم القاعدة)."""
    s = pd.to_numeric(g[col], errors="coerce").astype(np.float64)
    if col == "bar_cvd_delta":
        return float(np.nansum(s.to_numpy())), "sum"
    if col in ("cvd", "session_cvd"):
        if len(s) == 0:
            return float("nan"), "last_minus_first"
        return float(s.iloc[-1] - s.iloc[0]), "last_minus_first"
    return float(np.nanmean(s.to_numpy())), "mean"


def _session_open_close_return(g: pd.DataFrame) -> float:
    g = g.sort_values("_alpha_ts")
    if "open" not in g.columns or "close" not in g.columns:
        raise ValueError("يلزم أعمدة open و close لحساب عائد الجلسة.")
    o = float(pd.to_numeric(g["open"], errors="coerce").iloc[0])
    c = float(pd.to_numeric(g["close"], errors="coerce").iloc[-1])
    return float((c - o) / max(abs(o), 1e-12))


def build_session_table(
    df: pd.DataFrame,
    features: list[str],
    *,
    min_bars_per_session: int,
) -> pd.DataFrame:
    ts = df["_alpha_ts"]
    df = df.copy()
    df["_sess_day"] = ts.dt.normalize()

    rows: list[dict[str, Any]] = []
    for day, g in df.groupby("_sess_day", sort=True):
        g = g.sort_values("_alpha_ts")
        if len(g) < min_bars_per_session:
            continue
        try:
            sess_ret = _session_open_close_return(g)
        except Exception:
            continue
        row: dict[str, Any] = {"sess_day": day, "sess_ret": sess_ret, "n_bars": len(g)}
        for feat in features:
            if feat not in g.columns:
                continue
            val, _rule = _aggregate_within_session(g, feat)
            row[f"agg::{feat}"] = val
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def session_table_to_rigorous_frame(sess: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """جدول بصف واحد لكل جلسة → صيغة AlphaValidatorRigorous (ts_event + أعمدة فيتشر + sess_target)."""
    bridge = sess.copy()
    bridge["ts_event"] = pd.to_datetime(bridge["sess_day"])
    bridge["sess_target"] = pd.to_numeric(bridge["sess_ret"], errors="coerce").astype(np.float64)
    ok: list[str] = []
    for f in features:
        c = f"agg::{f}"
        if c not in bridge.columns:
            continue
        bridge[f] = pd.to_numeric(bridge[c], errors="coerce").astype(np.float64)
        ok.append(f)
    bridge = bridge.sort_values("ts_event").reset_index(drop=True)
    return bridge, ok


def _ic_and_qspread(sess: pd.DataFrame, col: str, quantiles: int) -> dict[str, Any]:
    sub = sess[["sess_ret", col]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(sub)
    if n < 8:
        return {"ic": np.nan, "p_value": np.nan, "n": n, "qspread_bps": np.nan, "pass_n": False}

    ic, p = stats.spearmanr(sub[col].to_numpy(), sub["sess_ret"].to_numpy())

    qspread_bps = np.nan
    if n >= quantiles * 5:
        try:
            sub = sub.copy()
            sub["_q"] = pd.qcut(sub[col], quantiles, labels=False, duplicates="drop")
            mu = sub.groupby("_q", observed=True)["sess_ret"].mean()
            if len(mu) >= 2:
                qspread_bps = float((mu.iloc[-1] - mu.iloc[0]) * 10000.0)
        except (ValueError, TypeError):
            pass

    return {
        "ic": float(ic) if ic == ic else np.nan,
        "p_value": float(p) if p == p else np.nan,
        "n": n,
        "qspread_bps": qspread_bps,
        "pass_n": True,
    }


def _agg_rule_name(feat: str) -> str:
    if feat == "bar_cvd_delta":
        return "sum"
    if feat in ("cvd", "session_cvd"):
        return "last_minus_first"
    return "mean"


def run_ranking(sess: pd.DataFrame, features: list[str], quantiles: int) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    for feat in features:
        col = f"agg::{feat}"
        if col not in sess.columns:
            continue
        rule = _agg_rule_name(feat)
        r = _ic_and_qspread(sess, col, quantiles)
        out_rows.append(
            {
                "feature": feat,
                "session_agg_rule": rule,
                "n_sessions": r["n"],
                "spearman_ic_vs_sess_ret": round(r["ic"], 5) if r["ic"] == r["ic"] else np.nan,
                "p_value": round(r["p_value"], 6) if r["p_value"] == r["p_value"] else np.nan,
                "quintile_ls_spread_bps": round(r["qspread_bps"], 3)
                if r["qspread_bps"] == r["qspread_bps"]
                else np.nan,
            }
        )

    res = pd.DataFrame(out_rows)
    if res.empty:
        return res
    res["abs_ic"] = res["spearman_ic_vs_sess_ret"].abs()
    return res.sort_values("abs_ic", ascending=False).drop(columns=["abs_ic"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Asia session-level edge (legacy ranking أو rigorous OOS)")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--session-profile", default="daytrade_default")
    ap.add_argument("--features", default=None)
    ap.add_argument("--min-bars-per-session", type=int, default=3)
    ap.add_argument("--quantiles", type=int, default=5)
    ap.add_argument(
        "--rigorous",
        action="store_true",
        help="مسار صارم مطابق لفكرة alpha_validation --rigorous (OOS + expanding quantile + FDR)",
    )
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--ic-period", default="M", help="لفترات IC على جدول الجلسات استخدم M أو Q غالبًا")
    ap.add_argument("--min-ic", type=float, default=0.05)
    ap.add_argument("--min-icir", type=float, default=0.28)
    ap.add_argument("--fdr-alpha", type=float, default=0.10)
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--min-train-sessions", type=int, default=12)
    ap.add_argument("--min-test-sessions", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    df = df.copy()
    df["_alpha_ts"] = _resolve_ts_series(df)

    df_asia = filter_session(df, "asia", args.session_profile)
    feats = [f for f in load_feature_list(args.features) if f in df_asia.columns]

    print(f"Asia bars: {len(df_asia):,} | features found: {len(feats)}")
    if len(df_asia) < args.min_bars_per_session:
        print("بيانات آسيا غير كافية.", file=sys.stderr)
        sys.exit(2)

    sess = build_session_table(df_asia, feats, min_bars_per_session=args.min_bars_per_session)
    if sess.empty:
        print("لم يُبنَ أي صف جلسة — تحقق من ts_event والفلتر.", file=sys.stderr)
        sys.exit(2)

    n_sess = len(sess)
    print(f"Sessions (UTC days with Asia window): {n_sess}")
    stem = Path(args.parquet).stem
    out_path = args.output_csv or str(
        Path(args.parquet).with_name(f"{stem}_asia_session_{'rigorous' if args.rigorous else 'legacy'}.csv")
    )

    if args.rigorous:
        bridge, feats_ok = session_table_to_rigorous_frame(sess, feats)
        if len(feats_ok) == 0:
            print("لا أعمدة agg متاحة للمسار الصارم.", file=sys.stderr)
            sys.exit(2)

        n_sp = max(2, min(args.n_splits, max(2, n_sess // 6)))
        q = max(3, int(args.quantiles))
        val = AlphaValidatorRigorous(
            train_ratio=float(args.train_ratio),
            min_ic=float(args.min_ic),
            min_icir=float(args.min_icir),
            n_splits=n_sp,
            n_quantiles=q,
            ic_period=str(args.ic_period),
            min_train_rows=max(6, int(args.min_train_sessions)),
            min_test_rows=max(4, int(args.min_test_sessions)),
            min_ic_period_rows=max(2, min(4, n_sess // 15) or 2),
            tar_min_rows_pre=max(q * 5, min(30, n_sess // 2)),
            tar_min_rows_post=max(q * 4, min(20, n_sess // 3)),
            stab_split_floor=max(2, n_sess // (n_sp * 4)),
            stab_sub_min_rows=max(3, min(8, n_sess // (n_sp * 3))),
            exp_quantile_min_hist=max(q, min(6, n_sess // 5)),
        )
        print(
            f"  [rigorous session] bridge rows={len(bridge)} | "
            f"n_splits={n_sp} | IC-period={args.ic_period!r}"
        )
        try:
            res = val.run_full_validation(
                bridge,
                feats_ok,
                "sess_target",
                fdr_alpha=float(args.fdr_alpha),
                verbose=True,
            )
        except ValueError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)
    else:
        res = run_ranking(sess, feats, args.quantiles)
        print(
            "\nملاحظة: المسار الافتراضي يستخدم qcut على كل الجلسات (lookahead في تعريف الكِنتيل). "
            "استخدم --rigorous لمسار أقرب لـ alpha_validation الصارم.",
        )

    res.to_csv(out_path, index=False)
    sess_out = out_path.replace(".csv", "_session_level.parquet")
    try:
        sess.to_parquet(sess_out, index=False)
    except Exception as e:
        print(f"تخطي حفظ session parquet: {e}", file=sys.stderr)
    else:
        print(f"Saved session table: {sess_out}")

    print("\nأعلى النتائج:\n")
    print(res.head(30).to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
