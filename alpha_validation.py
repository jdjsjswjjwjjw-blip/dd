"""
تشخيص Edge / Alpha على ميزات جدولية — فلتر جلسة (مثل آسيا UTC) وأي Parquet.
استخدم ``--session all`` لتشغيل المسار على **كل الصفوف** في الملف دون تقييد نافذة يومية.

ترتيب العمل الموصى به (منطقيًا وليس شرطًا تقنيًا في سطر واحد):
  ① التحقق من الفيتشرز التي تضيف قيمة — هذا الملف مع ``--rigorous``:
     IC/TAR على الاختبار فقط، استقرار أولي على الـ Train، FDR.
     المخرجات: عمود VALID_EDGE وقائمة مرشّحين؛ احفظها في ملف أسماء (--features لاحقًا).
  ② كشف الأنماط — بعد توسيع التاريخ: ``pattern_analysis.py`` على القائمة المختصرة فقط
     (Rolling IC، IC Decay، Regime IC). أو نفس الفكرة عبر ``--pattern-analysis`` هنا بعد الاختبار.
  ③ التحقق من ثبات الأنماط — الجزء الأول مدمج في ``--rigorous`` (test_stability).
     للثبات «حسب النمط»: راجع Rolling IC و Regime IC في مخرجات المرحلة ②؛ وكرّر ① على البيانات
     الموسّعة وبنفس القائمة المختصرة إذا تغيّر طول العينة بشكل كبير.

مساران:
  • افتراضي (Legacy): IC / qcut / تقسيمات بسيطة — سريع للمسح.
  • صارم (--rigorous): OOS للـ IC وTAR، كوانتايل متوسع (بدون lookahead)،
    استقرار على Train فقط، FDR (Benjamini–Hochberg) إن وُجد statsmodels.

تحليل أنماط (--pattern-analysis): Rolling IC، IC Decay (أفق العائد)، Regime IC؛
يُصدَّر إلى CSV بجانب ملف النتائج (يفضّل تشغيله بعد اختصار قائمة الفيتشرز يدويًا).

تعدد النوافذ الزمنية دفعة واحدة: ``--all-session-windows`` في هذا الملف (أو ``session_feature_validator.py`` كـ wrapper) يمرّ على آسيا/لندن/نيويورك والتقاطعات ويُخرج ``session_feature_map.json``.

فحص غير خطي (خمسينات + KNN/RF): ``--nonlinear-probe`` على جلسة واحدة، أو ``--nonlinear-all-session-windows`` لكل النوافذ والتقاطعات — مدمج هنا.

علاقات شرطية (طبقتين — فلتر سعر + تدفق): ``--nonlinear-two-layer`` مع أحد مسارات الـ nonlinear أعلاه؛
فلتر المنطقة إما ``dist_to_pdh`` / ``--nl-zone-feature`` أو أوضاع جلسة —
``asia_near_high`` / ``asia_near_low`` / ``london_near_high`` / ``london_near_low``
(مسافة السعر إلى قمة/قاع **جاري** داخل نفس يوم التداول داخل نافذة الجلسة)، أو ``--nl-sweep-asia-london-extremes`` مع ``--nonlinear-all-session-windows``
ليمرّ على قرب القمة والقاع لآسيا ولندن تلقائيًا.

للهدف يُفضّل **`fwd_ret_clean`** من prepare_day_trading؛ أو **`forward_return`** للمقارنة.

أمثلة (جلسة آسيا + مسار صارم):
  python alpha_validation.py --parquet FEATURES.parquet --session asia \\
      --forward-col fwd_ret_clean --rigorous --train-ratio 0.70

  python alpha_validation.py ... --session asia --legacy
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests

    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

from modules.feature_artifact_v19 import load_feature_artifact

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# يطابق prepare_day_trading.SESSION_PROFILES — أوقات UTC
SESSION_PROFILES: dict[str, dict[str, tuple[str, str]]] = {
    "daytrade_default": {
        "asia": ("00:00", "07:00"),
        "london": ("07:00", "12:00"),
        "overlap": ("12:00", "16:00"),
        "london_ny": ("13:30", "16:00"),
        "ny": ("13:30", "20:00"),
    },
    "stage1_like": {
        "asia": ("00:00", "08:00"),
        "london": ("08:00", "13:00"),
        "overlap": ("13:00", "16:00"),
        "london_ny": ("13:30", "16:00"),
        "ny": ("13:30", "20:00"),
    },
}


def _session_time_mask(times: pd.Series, start_hhmm: str, end_hhmm: str) -> pd.Series:
    start_t = pd.to_datetime(start_hhmm).time()
    end_t = pd.to_datetime(end_hhmm).time()
    if start_t <= end_t:
        return (times >= start_t) & (times < end_t)
    return (times >= start_t) | (times < end_t)


def _resolve_ts_series(df: pd.DataFrame) -> pd.Series:
    if "ts_event" in df.columns:
        return pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.to_datetime(df.index, utc=True, errors="coerce").tz_localize(None)
    raise ValueError("لا يوجد ts_event ولا فهرس زمني؛ أضف عمود ts_event أو استخدم Parquet بفهرس زمني.")


def filter_session(
    df: pd.DataFrame,
    session: str,
    profile: str,
) -> pd.DataFrame:
    session = session.lower().strip()
    if session == "all":
        out = df.copy()
        ts = _resolve_ts_series(df)
        out["_alpha_ts"] = ts.loc[out.index]
        return out
    if profile not in SESSION_PROFILES:
        raise ValueError(f"session-profile غير معروف: {profile}")
    windows = SESSION_PROFILES[profile]
    if session not in windows:
        raise ValueError(
            f"جلسة غير معروفة: {session}. الخيارات: asia, london, overlap, london_ny, ny, all"
        )
    start_hhmm, end_hhmm = windows[session]
    ts = _resolve_ts_series(df)
    tcol = ts.dt.time
    mask = _session_time_mask(tcol, start_hhmm, end_hhmm)
    out = df.loc[mask.fillna(False)].copy()
    out["_alpha_ts"] = ts.loc[out.index]
    return out


try:
    from prepare_day_trading import ORIGINAL_FEATURES as _ORIGINAL_FEATURES_DT

    DEFAULT_MICRO_FEATURES: list[str] = list(_ORIGINAL_FEATURES_DT)
except ImportError:
    DEFAULT_MICRO_FEATURES: list[str] = [
        "cvd",
        "session_cvd",
        "absorption_intensity",
        "cancel_ratio",
        "spoofing_ratio",
        "spoofing_duration",
        "liquidity_trap",
        "kyle_lambda",
        "hawkes_intensity",
        "vnet",
        "micro_atr",
        "volume_burst",
        "inter_event_time",
        "fisher_signal",
        "anomaly",
        "cvd_momentum",
        "cvd_price_divergence",
        "trend_strength",
        "correction_depth",
        "liquidity_sweep",
        "obi",
        "bid_wall_strength",
        "ask_wall_strength",
        "distance_to_wall",
        "gap_size",
        "liquidity_density",
        "micro_price",
        "current_vwap",
        "vwap_z_score",
        "pdh",
        "pdl",
        "dist_to_pdh",
        "price_position",
        "tick_count",
        "mbp_bar_coverage",
        "mbo_bar_coverage",
        "mbp_roll_lob_coverage",
        "liquidity_gaps",
        "is_london",
        "is_overlap",
        "is_ny",
    ]


def load_feature_list(arg: str | None) -> list[str]:
    if not arg:
        return list(DEFAULT_MICRO_FEATURES)
    p = Path(arg)
    if p.is_file():
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x) for x in data]
            raise ValueError("JSON يجب أن يكون قائمة أسماء أعمدة")
        return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    return [x.strip() for x in arg.split(",") if x.strip()]


# ─── كل الجلسات + التقاطعات (بدون ملف خارجي) ───────────────────────────────

MULTI_SESSION_WINDOWS: dict[str, tuple[str, str]] = {
    "asia": ("00:00", "07:00"),
    "london": ("07:00", "12:00"),
    "ny": ("13:30", "20:00"),
    "asia_london": ("06:00", "09:00"),
    "london_ny": ("13:30", "16:00"),
    "london_open": ("07:00", "09:00"),
    "ny_open": ("13:30", "15:30"),
    "end_of_day": ("19:00", "21:00"),
}
MULTI_SESSION_MIN_ROWS: dict[str, int] = {
    "asia": 80,
    "london": 80,
    "ny": 80,
    "asia_london": 40,
    "london_ny": 40,
    "london_open": 40,
    "ny_open": 40,
    "end_of_day": 30,
}

_MULTI_SINGLE_NAMES = frozenset(
    {"asia", "london", "ny", "london_open", "ny_open", "end_of_day"}
)
_MULTI_OVERLAP_NAMES = frozenset({"asia_london", "london_ny"})


def _filter_df_time_window(df: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    out = df.copy()
    ts = _resolve_ts_series(out)
    if "ts_event" not in out.columns:
        out["ts_event"] = ts
    tcol = ts.dt.time
    mask = _session_time_mask(tcol, start_hhmm, end_hhmm)
    out = out.loc[mask.fillna(False)].copy()
    out["_alpha_ts"] = ts.loc[out.index]
    return out.reset_index(drop=True)


def _rigorous_kwargs_from_n_rows(n_rows: int, train_ratio: float) -> dict[str, Any]:
    n_rows = int(n_rows)
    cut = int(n_rows * float(train_ratio))
    test_n = n_rows - cut
    min_train_rows = max(25, min(cut - 5, int(cut * 0.92)))
    min_test_rows = max(12, min(test_n - 5, int(test_n * 0.92)))
    min_train_rows = min(min_train_rows, cut - 3)
    min_test_rows = min(min_test_rows, test_n - 3)
    min_train_rows = max(25, min_train_rows)
    min_test_rows = max(12, min_test_rows)
    if cut < min_train_rows or test_n < min_test_rows:
        raise ValueError(f"تقسيم غير ممكن: n={n_rows}, cut={cut}, test={test_n}")

    n_sp = max(2, min(4, max(2, n_rows // 400)))
    q = 5
    return {
        "n_splits": n_sp,
        "min_train_rows": min_train_rows,
        "min_test_rows": min_test_rows,
        "min_ic_period_rows": max(6, min(12, n_rows // 100)),
        "tar_min_rows_pre": max(q * 8, min(120, n_rows // 3)),
        "tar_min_rows_post": max(q * 6, min(80, n_rows // 4)),
        "stab_split_floor": max(10, n_rows // (n_sp * 6)),
        "stab_sub_min_rows": max(12, min(20, n_rows // (n_sp * 5))),
        "exp_quantile_min_hist": max(q, min(16, n_rows // 40)),
    }


def run_multi_session_windows(
    df: pd.DataFrame,
    features: list[str],
    forward_col: str,
    *,
    train_ratio: float,
    min_ic: float,
    min_icir: float,
    ic_period: str,
    output_dir: str | Path,
    verbose_each: bool,
) -> dict[str, pd.DataFrame]:
    """يشغّل AlphaValidatorRigorous على MULTI_SESSION_WINDOWS ويكتب CSV لكل نافذة."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, pd.DataFrame] = {}

    for session_name, (start, end) in MULTI_SESSION_WINDOWS.items():
        print(f"\n{'═' * 50}")
        print(f"⏱  {session_name.upper()}  [{start} → {end}]")

        df_w = _filter_df_time_window(df, start, end)
        min_r = MULTI_SESSION_MIN_ROWS.get(session_name, 50)

        if len(df_w) < min_r:
            print(f"  ⚠️  صفوف غير كافية: {len(df_w)} < {min_r} — تخطي")
            continue

        feats_ok = [f for f in features if f in df_w.columns]
        if forward_col not in df_w.columns:
            print(f"  ⚠️  عمود الهدف مفقود: {forward_col} — تخطي")
            continue

        print(f"  صفوف: {len(df_w):,} | features: {len(feats_ok)}")

        try:
            kw = _rigorous_kwargs_from_n_rows(len(df_w), train_ratio)
        except ValueError as e:
            print(f"  ⚠️  {e}")
            continue

        try:
            val = AlphaValidatorRigorous(
                train_ratio=train_ratio,
                min_ic=min_ic,
                min_icir=min_icir,
                ic_period=ic_period,
                **kw,
            )
            res = val.run_full_validation(
                df_w,
                feats_ok,
                forward_col,
                verbose=verbose_each,
            )
        except ValueError as e:
            print(f"  ⚠️  {e}")
            continue

        res = res.copy()
        res.insert(0, "session", session_name)
        all_results[session_name] = res
        res.to_csv(out_dir / f"features_{session_name}.csv", index=False)

        valid = res[res["VALID_EDGE"]]
        print(f"  ✅ VALID_EDGE: {len(valid)} / {len(res)}")
        if len(valid):
            ic_col = pd.to_numeric(valid["ic"], errors="coerce").abs()
            top = valid.assign(_abs_ic=ic_col).nlargest(5, "_abs_ic")
            for _, r in top.iterrows():
                print(
                    f"     {str(r['feature']):<30} IC={pd.to_numeric(r['ic'], errors='coerce'):+.4f} "
                    f"net_bps={pd.to_numeric(r['net_bps'], errors='coerce'):.2f}"
                )

    return all_results


def _classify_cross_session(
    results: dict[str, pd.DataFrame],
    *,
    universal_min_sessions: int,
) -> dict[str, Any]:
    all_feats: set[str] = set()
    for res in results.values():
        all_feats.update(res["feature"].astype(str).tolist())

    universal: list[dict[str, Any]] = []
    bridge: list[dict[str, Any]] = []
    specific: list[dict[str, Any]] = []
    multi: list[dict[str, Any]] = []
    weak: list[str] = []

    for feat in sorted(all_feats):
        valid_in: list[str] = []
        ic_per_session: dict[str, float] = {}

        for s_name, res in results.items():
            row = res[res["feature"].astype(str) == feat]
            if row.empty:
                continue
            ic_val = row["ic"].iloc[0]
            ic_f = float(ic_val) if pd.notna(ic_val) else 0.0
            ic_per_session[s_name] = ic_f
            if bool(row["VALID_EDGE"].iloc[0]):
                valid_in.append(s_name)

        n_valid = len(valid_in)
        in_overlap = any(s in _MULTI_OVERLAP_NAMES for s in valid_in)
        in_single = any(s in _MULTI_SINGLE_NAMES for s in valid_in)

        entry = {
            "feature": feat,
            "valid_in": valid_in,
            "n_sessions": n_valid,
            "avg_ic": round(float(np.mean(list(ic_per_session.values()))) if ic_per_session else 0.0, 6),
            "ic_per_session": ic_per_session,
        }

        if n_valid == 0:
            weak.append(feat)
        elif n_valid >= universal_min_sessions:
            universal.append(entry)
        elif in_overlap and not in_single:
            bridge.append(entry)
        elif n_valid == 1:
            specific.append(entry)
        else:
            multi.append(entry)

    universal.sort(key=lambda x: abs(x["avg_ic"]), reverse=True)
    bridge.sort(key=lambda x: abs(x["avg_ic"]), reverse=True)
    multi.sort(key=lambda x: abs(x["avg_ic"]), reverse=True)
    specific.sort(key=lambda x: abs(x["avg_ic"]), reverse=True)

    return {
        "universal": universal,
        "bridge": bridge,
        "multi_session": multi,
        "specific": specific,
        "weak": weak,
        "meta": {"universal_min_sessions": universal_min_sessions},
    }


def _build_multi_session_map(cross: dict[str, Any], results: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    session_map: dict[str, list[str]] = {}

    for s_name in results.keys():
        feats: list[str] = []

        for u in cross["universal"]:
            feats.append(u["feature"])

        if s_name in _MULTI_OVERLAP_NAMES:
            for b in cross["bridge"]:
                feats.append(b["feature"])

        for m in cross["multi_session"]:
            if s_name in m["valid_in"]:
                feats.append(m["feature"])

        for sp in cross["specific"]:
            if sp["valid_in"] and sp["valid_in"][0] == s_name:
                feats.append(sp["feature"])

        res = results[s_name]
        ic_map = dict(zip(res["feature"].astype(str), pd.to_numeric(res["ic"], errors="coerce")))
        feats = sorted(set(feats), key=lambda f: abs(float(ic_map.get(f, 0.0))), reverse=True)

        session_map[s_name] = feats

        print(f"\n{s_name}: {len(feats)} features (مقترحة)")
        for f in feats[:8]:
            ic = ic_map.get(f, np.nan)
            print(f"  {str(f):<34} IC={ic:+.4f}" if ic == ic else f"  {str(f):<34} IC=n/a")

    return session_map


def _save_multi_session_artifacts(
    cross: dict[str, Any],
    session_map: dict[str, list[str]],
    output_dir: str | Path,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "session_feature_map.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(session_map, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Session feature map → {json_path}")

    rows: list[dict[str, Any]] = []
    for session, feats in session_map.items():
        for rank, feat in enumerate(feats, 1):
            rows.append({"session": session, "rank": rank, "feature": feat})
    pd.DataFrame(rows).to_csv(out_dir / "session_feature_map.csv", index=False)
    print(f"💾 Session feature map CSV → {out_dir / 'session_feature_map.csv'}")

    slim = {
        "meta": cross.get("meta", {}),
        "weak": cross["weak"],
        "universal_features": [x["feature"] for x in cross["universal"]],
        "bridge_features": [x["feature"] for x in cross["bridge"]],
        "multi_session_features": [x["feature"] for x in cross["multi_session"]],
        "specific_features": [x["feature"] for x in cross["specific"]],
    }
    with open(out_dir / "cross_session_summary.json", "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    print(f"💾 Cross-session summary → {out_dir / 'cross_session_summary.json'}")

    detail = {k: v for k, v in cross.items() if k != "meta"}
    with open(out_dir / "cross_session_detail.json", "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2, default=str)


# ─── Nonlinear probe (خمسينات + KNN/RF) — مدمج لتجنب ملف منفصل على السيرفر ──


def _nl_time_split(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.sort_values("ts_event").reset_index(drop=True)
    cut = int(len(d) * float(train_ratio))
    if cut < 50 or len(d) - cut < 25:
        raise ValueError(f"تقسيم زمني غير كافٍ: n={len(d)}, cut={cut}")
    return d.iloc[:cut].copy(), d.iloc[cut:].copy()


def _nl_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 15:
        return float("nan")
    r, _ = stats.spearmanr(a[m], b[m])
    return float(r) if r == r else float("nan")


def _nl_quintile_probe_row(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
    target: str,
) -> dict[str, Any]:
    tr = train[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
    te = test[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(tr) < 80 or len(te) < 30:
        return {"feature": feature, "error": "صفوف غير كافية"}

    qs = tr[feature].quantile([0.2, 0.4, 0.6, 0.8]).values.astype(float)

    def bucket(x: float) -> int:
        if not np.isfinite(x):
            return -1
        for i, q in enumerate(qs):
            if x <= q:
                return i + 1
        return 5

    te = te.copy()
    te["_b"] = te[feature].map(bucket)
    te = te[te["_b"] >= 1]
    mu = te.groupby("_b", observed=True)[target].mean()
    spread = float(mu.iloc[-1] - mu.iloc[0]) if len(mu) >= 2 else float("nan")
    mids = np.array(mu.index.astype(float))
    vals = mu.values.astype(float)
    shape_ic = _nl_spearman(mids, vals)

    return {
        "feature": feature,
        "n_test_bucketed": int(len(te)),
        "mean_ret_q1": round(float(mu.iloc[0]), 6) if 1 in mu.index else np.nan,
        "mean_ret_q5": round(float(mu.iloc[-1]), 6) if len(mu) else np.nan,
        "spread_q5_minus_q1": round(spread, 8),
        "spearman_bucket_mid_vs_mu": round(shape_ic, 5),
        "per_bucket_mean": {int(k): round(float(v), 6) for k, v in mu.items()},
    }


def _nl_knn_rf_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    knn_k: int,
    rf_trees: int,
    rf_depth: int | None,
    seed: int,
) -> dict[str, Any]:
    if not _HAS_SKLEARN:
        return {"error": "ثبّت scikit-learn: pip install scikit-learn"}

    cols = features + [target]
    tr = train[cols].replace([np.inf, -np.inf], np.nan).dropna()
    te = test[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(tr) < 100 or len(te) < 40:
        return {"error": "صفوف غير كافية للـ KNN/RF"}

    X_tr = tr[features].to_numpy(dtype=float)
    y_tr = tr[target].to_numpy(dtype=float)
    X_te = te[features].to_numpy(dtype=float)
    y_te = te[target].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    knn = KNeighborsRegressor(n_neighbors=min(knn_k, len(X_tr) // 3))
    knn.fit(X_trs, y_tr)
    pred_k = knn.predict(X_tes)

    rf = RandomForestRegressor(
        n_estimators=rf_trees,
        max_depth=rf_depth,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_trs, y_tr)
    pred_r = rf.predict(X_tes)

    ic_knn = _nl_spearman(pred_k, y_te)
    ic_rf = _nl_spearman(pred_r, y_te)

    best_feat = ""
    best_ic = -np.inf
    for f in features:
        ic1 = _nl_spearman(te[f].to_numpy(), y_te)
        if ic1 == ic1 and abs(ic1) > abs(best_ic):
            best_ic = ic1
            best_feat = f

    return {
        "n_train": len(tr),
        "n_test": len(te),
        "spearman_knn_pred_vs_y": round(ic_knn, 5),
        "spearman_rf_pred_vs_y": round(ic_rf, 5),
        "best_single_feature": best_feat,
        "best_single_spearman_vs_y": round(float(best_ic), 5) if best_feat else np.nan,
    }


def _parse_nl_flow_features(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


_NL_SESSION_INTRADAY_MODES: frozenset[str] = frozenset(
    {"asia_near_high", "asia_near_low", "london_near_high", "london_near_low"}
)


def _nl_resolve_price_col(df: pd.DataFrame, preferred: str | None) -> str | None:
    if preferred and str(preferred).strip() and preferred in df.columns:
        return preferred
    for c in ("micro_price", "close", "last_price", "mid_price", "price", "open"):
        if c in df.columns:
            return c
    return None


def _nl_intraday_dist_to_running_extremes(df: pd.DataFrame, price_col: str) -> tuple[pd.Series, pd.Series]:
    """ضمن الصفوف الحالية (نافذة جلسة واحدة مسبقًا): لكل يوم تقويمي، مسافة السعر إلى cummax/cummin حتى اللحظة."""
    ts = _resolve_ts_series(df)
    p = df[price_col].replace([np.inf, -np.inf], np.nan).astype(float)
    day = ts.dt.normalize()
    dh = pd.Series(np.nan, index=df.index, dtype=float)
    dl = pd.Series(np.nan, index=df.index, dtype=float)
    tmp = pd.DataFrame({"p": p.values, "day": day.values, "_ts": ts.values}, index=df.index)
    for _, grp in tmp.groupby("day", sort=False):
        grp = grp.sort_values("_ts")
        ix = grp.index
        pv = grp["p"].to_numpy(dtype=float)
        cmax = pd.Series(pv).cummax().to_numpy()
        cmin = pd.Series(pv).cummin().to_numpy()
        dh.loc[ix] = cmax - pv
        dl.loc[ix] = pv - cmin
    return dh, dl


def _nl_zone_series_for_mode(
    df: pd.DataFrame,
    mode_n: str,
    *,
    zone_feat: str,
    session_name: str | None,
    price_col_preferred: str | None,
) -> tuple[pd.Series | None, str]:
    """إرجاع سلسلة المسافة z للفلتر أو None إذا يُستعاض عمود جاهز/intraday."""
    if mode_n in _NL_SESSION_INTRADAY_MODES:
        want = "asia" if mode_n.startswith("asia_") else "london"
        sn = (session_name or "").lower().strip()
        if sn != want:
            print(
                f"⚠️  nl-zone-mode={mode_n!r} مخصّص لنافذة «{want}» فقط — الجلسة الحالية {session_name!r}. "
                "تخطي فلتر المنطقة.",
                file=sys.stderr,
            )
            return None, mode_n
        pc = _nl_resolve_price_col(df, price_col_preferred)
        if not pc:
            print(
                "⚠️  لا عمود سعر (جرّب micro_price أو close أو --nl-price-col) — تخطي فلتر قمة/قاع الجلسة.",
                file=sys.stderr,
            )
            return None, mode_n
        dh, dl = _nl_intraday_dist_to_running_extremes(df, pc)
        z = dh if mode_n.endswith("_near_high") else dl
        label = f"intraday_{mode_n}[price={pc}]"
        return z, label

    if zone_feat not in df.columns:
        print(f"⚠️  nl-zone-feature {zone_feat!r} غير موجود — بدون فلتر المنطقة.", file=sys.stderr)
        return None, zone_feat
    z = df[zone_feat].replace([np.inf, -np.inf], np.nan)
    return z, zone_feat


def _nl_apply_zone_filter(
    df: pd.DataFrame,
    zone_feat: str,
    mode: str,
    frac: float,
    *,
    session_name: str | None = None,
    price_col_preferred: str | None = None,
) -> pd.DataFrame:
    """طبقة 1: ذيل كمّي على مسافة (عمود أو قمة/قاع جلسة آسيا/لندن جارية)."""
    f = float(frac)
    if not 0 < f <= 0.5:
        f = min(max(f, 1e-6), 0.5)
        print(f"⚠️  nl-zone-fraction يُقصّ إلى (0, 0.5] — استخدمنا {f}", file=sys.stderr)
    mode_n = mode.lower().strip().replace("-", "_")

    z_raw, z_label = _nl_zone_series_for_mode(
        df,
        mode_n,
        zone_feat=zone_feat,
        session_name=session_name,
        price_col_preferred=price_col_preferred,
    )
    if z_raw is None:
        return df

    z = z_raw.replace([np.inf, -np.inf], np.nan)
    valid = z.notna()
    if int(valid.sum()) < 50:
        print(f"⚠️  صفوف صالحة قليلة لـ {z_label} — بدون فلتر المنطقة.", file=sys.stderr)
        return df
    qcol = z.loc[valid].astype(float)

    if mode_n in _NL_SESSION_INTRADAY_MODES:
        thr = float(qcol.quantile(f))
        mask = z <= thr
    elif mode_n in ("near_pdh", "near"):
        thr = float(qcol.quantile(f))
        mask = z <= thr
    elif mode_n in ("far_pdh", "far"):
        thr = float(qcol.quantile(1.0 - f))
        mask = z >= thr
    else:
        print(
            f"⚠️  nl-zone-mode غير معروف {mode!r} — استخدم near_pdh | far_pdh | "
            "asia_near_high | asia_near_low | london_near_high | london_near_low",
            file=sys.stderr,
        )
        return df

    out = df.loc[mask.fillna(False)].copy()
    print(
        f"  📍 Two-layer zone | {z_label!r} mode={mode_n!r} frac={f:g} → rows {len(df):,} → {len(out):,} "
        f"(thr={thr:.6g})"
    )
    if len(out) < 30:
        print(f"⚠️  بعد فلتر المنطقة بقيت {len(out)} صف — قد يفشل أو يضعف الـ probe.", file=sys.stderr)
    return out


def _run_nonlinear_probe_cli(args: Any, df_session: pd.DataFrame, feats: list[str]) -> None:
    methods = {m.strip().lower() for m in str(args.nonlinear_methods).split(",") if m.strip()}
    fc = args.forward_col
    df_s = df_session.replace([np.inf, -np.inf], np.nan)
    if getattr(args, "nonlinear_two_layer", False):
        df_s = _nl_apply_zone_filter(
            df_s,
            getattr(args, "nl_zone_feature", "dist_to_pdh"),
            getattr(args, "nl_zone_mode", "near_pdh"),
            float(getattr(args, "nl_zone_fraction", 0.25)),
            session_name=getattr(args, "session", None),
            price_col_preferred=getattr(args, "nl_price_col", None),
        )
    train_df, test_df = _nl_time_split(df_s, args.train_ratio)

    stem = Path(args.parquet).stem
    prefix = (
        args.nonlinear_out_prefix
        if args.nonlinear_out_prefix
        else str(Path(args.parquet).with_name(f"{stem}_nonlinear_{args.session}"))
    )

    print(
        f"🔬 Nonlinear probe | rows={len(df_s):,} train={len(train_df):,} test={len(test_df):,} | "
        f"session={args.session!r} | features={len(feats)}"
    )

    if "quintile" in methods:
        rows = [_nl_quintile_probe_row(train_df, test_df, f, fc) for f in feats]
        qpath = f"{prefix}_quintile_probe.csv"
        pd.DataFrame(rows).to_csv(qpath, index=False)
        print(f"💾 Quintile probe → {qpath}")

    if "knn" in methods or "rf" in methods:
        if not _HAS_SKLEARN:
            print("⚠️  تخطي KNN/RF — pip install scikit-learn", file=sys.stderr)
        else:
            depth = int(args.nonlinear_rf_max_depth)
            summ = _nl_knn_rf_summary(
                train_df,
                test_df,
                feats,
                fc,
                knn_k=int(args.nonlinear_knn_k),
                rf_trees=int(args.nonlinear_rf_trees),
                rf_depth=depth if depth > 0 else None,
                seed=int(args.nonlinear_seed),
            )
            mpath = f"{prefix}_knn_rf_summary.csv"
            pd.DataFrame([summ]).to_csv(mpath, index=False)
            print(f"💾 KNN/RF summary → {mpath}")
            print(summ)

    print("✅ انتهى --nonlinear-probe")


def _nl_zone_modes_for_batch_session(args: Any, session_name: str) -> list[str]:
    """عند --nl-sweep-asia-london-extremes نمرّ قرب القمة ثم قرب القاع لآسيا ولندن فقط."""
    if getattr(args, "nl_sweep_asia_london_extremes", False) and getattr(
        args, "nonlinear_two_layer", False
    ):
        if session_name == "asia":
            return ["asia_near_high", "asia_near_low"]
        if session_name == "london":
            return ["london_near_high", "london_near_low"]
    return [str(getattr(args, "nl_zone_mode", "near_pdh"))]


def _run_nonlinear_probe_all_windows(args: Any, df: pd.DataFrame, feats_all: list[str]) -> None:
    """نفس نوافذ MULTI_SESSION_WINDOWS المستخدمة في --all-session-windows."""
    df = df.replace([np.inf, -np.inf], np.nan)
    out_root = Path(args.nonlinear_batch_out)
    out_root.mkdir(parents=True, exist_ok=True)
    stem = Path(args.parquet).stem

    for session_name, (start, end) in MULTI_SESSION_WINDOWS.items():
        print(f"\n{'═' * 60}\n🔬 Nonlinear — {session_name.upper()}  [{start} → {end}]")
        df_w = _filter_df_time_window(df, start, end)
        min_r = MULTI_SESSION_MIN_ROWS.get(session_name, 50)
        if len(df_w) < min_r:
            print(f"  ⚠️  تخطي: صفوف {len(df_w)} < {min_r}")
            continue
        if args.forward_col not in df_w.columns:
            print("  ⚠️  تخطي: عمود الهدف غير موجود")
            continue
        feats_ok = [f for f in feats_all if f in df_w.columns]
        if not feats_ok:
            print("  ⚠️  تخطي: لا فيتشر مطابقة")
            continue

        zone_modes = _nl_zone_modes_for_batch_session(args, session_name)
        for zm in zone_modes:
            zm_tag = f"_{zm}" if len(zone_modes) > 1 else ""
            prefix = str(out_root / f"{stem}_nonlinear_{session_name}{zm_tag}")
            ns = argparse.Namespace(
                parquet=args.parquet,
                session=session_name,
                forward_col=args.forward_col,
                train_ratio=args.train_ratio,
                nonlinear_methods=args.nonlinear_methods,
                nonlinear_out_prefix=prefix,
                nonlinear_knn_k=args.nonlinear_knn_k,
                nonlinear_rf_trees=args.nonlinear_rf_trees,
                nonlinear_rf_max_depth=args.nonlinear_rf_max_depth,
                nonlinear_seed=args.nonlinear_seed,
                nonlinear_two_layer=getattr(args, "nonlinear_two_layer", False),
                nl_zone_feature=getattr(args, "nl_zone_feature", "dist_to_pdh"),
                nl_zone_mode=zm,
                nl_zone_fraction=getattr(args, "nl_zone_fraction", 0.25),
                nl_price_col=getattr(args, "nl_price_col", None),
            )
            _run_nonlinear_probe_cli(ns, df_w, feats_ok)

    print(f"\n✅ انتهى --nonlinear-all-session-windows → {out_root}")


# ─── Rigorous helpers ───────────────────────────────────────────────────────


def _clean_pair(df: pd.DataFrame, feat: str, target: str) -> pd.DataFrame:
    sub = df[[feat, target]].copy()
    return sub.replace([np.inf, -np.inf], np.nan).dropna()


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 4:
        return np.nan, np.nan
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan, np.nan
    ic, pval = stats.spearmanr(x, y)
    return float(ic), float(pval)


def _expanding_rank_quantile(
    series: pd.Series,
    n_quantiles: int = 5,
    *,
    min_hist: int | None = None,
) -> pd.Series:
    """كوانتايل تقريبي سببي: في t يُستخدم التاريخ [0..t] فقط."""
    out = pd.Series(index=series.index, dtype=np.float64)
    arr = series.to_numpy(dtype=float)
    mh = int(min_hist) if min_hist is not None else max(int(n_quantiles) * 2, 8)
    for i in range(len(arr)):
        window = arr[: i + 1]
        if np.sum(~np.isnan(window)) < mh:
            out.iloc[i] = np.nan
            continue
        val = arr[i]
        if np.isnan(val):
            out.iloc[i] = np.nan
            continue
        pct = float(np.nanmean(window <= val))
        q = int(min(pct * n_quantiles, n_quantiles - 1))
        out.iloc[i] = q
    return out


class AlphaValidatorRigorous:
    """
    IC + TAR على OOS فقط؛ استقرار على Train فقط؛ TAR بكوانتايل متوسع؛ FDR اختياري.
    """

    def __init__(
        self,
        *,
        train_ratio: float = 0.70,
        min_ic: float = 0.05,
        min_icir: float = 0.35,
        ic_ttest_alpha: float = 0.10,
        n_splits: int = 4,
        n_quantiles: int = 5,
        cost_per_trade: float = 2e-4,
        ic_period: str = "W",
        stability_pos_ratio: float = 0.75,
        min_train_rows: int = 80,
        min_test_rows: int = 40,
        min_ic_period_rows: int = 15,
        tar_min_rows_pre: int | None = None,
        tar_min_rows_post: int | None = None,
        stab_split_floor: int | None = None,
        stab_sub_min_rows: int | None = None,
        exp_quantile_min_hist: int | None = None,
    ) -> None:
        if not 0.35 <= train_ratio <= 0.90:
            raise ValueError("train_ratio يجب أن يكون بين 0.35 و 0.90")
        self.train_ratio = train_ratio
        self.min_ic = min_ic
        self.min_icir = min_icir
        self.ic_ttest_alpha = ic_ttest_alpha
        self.n_splits = max(2, int(n_splits))
        self.n_quantiles = max(3, int(n_quantiles))
        self.cost_per_trade = float(cost_per_trade)
        self.ic_period = ic_period
        self.stability_pos_ratio = stability_pos_ratio
        self.min_train_rows = min_train_rows
        self.min_test_rows = min_test_rows
        self.min_ic_period_rows = min_ic_period_rows
        self.tar_min_rows_pre = tar_min_rows_pre
        self.tar_min_rows_post = tar_min_rows_post
        self.stab_split_floor = stab_split_floor
        self.stab_sub_min_rows = stab_sub_min_rows
        self.exp_quantile_min_hist = exp_quantile_min_hist

    def _split_train_test(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        d = df.sort_values("ts_event").reset_index(drop=True)
        cut = int(len(d) * self.train_ratio)
        if cut < self.min_train_rows or (len(d) - cut) < self.min_test_rows:
            raise ValueError(
                f"بيانات غير كافية بعد الفلتر: n={len(d)}, cut={cut}. "
                f"يلزم train≥{self.min_train_rows} و test≥{self.min_test_rows} — وسّع الفترة أو خفّض النسبة."
            )
        return d.iloc[:cut].copy(), d.iloc[cut:].copy()

    def test_ic(self, test_df: pd.DataFrame, feature: str, forward_col: str) -> dict[str, Any]:
        df = test_df.copy()
        df["_period"] = pd.to_datetime(df["ts_event"]).dt.to_period(self.ic_period)
        period_ics: list[float] = []
        period_pvals: list[float] = []
        for _, grp in df.groupby("_period", sort=True):
            sub = _clean_pair(grp, feature, forward_col)
            if len(sub) < self.min_ic_period_rows:
                continue
            ic, pval = _spearman_safe(
                sub[feature].to_numpy(dtype=np.float64),
                sub[forward_col].to_numpy(dtype=np.float64),
            )
            if ic == ic and not np.isnan(ic):
                period_ics.append(ic)
                period_pvals.append(pval)

        if len(period_ics) < 2:
            return {
                "ic": np.nan,
                "ic_std": np.nan,
                "icir": np.nan,
                "n_periods": len(period_ics),
                "ttest_pval": np.nan,
                "passed": False,
                "stable": False,
                "reason": "فترات IC غير كافية",
            }

        ic_mean = float(np.mean(period_ics))
        ic_std = float(np.std(period_ics, ddof=1)) + 1e-9
        icir = ic_mean / ic_std
        _, ttest_pval = stats.ttest_1samp(period_ics, 0.0)

        passed = (
            abs(ic_mean) > self.min_ic
            and abs(icir) > self.min_icir
            and float(ttest_pval) < self.ic_ttest_alpha
        )
        return {
            "ic": round(ic_mean, 5),
            "ic_std": round(ic_std, 5),
            "icir": round(icir, 4),
            "n_periods": len(period_ics),
            "ttest_pval": round(float(ttest_pval), 6),
            "passed": passed,
            "stable": abs(icir) > 0.50,
        }

    def test_tar(self, test_df: pd.DataFrame, feature: str, forward_col: str) -> dict[str, Any]:
        df = test_df.sort_values("ts_event").reset_index(drop=True)
        m = df[[feature, forward_col]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        df = df.loc[m].reset_index(drop=True)
        pre_need = self.tar_min_rows_pre if self.tar_min_rows_pre is not None else self.n_quantiles * 15
        if len(df) < pre_need:
            return {
                "ls_spread_bps": np.nan,
                "cost_bps": np.nan,
                "net_bps": np.nan,
                "turnover": np.nan,
                "passed": False,
                "reason": "صفوف غير كافية بعد dropna",
            }

        min_hist = self.exp_quantile_min_hist
        df["_q"] = _expanding_rank_quantile(
            df[feature].astype(np.float64),
            self.n_quantiles,
            min_hist=min_hist,
        )
        df = df.dropna(subset=["_q"])
        post_need = self.tar_min_rows_post if self.tar_min_rows_post is not None else self.n_quantiles * 10
        if len(df) < post_need:
            return {
                "ls_spread_bps": np.nan,
                "cost_bps": np.nan,
                "net_bps": np.nan,
                "turnover": np.nan,
                "passed": False,
                "reason": "كوانتايل متوسع غير كافٍ",
            }

        q_returns = df.groupby("_q", observed=True)[forward_col].mean()
        if len(q_returns) < 2:
            return {
                "ls_spread_bps": np.nan,
                "cost_bps": np.nan,
                "net_bps": np.nan,
                "turnover": np.nan,
                "passed": False,
                "reason": "كوانتايلات ناقصة",
            }

        long_ret = float(q_returns.iloc[-1])
        short_ret = float(q_returns.iloc[0])
        ls_spread = long_ret - short_ret

        df["_signal"] = (df["_q"] >= self.n_quantiles - 1).astype(np.int8)
        turnover = float(df["_signal"].diff().fillna(0).abs().mean())
        cost = turnover * self.cost_per_trade * 2.0
        net = ls_spread - cost

        return {
            "ls_spread_bps": round(ls_spread * 10_000, 3),
            "cost_bps": round(cost * 10_000, 3),
            "net_bps": round(net * 10_000, 3),
            "turnover": round(turnover, 5),
            "passed": bool(net > 0 and ls_spread > 0),
        }

    def test_stability(self, train_df: pd.DataFrame, feature: str, forward_col: str) -> dict[str, Any]:
        df = train_df.sort_values("ts_event").reset_index(drop=True)
        split_floor = self.stab_split_floor if self.stab_split_floor is not None else 25
        sub_need = self.stab_sub_min_rows if self.stab_sub_min_rows is not None else 15
        split_size = max(len(df) // self.n_splits, split_floor)
        split_ics: list[float] = []
        for i in range(self.n_splits):
            chunk = df.iloc[i * split_size : (i + 1) * split_size]
            sub = _clean_pair(chunk, feature, forward_col)
            if len(sub) < sub_need:
                continue
            ic, _ = _spearman_safe(
                sub[feature].to_numpy(dtype=np.float64),
                sub[forward_col].to_numpy(dtype=np.float64),
            )
            split_ics.append(float(ic) if ic == ic and not np.isnan(ic) else 0.0)

        if len(split_ics) < 2:
            return {
                "split_ics": [],
                "positive_ratio": 0.0,
                "ic_consistency": 0.0,
                "trend_ok": False,
                "passed": False,
                "reason": "أجزاء غير كافية",
            }

        positive_ratio = sum(1 for ic in split_ics if ic > 0) / len(split_ics)
        ic_consistency = float(np.mean(np.abs(split_ics)))

        trend_ok = True
        if len(split_ics) >= 3:
            fh = float(np.mean(split_ics[: len(split_ics) // 2]))
            sh = float(np.mean(split_ics[len(split_ics) // 2 :]))
            if fh != 0 and (sh / (fh + 1e-9)) < -0.5:
                trend_ok = False

        passed = positive_ratio >= self.stability_pos_ratio and trend_ok
        return {
            "split_ics": [round(x, 5) for x in split_ics],
            "positive_ratio": round(positive_ratio, 4),
            "ic_consistency": round(ic_consistency, 5),
            "trend_ok": trend_ok,
            "passed": passed,
        }

    @staticmethod
    def _apply_fdr(results_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        out = results_df.copy()
        if not _HAS_STATSMODELS:
            warnings.warn(
                "statsmodels غير مثبت — تخطي FDR. pip install statsmodels",
                stacklevel=2,
            )
            out["fdr_adjusted_pval"] = np.nan
            out["passed_fdr"] = True
            return out

        pvals = pd.to_numeric(out["ic_ttest_pval"], errors="coerce").fillna(1.0).to_numpy()
        reject, p_corr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        out["fdr_adjusted_pval"] = np.round(p_corr, 6)
        out["passed_fdr"] = reject
        return out

    def run_full_validation(
        self,
        df: pd.DataFrame,
        features: list[str],
        forward_ret_col: str,
        *,
        fdr_alpha: float = 0.10,
        verbose: bool = True,
    ) -> pd.DataFrame:
        if "ts_event" not in df.columns:
            raise ValueError("يلزم عمود ts_event للمسار الصارم.")
        if forward_ret_col not in df.columns:
            raise ValueError(f"عمود الهدف غير موجود: {forward_ret_col}")

        feats_ok = [f for f in features if f in df.columns]
        train_df, test_df = self._split_train_test(df)

        if verbose:
            print(
                f"\n📊 [rigorous] Train={len(train_df):,} | Test={len(test_df):,} | "
                f"features={len(feats_ok)} | IC-period={self.ic_period!r}"
            )

        rows: list[dict[str, Any]] = []
        for feat in feats_ok:
            ic_r = self.test_ic(test_df, feat, forward_ret_col)
            tar_r = self.test_tar(test_df, feat, forward_ret_col)
            stb_r = self.test_stability(train_df, feat, forward_ret_col)

            raw_ok = ic_r.get("passed") and tar_r.get("passed") and stb_r.get("passed")
            rows.append(
                {
                    "feature": feat,
                    "ic": ic_r.get("ic"),
                    "ic_std": ic_r.get("ic_std"),
                    "icir": ic_r.get("icir"),
                    "ic_n_periods": ic_r.get("n_periods"),
                    "ic_ttest_pval": ic_r.get("ttest_pval"),
                    "ic_stable": ic_r.get("stable"),
                    "ic_passed": ic_r.get("passed"),
                    "net_bps": tar_r.get("net_bps"),
                    "ls_spread_bps": tar_r.get("ls_spread_bps"),
                    "cost_bps": tar_r.get("cost_bps"),
                    "turnover": tar_r.get("turnover"),
                    "tar_passed": tar_r.get("passed"),
                    "split_ics": str(stb_r.get("split_ics", [])),
                    "positive_ratio": stb_r.get("positive_ratio"),
                    "ic_consistency": stb_r.get("ic_consistency"),
                    "trend_ok": stb_r.get("trend_ok"),
                    "stability_passed": stb_r.get("passed"),
                    "VALID_EDGE_raw": raw_ok,
                }
            )

        res = pd.DataFrame(rows)
        res = self._apply_fdr(res, alpha=fdr_alpha)
        res["VALID_EDGE"] = res["VALID_EDGE_raw"].astype(bool) & res["passed_fdr"].astype(bool)
        res["_abs_ic"] = pd.to_numeric(res["ic"], errors="coerce").abs()
        res = res.sort_values("_abs_ic", ascending=False).drop(columns=["_abs_ic"]).reset_index(drop=True)

        if verbose:
            ok = res[res["VALID_EDGE"]]["feature"].tolist()
            ok_raw = res[res["VALID_EDGE_raw"]]["feature"].tolist()
            print("═" * 58)
            print(f"نجح (3 اختبارات قبل FDR): {len(ok_raw)} | بعد FDR: {len(ok)}")
            if ok:
                print("مرشحون:")
                for _, row in res[res["VALID_EDGE"]].iterrows():
                    print(
                        f"  {row['feature']:<34} IC={row['ic']:+.4f} ICIR={row['icir']:+.3f} "
                        f"net_bps={row['net_bps']}"
                    )
            else:
                print("⚠️ لا مرشح بعد المعايير الصارمة.")
            print("═" * 58)

        return res


# ─── Legacy validator (سريع) ─────────────────────────────────────────────────


class AlphaValidator:
    def __init__(
        self,
        *,
        min_ic: float,
        min_icir: float,
        ic_stable_icir: float,
        min_rows_period: int,
        ic_period: str,
        stability_positive_ratio: float,
        cost_per_turnover: float,
        quantiles: int,
        long_q: int,
        short_q: int,
    ):
        self.min_ic = min_ic
        self.min_icir = min_icir
        self.ic_stable_icir = ic_stable_icir
        self.min_rows_period = min_rows_period
        self.ic_period = ic_period
        self.stability_positive_ratio = stability_positive_ratio
        self.cost_per_turnover = cost_per_turnover
        self.quantiles = quantiles
        self.long_q = long_q
        self.short_q = short_q

    def test_ic(
        self,
        df: pd.DataFrame,
        feature: str,
        forward_col: str,
    ) -> dict[str, Any]:
        need = ["_alpha_ts", feature, forward_col]
        sub = df[need].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < self.min_rows_period * 3:
            return {"ic": 0.0, "icir": 0.0, "passed": False, "stable": False, "n_periods": 0}

        sub = sub.sort_values("_alpha_ts")
        sub["_per"] = sub["_alpha_ts"].dt.to_period(self.ic_period)

        monthly_ic: list[float] = []
        for _, grp in sub.groupby("_per", sort=True):
            if len(grp) < self.min_rows_period:
                continue
            ic, _ = stats.spearmanr(
                grp[feature].to_numpy(dtype=np.float64),
                grp[forward_col].to_numpy(dtype=np.float64),
            )
            if ic == ic and not np.isnan(ic):
                monthly_ic.append(float(ic))

        if len(monthly_ic) < 2:
            ic_mean = float(np.mean(monthly_ic)) if monthly_ic else 0.0
            ic_std = 1e-9
            icir = 0.0
        else:
            ic_mean = float(np.mean(monthly_ic))
            ic_std = float(np.std(monthly_ic, ddof=1)) + 1e-9
            icir = ic_mean / ic_std

        passed = abs(ic_mean) > self.min_ic and abs(icir) > self.min_icir
        stable = abs(icir) > self.ic_stable_icir
        return {
            "ic": round(ic_mean, 5),
            "icir": round(icir, 5),
            "passed": passed,
            "stable": stable,
            "n_periods": len(monthly_ic),
        }

    def test_tar(
        self,
        df: pd.DataFrame,
        feature: str,
        forward_col: str,
    ) -> dict[str, Any]:
        sub = df[[feature, forward_col, "_alpha_ts"]].replace([np.inf, -np.inf], np.nan).dropna()
        sub = sub.sort_values("_alpha_ts")
        if len(sub) < self.quantiles * 30:
            return {"ls_spread": 0.0, "cost_adj": 0.0, "net_bps": 0.0, "turnover": 0.0, "passed": False}

        try:
            sub["_q"] = pd.qcut(sub[feature], self.quantiles, labels=False, duplicates="drop")
        except ValueError:
            return {"ls_spread": 0.0, "cost_adj": 0.0, "net_bps": 0.0, "turnover": 0.0, "passed": False}

        sub = sub.dropna(subset=["_q"])
        if len(sub) < self.quantiles * 30:
            return {"ls_spread": 0.0, "cost_adj": 0.0, "net_bps": 0.0, "turnover": 0.0, "passed": False}

        qmax = int(sub["_q"].max())
        if qmax < self.long_q or 0 not in sub["_q"].unique():
            return {"ls_spread": 0.0, "cost_adj": 0.0, "net_bps": 0.0, "turnover": 0.0, "passed": False}

        mu = sub.groupby("_q", observed=True)[forward_col].mean()
        long_ret = float(mu.loc[self.long_q]) if self.long_q in mu.index else float(mu.iloc[-1])
        short_ret = float(mu.loc[self.short_q]) if self.short_q in mu.index else float(mu.iloc[0])
        ls_spread = long_ret - short_ret

        sig = (sub["_q"] == self.long_q).astype(np.int8)
        turnover = float(sig.diff().fillna(0).abs().mean())
        cost = turnover * self.cost_per_turnover * 2.0
        net = ls_spread - cost

        return {
            "ls_spread": round(ls_spread * 10000, 3),
            "cost_adj": round(cost * 10000, 3),
            "net_bps": round(net * 10000, 3),
            "turnover": round(turnover, 5),
            "passed": net > 0,
        }

    def test_stability(
        self,
        df: pd.DataFrame,
        feature: str,
        forward_col: str,
        n_splits: int,
    ) -> dict[str, Any]:
        sub = df[["_alpha_ts", feature, forward_col]].replace([np.inf, -np.inf], np.nan).dropna()
        sub = sub.sort_values("_alpha_ts")[[feature, forward_col]]
        if len(sub) < n_splits * 15:
            return {"positive_ratio": 0.0, "mean_abs_ic": 0.0, "split_ics": [], "passed": False}

        splits = np.array_split(sub, n_splits)
        split_ics: list[float] = []
        for sp in splits:
            if len(sp) < 10:
                continue
            ic, _ = stats.spearmanr(
                sp[feature].to_numpy(dtype=np.float64),
                sp[forward_col].to_numpy(dtype=np.float64),
            )
            split_ics.append(float(ic) if ic == ic and not np.isnan(ic) else 0.0)

        if not split_ics:
            return {"positive_ratio": 0.0, "mean_abs_ic": 0.0, "split_ics": [], "passed": False}

        positive_ratio = sum(1 for x in split_ics if x > 0) / len(split_ics)
        mean_abs_ic = float(np.mean(np.abs(split_ics)))
        passed = positive_ratio >= self.stability_positive_ratio
        return {
            "positive_ratio": round(positive_ratio, 4),
            "mean_abs_ic": round(mean_abs_ic, 5),
            "split_ics": [round(x, 4) for x in split_ics],
            "passed": passed,
        }

    def run(
        self,
        df: pd.DataFrame,
        features: list[str],
        forward_col: str,
        n_splits: int,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for feat in features:
            if feat not in df.columns:
                continue
            ic_res = self.test_ic(df, feat, forward_col)
            tar_res = self.test_tar(df, feat, forward_col)
            stb_res = self.test_stability(df, feat, forward_col, n_splits)
            passed_all = ic_res["passed"] and tar_res["passed"] and stb_res["passed"]
            rows.append(
                {
                    "feature": feat,
                    "ic_mean_period": ic_res["ic"],
                    "icir": ic_res["icir"],
                    "ic_periods_n": ic_res["n_periods"],
                    "ic_pass": ic_res["passed"],
                    "ls_spread_bps": tar_res["ls_spread"],
                    "cost_bps": tar_res["cost_adj"],
                    "net_ls_bps": tar_res["net_bps"],
                    "turnover": tar_res["turnover"],
                    "tar_pass": tar_res["passed"],
                    "stab_pos_ratio": stb_res["positive_ratio"],
                    "stab_mean_abs_ic": stb_res["mean_abs_ic"],
                    "stab_pass": stb_res["passed"],
                    "VALID_EDGE_strict": passed_all,
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values("ic_mean_period", key=lambda s: s.abs(), ascending=False)


def main() -> None:
    try:
        for _stream in (sys.stdout, sys.stderr):
            if hasattr(_stream, "reconfigure"):
                _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Alpha / IC diagnostics (session filter + optional rigorous OOS path)")
    ap.add_argument(
        "--parquet",
        required=True,
        help=(
            "مسار ملف جدول واحد (parquet/csv)، أو عدة مسارات مفصولة بـ ';' لدمج شهرين/أكثر، "
            "أو مجلد يحوي *.parquet، أو مجلد final/، أو جذر مخرجات prepare_training_data "
            "(features_*.parquet)، أو artifact_manifest.json"
        ),
    )
    ap.add_argument("--output-csv", default=None)
    ap.add_argument(
        "--session",
        default="asia",
        help=(
            "asia | london | overlap | london_ny (13:30–16 UTC) | ny | "
            "all — كامل ملف الداتا بدون فلتر وقت UTC"
        ),
    )
    ap.add_argument("--session-profile", default="daytrade_default")
    ap.add_argument(
        "--forward-col",
        default="forward_return",
        help="يُفضّل fwd_ret_clean للمسار الصارم؛ forward_return للتوافق مع ملفات قديمة",
    )
    ap.add_argument("--features", default=None)
    ap.add_argument("--rigorous", action="store_true", help="مسار OOS + expanding quantile + FDR + stability على Train فقط")
    ap.add_argument("--legacy", action="store_true", help="المسار السريع القديم (يعطل الصارم إن مرّر معه)")
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--fdr-alpha", type=float, default=0.10)
    ap.add_argument("--ic-ttest-alpha", type=float, default=0.10)
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--min-ic", type=float, default=0.05)
    ap.add_argument("--min-icir", type=float, default=0.05)
    ap.add_argument("--min-icir-rigorous", type=float, default=0.35, help="عتبة ICIR للمسار الصارم فقط")
    ap.add_argument("--ic-period", default="W")
    ap.add_argument("--stab-pos-ratio", type=float, default=0.75)
    ap.add_argument("--cost-per-turnover", type=float, default=2e-4)
    ap.add_argument("--quantiles", type=int, default=5)
    ap.add_argument("--long-q", type=int, default=4)
    ap.add_argument("--short-q", type=int, default=0)
    ap.add_argument("--min-rows-period", type=int, default=20)
    ap.add_argument("--ic-stable-icir", type=float, default=0.5)
    ap.add_argument(
        "--pattern-analysis",
        action="store_true",
        help="تصدير Rolling IC / IC Decay / Regime IC (ذيل تقييم افتراضي 30%%)",
    )
    ap.add_argument("--pattern-rolling-window", type=int, default=500)
    ap.add_argument("--pattern-max-horizon", type=int, default=48)
    ap.add_argument("--pattern-eval-tail", type=float, default=0.30)
    ap.add_argument("--pattern-regime-lookback", type=int, default=20)
    ap.add_argument("--pattern-regime-q-window", type=int, default=120)
    ap.add_argument("--pattern-forward-offset", type=int, default=1)
    ap.add_argument(
        "--all-session-windows",
        action="store_true",
        help="تشغيل المسار الصارم على كل نوافذ UTC (asia/london/ny + تقاطعات + فتح) وحفظ ملخصات في multi-session-out",
    )
    ap.add_argument(
        "--multi-session-out",
        default="session_validation",
        help="مجلد المخرجات عند --all-session-windows",
    )
    ap.add_argument(
        "--batch-ic-period",
        default="D",
        help="فترة IC داخل كل نافذة عند --all-session-windows (غالبًا D لشموع داخل اليوم)",
    )
    ap.add_argument("--multi-universal-min", type=int, default=5)
    ap.add_argument("--multi-verbose", action="store_true", help="طباعة verbose من كل run_full_validation")
    ap.add_argument(
        "--nonlinear-probe",
        action="store_true",
        help="فحص غير خطي (خمسينات من Train على Test + KNN/RF)؛ لا يحتاج nonlinear_feature_probe.py",
    )
    ap.add_argument("--nonlinear-methods", default="quintile,knn", help="quintile,knn,rf")
    ap.add_argument("--nonlinear-out-prefix", default=None)
    ap.add_argument("--nonlinear-knn-k", type=int, default=5)
    ap.add_argument("--nonlinear-rf-trees", type=int, default=100)
    ap.add_argument("--nonlinear-rf-max-depth", type=int, default=6, help="0 = بدون حد أقصى للعمق في RF")
    ap.add_argument("--nonlinear-seed", type=int, default=42)
    ap.add_argument(
        "--nonlinear-all-session-windows",
        action="store_true",
        help="فحص غير خطي على كل نوافذ UTC (asia/london/ny + تقاطعات + فتح) — نفس مجموعة --all-session-windows",
    )
    ap.add_argument(
        "--nonlinear-batch-out",
        default="nonlinear_session_validation",
        help="مجلد حفظ ملفات nonlinear عند --nonlinear-all-session-windows",
    )
    ap.add_argument(
        "--nonlinear-two-layer",
        action="store_true",
        help=(
            "مع --nonlinear-probe أو --nonlinear-all-session-windows: فلتر منطقة سعرية (طبقة 1) "
            "ثم quintile/KNN/RF على ميزات التدفق فقط (--nl-flow-features)"
        ),
    )
    ap.add_argument(
        "--nl-zone-feature",
        default="dist_to_pdh",
        help="عمود طبقة المنطقة (افتراضي: dist_to_pdh؛ مسافة أصغر = أقرب لـ PDH)",
    )
    ap.add_argument(
        "--nl-zone-mode",
        default="near_pdh",
        help=(
            "near_pdh | far_pdh | asia_near_high | asia_near_low | london_near_high | london_near_low "
            "(الأخيرة: قرب قمة/قاع جاري داخل يوم التداول ضمن نفس نافذة الجلسة؛ تحتاج عمود سعر — micro_price أو --nl-price-col)"
        ),
    )
    ap.add_argument(
        "--nl-zone-fraction",
        type=float,
        default=0.25,
        help="حصّة الصفوف المحفوظة من الذيل (0–0.5)، مثل 0.25 ≈ أقرب ربع لمرجع المنطقة",
    )
    ap.add_argument(
        "--nl-price-col",
        default=None,
        help="عمود السعر لحساب قمة/قاع الجلسة (asia/london_near_*); افتراضي: أول عمود موجود من micro_price ثم close…",
    )
    ap.add_argument(
        "--nl-sweep-asia-london-extremes",
        action="store_true",
        help=(
            "مع --nonlinear-two-layer و --nonlinear-all-session-windows: لنافذتي asia ولندن فقط، "
            "شغّل تلقائيًا قرب القمة ثم قرب القاع (أربعة مخرجات فرعية لكل من الجلستين)"
        ),
    )
    ap.add_argument(
        "--nl-flow-features",
        default="absorption_intensity,cvd,obi,hawkes_intensity",
        help="قائمة مفصولة بفواصل — ميزات طبقة التدفق عند --nonlinear-two-layer",
    )
    args = ap.parse_args()

    use_rigorous = bool(args.rigorous) and not bool(args.legacy)

    path_in = os.path.abspath(os.path.expanduser(str(args.parquet)))
    try:
        df = load_feature_artifact(path_in)
    except FileNotFoundError as exc:
        raise SystemExit(f"تعذّر تحميل البيانات من {path_in!r}: {exc}") from exc
    df = df.copy()
    if df.empty:
        raise SystemExit(f"إطار فارغ بعد التحميل من {path_in!r}")
    ts = _resolve_ts_series(df)
    df["_alpha_ts"] = ts
    if "ts_event" not in df.columns:
        df["ts_event"] = ts
    else:
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)

    if args.forward_col not in df.columns:
        raise SystemExit(
            f"العمود {args.forward_col} غير موجود بعد التحميل. جرّب --forward-col forward_return "
            f"(مصفاة prepare_training_data) أو fwd_ret_clean (day trade)."
        )

    if getattr(args, "nonlinear_two_layer", False) and (
        args.nonlinear_probe or args.nonlinear_all_session_windows
    ):
        want_flow = _parse_nl_flow_features(args.nl_flow_features)
        feats_all = [f for f in want_flow if f in df.columns]
        missing_flow = [f for f in want_flow if f not in df.columns]
        if missing_flow:
            print(
                f"⚠️  --nonlinear-two-layer: أعمدة تدفق غير موجودة وتُتخطّى: {missing_flow}",
                file=sys.stderr,
            )
    else:
        feats_all = [f for f in load_feature_list(args.features) if f in df.columns]

    if args.nonlinear_all_session_windows:
        if not feats_all:
            print("❌ لا توجد فيتشر مطابقة للـ nonlinear-all-session-windows.", file=sys.stderr)
            sys.exit(2)
        _run_nonlinear_probe_all_windows(args, df, feats_all)
        sys.exit(0)

    if args.nonlinear_probe:
        df_nl = filter_session(df, args.session, args.session_profile)
        feats_nl = [f for f in feats_all if f in df_nl.columns]
        if not feats_nl:
            print("❌ لا توجد فيتشر مطابقة بعد فلتر الجلسة للـ nonlinear-probe.", file=sys.stderr)
            sys.exit(2)
        if args.forward_col not in df_nl.columns:
            print(f"❌ عمود {args.forward_col} غير موجود بعد الفلتر.", file=sys.stderr)
            sys.exit(2)
        _run_nonlinear_probe_cli(args, df_nl, feats_nl)
        sys.exit(0)

    if args.all_session_windows:
        if not feats_all:
            print("❌ لا توجد فيتشر مطابقة لأعمدة الجدول.", file=sys.stderr)
            sys.exit(2)
        if not _HAS_STATSMODELS:
            print("ملاحظة: ثبّت statsmodels لتفعيل FDR الكامل: pip install statsmodels", file=sys.stderr)
        print(
            f"🔀 Multi-session windows | rows={len(df):,} | features={len(feats_all)} | "
            f"out={args.multi_session_out!r} | IC-period={args.batch_ic_period!r}"
        )
        results = run_multi_session_windows(
            df,
            feats_all,
            args.forward_col,
            train_ratio=args.train_ratio,
            min_ic=args.min_ic,
            min_icir=args.min_icir_rigorous,
            ic_period=args.batch_ic_period,
            output_dir=args.multi_session_out,
            verbose_each=args.multi_verbose,
        )
        if not results:
            print("❌ لا نتائج — وسّع التاريخ أو خفّض MULTI_SESSION_MIN_ROWS.", file=sys.stderr)
            sys.exit(2)
        print(f"\n{'═' * 50}\n🔀 Cross-Session Classification")
        cross = _classify_cross_session(results, universal_min_sessions=args.multi_universal_min)
        print(f"\n  Universal ({len(cross['universal'])}): {[u['feature'] for u in cross['universal']]}")
        print(f"  Bridge    ({len(cross['bridge'])}): {[b['feature'] for b in cross['bridge']]}")
        print(f"  Multi     ({len(cross['multi_session'])}): {[m['feature'] for m in cross['multi_session']]}")
        print(f"  Specific  ({len(cross['specific'])}): …")
        print(f"  Weak      ({len(cross['weak'])}): لا VALID_EDGE في أي نافذة")
        print(f"\n{'═' * 50}\n📋 Session Feature Maps")
        session_map = _build_multi_session_map(cross, results)
        _save_multi_session_artifacts(cross, session_map, args.multi_session_out)
        print("\n✅ انتهى --all-session-windows.")
        sys.exit(0)

    df_s = filter_session(df, args.session, args.session_profile)
    feats = [f for f in feats_all if f in df_s.columns]
    print(f"Rows (filtered): {len(df_s):,} | session={args.session!r} profile={args.session_profile!r}")
    print(f"Features: {len(feats)} | mode={'rigorous' if use_rigorous else 'legacy'}")

    stem = Path(args.parquet).stem
    out_default = Path(args.parquet).with_name(
        f"{stem}_alpha_{args.session}_{'rigorous' if use_rigorous else 'legacy'}.csv"
    )
    out_path = args.output_csv or str(out_default)

    if use_rigorous:
        if not _HAS_STATSMODELS:
            print("ملاحظة: ثبّت statsmodels لتفعيل FDR الكامل: pip install statsmodels", file=sys.stderr)
        try:
            val = AlphaValidatorRigorous(
                train_ratio=args.train_ratio,
                min_ic=args.min_ic,
                min_icir=args.min_icir_rigorous,
                ic_ttest_alpha=args.ic_ttest_alpha,
                n_splits=args.n_splits,
                n_quantiles=args.quantiles,
                cost_per_trade=args.cost_per_turnover,
                ic_period=args.ic_period,
                stability_pos_ratio=args.stab_pos_ratio,
            )
            res = val.run_full_validation(
                df_s,
                feats,
                args.forward_col,
                fdr_alpha=args.fdr_alpha,
                verbose=True,
            )
        except ValueError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)
    else:
        val = AlphaValidator(
            min_ic=args.min_ic,
            min_icir=args.min_icir,
            ic_stable_icir=args.ic_stable_icir,
            min_rows_period=args.min_rows_period,
            ic_period=args.ic_period,
            stability_positive_ratio=args.stab_pos_ratio,
            cost_per_turnover=args.cost_per_turnover,
            quantiles=args.quantiles,
            long_q=args.long_q,
            short_q=args.short_q,
        )
        res = val.run(df_s, feats, args.forward_col, args.n_splits)

    if res.empty:
        print("لا نتائج.", file=sys.stderr)
        sys.exit(2)

    if not use_rigorous:
        strict = res[res["VALID_EDGE_strict"]]
        print(f"\nStrict pass (legacy): {len(strict)} / {len(res)}")
        print(res.head(25).to_string(index=False))

    res.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    if args.pattern_analysis:
        from pattern_analysis import run_pattern_analysis

        df_pa = df_s.copy()
        if "ts_event" not in df_pa.columns:
            df_pa["ts_event"] = df_pa["_alpha_ts"]

        prefix_base = str(Path(out_path).with_suffix("")) + "_pattern"
        pr, pd_decay, preg = run_pattern_analysis(
            df_pa,
            feats,
            args.forward_col,
            rolling_window_bars=args.pattern_rolling_window,
            max_horizon=args.pattern_max_horizon,
            eval_tail_frac=args.pattern_eval_tail,
            regime_lookback=args.pattern_regime_lookback,
            regime_quantile_window=args.pattern_regime_q_window,
            forward_start_offset=args.pattern_forward_offset,
        )
        pr.to_csv(f"{prefix_base}_rolling_ic.csv", index=False)
        pd_decay.to_csv(f"{prefix_base}_ic_decay.csv", index=False)
        preg.to_csv(f"{prefix_base}_regime_ic.csv", index=False)
        print(
            f"Pattern analysis: {prefix_base}_rolling_ic.csv | "
            f"{prefix_base}_ic_decay.csv | {prefix_base}_regime_ic.csv"
        )


if __name__ == "__main__":
    main()
