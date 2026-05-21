"""
diagnose_stage1_only.py
=======================

Strong diagnostics focused on Stage1 artifacts only.
"""

from __future__ import annotations

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.feature_selection import mutual_info_classif
except Exception:
    mutual_info_classif = None

try:
    from prepare_training_data import CATBOOST_ADVISOR_FEATURES
except Exception:
    CATBOOST_ADVISOR_FEATURES = []


FORBIDDEN_SIGNAL_COLS = {
    "bias_label",
    "soft_label",
    "soft_label_long",
    "soft_label_short",
    "path_outcome",
    "event_score",
    "forward_return",
    "label_end_ts",
    "train_event_flag",
    "event_flag",
    "signal_quality",
    "conf_label",
    "is_event",
    "is_expansion",
    "adverse_path_flag",
    "bias_label_detail",
    "neutral_reason",
    "soft_sample_weight",
    "mc_sample_weight",
    "label_confidence",
    "label_horizon_steps",
    "effective_horizon",
    "timeout_move_exceeded_band",
}


def _load_parquet(path: str, max_files: int = 120) -> pd.DataFrame:
    p = Path(path)
    if p.is_file():
        return pd.read_parquet(p)
    files = sorted(p.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {path}")
    use_files = files[: max(1, int(max_files))]
    return pd.concat([pd.read_parquet(f) for f in use_files], ignore_index=True)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _ts_profile(df: pd.DataFrame) -> dict[str, Any]:
    if "ts_event" not in df.columns:
        return {"available": False, "reason": "ts_event missing"}
    ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    valid = ts.dropna()
    return {
        "available": True,
        "rows": int(len(df)),
        "invalid_ts": int(ts.isna().sum()),
        "duplicates": int(ts.duplicated().sum()),
        "monotonic": bool(valid.is_monotonic_increasing),
        "min": None if valid.empty else str(valid.min()),
        "max": None if valid.empty else str(valid.max()),
    }


def _feature_health(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    present = [f for f in features if f in df.columns]
    dead: list[str] = []
    weak: list[str] = []
    rows: list[dict[str, Any]] = []
    for f in present:
        s = _num(df[f])
        v = s.dropna()
        zero = float((v.abs() <= 1e-12).mean()) if len(v) else 1.0
        std = float(v.std(ddof=0)) if len(v) else 0.0
        nunique = int(v.nunique(dropna=True)) if len(v) else 0
        if zero >= 0.999 or std <= 1e-10 or nunique <= 1:
            dead.append(f)
        elif zero >= 0.98 or std < 1e-6 or nunique <= 3:
            weak.append(f)
        rows.append(
            {
                "feature": f,
                "nan_rate": float(s.isna().mean()),
                "zero_rate": zero,
                "std": std,
                "nunique": nunique,
            }
        )
    return {
        "tested": int(len(features)),
        "present": int(len(present)),
        "missing": [f for f in features if f not in df.columns],
        "dead": dead,
        "weak": weak,
        "profile": rows,
    }


def _label_profile(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "bias_label" in df.columns:
        y = _num(df["bias_label"]).fillna(2).astype(int)
        counts = y.value_counts().sort_index().to_dict()
        out["bias_counts"] = {str(k): int(v) for k, v in counts.items()}
        out["directional_rate"] = float(np.isin(y, [0, 1]).mean())
    if "train_event_flag" in df.columns:
        te = _num(df["train_event_flag"]).fillna(0).astype(int)
        out["train_event_rate"] = float((te > 0).mean())
    if "path_outcome" in df.columns:
        po = _num(df["path_outcome"]).fillna(-1).astype(int)
        out["path_outcome_counts"] = {str(k): int(v) for k, v in po.value_counts().sort_index().to_dict().items()}
    return out


def _soft_semantics(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    needed = {"bias_label", "soft_label"}
    if not needed.issubset(df.columns):
        out["reason"] = "bias_label/soft_label missing"
        return out
    y = _num(df["bias_label"]).fillna(2).astype(int)
    s = _num(df["soft_label"]).fillna(0.5).clip(1e-4, 1 - 1e-4)
    out["available"] = True
    out["soft_mean_all"] = float(s.mean())
    out["soft_mean_long_rows"] = float(s[y == 0].mean()) if int((y == 0).sum()) else None
    out["soft_mean_short_rows"] = float(s[y == 1].mean()) if int((y == 1).sum()) else None
    if "soft_label_long" in df.columns:
        sl = _num(df["soft_label_long"]).fillna(0.5).clip(1e-4, 1 - 1e-4)
        out["soft_label_long_mean"] = float(sl.mean())
        out["soft_label_long_on_long_rows"] = float(sl[y == 0].mean()) if int((y == 0).sum()) else None
        out["soft_label_long_on_short_rows"] = float(sl[y == 1].mean()) if int((y == 1).sum()) else None
        out["soft_target_ready"] = True
    else:
        out["soft_target_ready"] = False
    out["note"] = (
        "soft_label is P(win|current direction), while soft_label_long is P(LONG). "
        "Use soft_label_long for directional regression targets."
    )
    return out


def _mi_top(df: pd.DataFrame, top_k: int = 20) -> dict[str, Any]:
    if mutual_info_classif is None:
        return {"available": False, "reason": "sklearn unavailable"}
    if "bias_label" not in df.columns:
        return {"available": False, "reason": "bias_label missing"}
    y = _num(df["bias_label"]).fillna(2).astype(int)
    candidates = [c for c in df.columns if c not in ("ts_event",) and c not in FORBIDDEN_SIGNAL_COLS]
    num_cols: list[str] = []
    for c in candidates:
        s = _num(df[c])
        if int(s.notna().sum()) >= 40 and float(s.fillna(0.0).std(ddof=0)) > 1e-10:
            num_cols.append(c)
    if not num_cols:
        return {"available": True, "features": 0, "top": []}
    x = df[num_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    mi = mutual_info_classif(x.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.int32), random_state=42)
    order = np.argsort(mi)[::-1]
    top = [{"feature": num_cols[int(i)], "mi": float(mi[int(i)])} for i in order[:top_k]]
    return {"available": True, "features": int(len(num_cols)), "top": top}


def _leakage_audit(df: pd.DataFrame) -> dict[str, Any]:
    present_forbidden = [c for c in sorted(FORBIDDEN_SIGNAL_COLS) if c in df.columns]
    suspicious = []
    if "bias_label" in df.columns and mutual_info_classif is not None:
        y = _num(df["bias_label"]).fillna(2).astype(int)
        for c in present_forbidden:
            if c in ("bias_label", "label_end_ts", "train_event_flag"):
                continue
            s = _num(df[c])
            if int(s.notna().sum()) < 40 or float(s.fillna(0.0).std(ddof=0)) <= 1e-10:
                continue
            xx = s.fillna(s.median() if np.isfinite(s.median()) else 0.0).to_numpy(dtype=np.float64).reshape(-1, 1)
            try:
                val = float(mutual_info_classif(xx, y.to_numpy(dtype=np.int32), random_state=42)[0])
            except Exception:
                val = 0.0
            if val >= 0.10:
                suspicious.append({"feature": c, "mi_vs_bias": val})
    return {
        "forbidden_columns_present": present_forbidden,
        "high_mi_forbidden_columns": suspicious,
        "note": "Presence here is expected in artifacts; they must stay excluded from model inputs.",
    }


def _markdown(rep: dict[str, Any]) -> str:
    h = rep["feature_health"]
    lines = [
        "# Stage1 Only Diagnostic",
        "",
        "## Executive Summary",
        f"- Rows: {rep['rows']}",
        f"- Time monotonic: {rep['time_profile'].get('monotonic')} | duplicates={rep['time_profile'].get('duplicates')}",
        f"- Core features present: {h['present']}/{h['tested']}",
        f"- Dead features: {len(h['dead'])}",
        f"- Weak features: {len(h['weak'])}",
        f"- Directional rate: {rep['labels'].get('directional_rate')}",
        f"- Train-event rate: {rep['labels'].get('train_event_rate')}",
        f"- Soft target ready (`soft_label_long`): {rep['soft_semantics'].get('soft_target_ready')}",
        "",
        "## Dead Features",
        f"- {h['dead']}",
        "",
        "## Soft Label Semantics",
        f"- soft_mean_all={rep['soft_semantics'].get('soft_mean_all')}",
        f"- soft_mean_long_rows={rep['soft_semantics'].get('soft_mean_long_rows')}",
        f"- soft_mean_short_rows={rep['soft_semantics'].get('soft_mean_short_rows')}",
        f"- soft_label_long_on_long_rows={rep['soft_semantics'].get('soft_label_long_on_long_rows')}",
        f"- soft_label_long_on_short_rows={rep['soft_semantics'].get('soft_label_long_on_short_rows')}",
        "",
        "## Leakage Audit (Artifact Level)",
        f"- Forbidden columns present: {rep['leakage'].get('forbidden_columns_present')}",
        f"- High-MI forbidden cols: {rep['leakage'].get('high_mi_forbidden_columns')}",
        "",
        "## Top MI (Allowed Features Only)",
    ]
    for row in rep["mi_allowed"].get("top", [])[:15]:
        lines.append(f"- {row['feature']}: {row['mi']:.5f}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Strong Stage1-only diagnostics")
    ap.add_argument("--stage1", required=True, help="Stage1 parquet file or directory")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_files", type=int, default=120)
    ap.add_argument("--print_report", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = _load_parquet(args.stage1, max_files=args.max_files)

    health = _feature_health(df, list(CATBOOST_ADVISOR_FEATURES))
    report = {
        "inputs": {"stage1": args.stage1},
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "time_profile": _ts_profile(df),
        "feature_health": health,
        "labels": _label_profile(df),
        "soft_semantics": _soft_semantics(df),
        "mi_allowed": _mi_top(df),
        "leakage": _leakage_audit(df),
    }

    json_path = os.path.join(args.out_dir, "stage1_only_diagnostic.json")
    md_path = os.path.join(args.out_dir, "stage1_only_diagnostic.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    md = _markdown(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"saved_json={json_path}")
    print(f"saved_md={md_path}")
    if args.print_report:
        print()
        print(md)


if __name__ == "__main__":
    main()

