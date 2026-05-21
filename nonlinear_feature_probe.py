"""
Wrapper لـ alpha_validation: nonlinear على جلسة واحدة أو كل النوافذ + التقاطعات.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from alpha_validation import (
    _run_nonlinear_probe_all_windows,
    _run_nonlinear_probe_cli,
    _resolve_ts_series,
    filter_session,
    load_feature_list,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Nonlinear probe — جلسة واحدة أو كل النوافذ")
    p.add_argument("--parquet", required=True)
    p.add_argument("--session", default="asia", help="مع --all-session-windows يُتجاهل")
    p.add_argument("--session-profile", default="daytrade_default")
    p.add_argument("--forward-col", default="fwd_ret_clean")
    p.add_argument("--features", default=None)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--methods", default="quintile,knn")
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--rf-trees", type=int, default=100)
    p.add_argument("--rf-max-depth", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-prefix", default=None)
    p.add_argument(
        "--all-session-windows",
        "--all_session_windows",
        action="store_true",
        dest="all_session_windows",
        help="كل الجلسات والتقاطعات (asia/london/ny/asia_london/london_ny/…)",
    )
    p.add_argument(
        "--batch-out",
        "--batch_out",
        default="nonlinear_session_validation",
        dest="batch_out",
        help="مجلد المخرجات مع --all-session-windows",
    )
    a = p.parse_args()

    df = pd.read_parquet(a.parquet)
    df = df.copy()
    df["_alpha_ts"] = _resolve_ts_series(df)
    if "ts_event" not in df.columns:
        df["ts_event"] = df["_alpha_ts"]
    else:
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)

    if a.forward_col not in df.columns:
        print(f"❌ عمود غير موجود: {a.forward_col}", file=sys.stderr)
        sys.exit(2)

    feats_all = [f for f in load_feature_list(a.features) if f in df.columns]
    if not feats_all:
        print("❌ لا فيتشر مطابقة.", file=sys.stderr)
        sys.exit(2)

    stem = Path(a.parquet).stem
    prefix = a.output_prefix or str(Path(a.parquet).with_name(f"{stem}_nonlinear_{a.session}"))

    bridge = argparse.Namespace(
        parquet=a.parquet,
        session=a.session,
        forward_col=a.forward_col,
        train_ratio=a.train_ratio,
        nonlinear_methods=a.methods,
        nonlinear_out_prefix=prefix,
        nonlinear_knn_k=a.knn_k,
        nonlinear_rf_trees=a.rf_trees,
        nonlinear_rf_max_depth=a.rf_max_depth,
        nonlinear_seed=a.seed,
        nonlinear_batch_out=a.batch_out,
    )

    if a.all_session_windows:
        _run_nonlinear_probe_all_windows(bridge, df, feats_all)
    else:
        df_s = filter_session(df, a.session, a.session_profile)
        feats = [f for f in feats_all if f in df_s.columns]
        if not feats:
            print("❌ لا فيتشر بعد فلتر الجلسة.", file=sys.stderr)
            sys.exit(2)
        _run_nonlinear_probe_cli(bridge, df_s, feats)


if __name__ == "__main__":
    main()
