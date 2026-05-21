"""
Thin CLI wrapper — المنطق الكامل داخل alpha_validation.py (--all-session-windows).

  python session_feature_validator.py --parquet FEAT.parquet --forward-col fwd_ret_clean

أو مباشرة:

  python alpha_validation.py --parquet FEAT.parquet --forward-col fwd_ret_clean --all-session-windows
"""

from __future__ import annotations

import sys

from alpha_validation import (
    _build_multi_session_map,
    _classify_cross_session,
    _save_multi_session_artifacts,
    load_feature_list,
    run_multi_session_windows,
)


def main() -> None:
    import argparse
    import pandas as pd

    from alpha_validation import _HAS_STATSMODELS, _resolve_ts_series

    p = argparse.ArgumentParser(description="Wrapper لـ alpha_validation --all-session-windows")
    p.add_argument("--parquet", required=True)
    p.add_argument("--features", default=None)
    p.add_argument("--forward-col", default="fwd_ret_clean")
    p.add_argument("--output-dir", default="session_validation")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--min-ic", type=float, default=0.05)
    p.add_argument("--min-icir", type=float, default=0.30)
    p.add_argument("--ic-period", default="D")
    p.add_argument("--universal-min-sessions", type=int, default=5)
    p.add_argument("--verbose-validator", action="store_true")
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    df = df.copy()
    ts = _resolve_ts_series(df)
    df["_alpha_ts"] = ts
    if "ts_event" not in df.columns:
        df["ts_event"] = ts
    else:
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)

    if args.forward_col not in df.columns:
        print(f"❌ عمود غير موجود: {args.forward_col}", file=sys.stderr)
        sys.exit(2)

    feats = [f for f in load_feature_list(args.features) if f in df.columns]
    if not feats:
        print("❌ لا فيتشر مطابقة.", file=sys.stderr)
        sys.exit(2)

    if not _HAS_STATSMODELS:
        print("ملاحظة: pip install statsmodels لتفعيل FDR", file=sys.stderr)

    results = run_multi_session_windows(
        df,
        feats,
        args.forward_col,
        train_ratio=args.train_ratio,
        min_ic=args.min_ic,
        min_icir=args.min_icir,
        ic_period=args.ic_period,
        output_dir=args.output_dir,
        verbose_each=args.verbose_validator,
    )
    if not results:
        print("❌ لا نتائج.", file=sys.stderr)
        sys.exit(2)

    cross = _classify_cross_session(results, universal_min_sessions=args.universal_min_sessions)
    session_map = _build_multi_session_map(cross, results)
    _save_multi_session_artifacts(cross, session_map, args.output_dir)
    print("\n✅ اكتمل.")


if __name__ == "__main__":
    main()
