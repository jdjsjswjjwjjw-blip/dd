"""
prepare_day_trading_enriched.py — Phase 3 integration:
prepare_day_trading + V19.2 features enrichment.

ينتج نفس مخرج prepare_day_trading.py لكن مع 40+ feature إضافي:
  +18 sim_* (feature_simulators)
  + 8 wall_depth_* (لو mbp متاح)
  + 5 iceberg_* (لو mbo متاح)
  + zone_full + 12 levels (session_mapper)
  + 9 bp_* (Bell pairs)
  = 87 → ~138 feature (التقرير المرحلة 3)

استخدام (مماثل لـ prepare_day_trading.py):
    python prepare_day_trading_enriched.py \\
        --mbo path/to/mbo.parquet \\
        --mbp path/to/mbp.parquet \\
        --output pipeline_day_trading/enriched_features

الأمر يستدعي prepare_day_trading.py الأصلي ثم يضيف V19.2 features.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# الـ ROOT path للـ modules
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def enrich_parquet(
    input_path: str | Path,
    output_path: str | Path,
    mbp_path: str | Path | None = None,
    mbo_path: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """يأخذ parquet (مخرج prepare_day_trading) ويضيف V19.2 features.

    Parameters
    ----------
    input_path : parquet ناتج من prepare_day_trading.py
    output_path : parquet للناتج المُثرى
    mbp_path : optional MBP-10 parquet (للـ wall_depth_simulator)
    mbo_path : optional MBO parquet (للـ iceberg_simulator)
    verbose : print progress

    Returns
    -------
    dict مع counts (before, after, added)
    """
    from modules.feature_enrichment import (
        enrich_features, EnrichmentConfig, feature_count_summary,
    )

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"📥 قراءة {input_path}")
    df = pd.read_parquet(input_path)
    n_before = len(df.columns)

    mbp = None
    mbo = None
    if mbp_path is not None:
        if verbose:
            print(f"📥 قراءة MBP من {mbp_path}")
        mbp = pd.read_parquet(mbp_path)
    if mbo_path is not None:
        if verbose:
            print(f"📥 قراءة MBO من {mbo_path}")
        mbo = pd.read_parquet(mbo_path)

    cfg = EnrichmentConfig(
        add_simulators=True,
        add_wall_depth=(mbp is not None),
        add_iceberg=(mbo is not None),
        add_session_mapping=True,
        add_bell_pairs=True,
        skip_missing_cols=True,
        verbose=verbose,
    )

    if verbose:
        print("\n🔮 Phase 3: Feature Enrichment...")
        summary = feature_count_summary(cfg)
        print(f"   متوقع إضافة: {summary}")

    enriched = enrich_features(df, config=cfg, mbp=mbp, mbo=mbo)
    n_after = len(enriched.columns)

    if verbose:
        print(f"\n💾 حفظ {output_path}")
    enriched.to_parquet(output_path, index=False)

    if verbose:
        print(f"\n✅ تم: {n_before} → {n_after} عمود (+{n_after - n_before})")

    return {
        "input_cols": n_before,
        "output_cols": n_after,
        "added": n_after - n_before,
        "rows": len(enriched),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Phase 3: enrich prepare_day_trading output with V19.2 features. "
            "اختياري: لو ما عندك مخرج prepare_day_trading جاهز، "
            "شغّل prepare_day_trading.py أولاً ثم هذا الـ script."
        )
    )
    ap.add_argument(
        "--input",
        required=True,
        help="مسار parquet ناتج من prepare_day_trading.py",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="مسار الـ enriched parquet (default: input_path._enriched.parquet)",
    )
    ap.add_argument("--mbp", default=None, help="optional: MBP-10 parquet للـ wall_depth")
    ap.add_argument("--mbo", default=None, help="optional: MBO parquet للـ iceberg")
    ap.add_argument("--quiet", action="store_true", help="suppress progress prints")
    args = ap.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        output_path = input_path.with_name(input_path.stem + "_enriched.parquet")
    else:
        output_path = Path(args.output)

    result = enrich_parquet(
        input_path=input_path,
        output_path=output_path,
        mbp_path=args.mbp,
        mbo_path=args.mbo,
        verbose=not args.quiet,
    )

    print(f"\n📋 {result}")


if __name__ == "__main__":
    main()
