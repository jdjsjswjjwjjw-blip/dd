"""
modules/feature_enrichment.py — Phase 3: تكامل features V19.2 مع المشروع الرئيسي.
══════════════════════════════════════════════════════════════════════

الفكرة (من التقرير، المرحلة 3):
  prepare_day_trading.py يولّد 87 feature.
  لو ضفنا V19.2 outputs + Bell pairs:
    + 18 sim_* من feature_simulators
    +  8 wall_depth_*
    +  5 iceberg_*
    + zones + 12 levels × 5 events (context)
    +  9 bp_* من Bell pairs
    = ~138 feature (4× تحسين density)

النموذج (CNN/LSTM/CatBoost) يستفيد من:
  • microstructure context (sim_*)
  • institutional signals (iceberg_*, wall_*)
  • session awareness (zones, levels)
  • entangled correlations (bp_*)

API:
    enrich_features(df, include=...) -> DataFrame
        يأخذ مخرج prepare_day_trading ويضيف V19.2 features.

    list_added_columns() -> list[str]
        الأعمدة المتوقع إضافتها.

ملاحظات:
  - الـ enrichment غير-destructive (df الأصلي يبقى).
  - skip_missing=True يتيح تشغيل partial (لو columns مفقودة).
  - الاستخدام: graceful integration بدون لمس prepare_day_trading.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ── إضافة v19_2/ إلى sys.path للاستيراد ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_V19_2 = os.path.join(os.path.dirname(_HERE), "v19_2")
if os.path.isdir(_V19_2) and _V19_2 not in sys.path:
    sys.path.insert(0, _V19_2)


# ── V19.2 components — lazy import ──
def _import_v19_2():
    """يستورد V19.2 components عند الحاجة (lazy لتجنّب hard dependency)."""
    from feature_simulators import run_all_simulators, SIMULATOR_OUTPUTS
    from wall_depth_simulator import simulate_wall_depth, WALL_DEPTH_OUTPUTS
    from iceberg_simulator import simulate_iceberg, ICEBERG_OUTPUTS
    from session_mapper import map_sessions
    from quantum.bell_pairs import apply_bell_pairs, get_bell_pair_columns
    return {
        "run_all_simulators": run_all_simulators,
        "SIMULATOR_OUTPUTS": SIMULATOR_OUTPUTS,
        "simulate_wall_depth": simulate_wall_depth,
        "WALL_DEPTH_OUTPUTS": WALL_DEPTH_OUTPUTS,
        "simulate_iceberg": simulate_iceberg,
        "ICEBERG_OUTPUTS": ICEBERG_OUTPUTS,
        "map_sessions": map_sessions,
        "apply_bell_pairs": apply_bell_pairs,
        "get_bell_pair_columns": get_bell_pair_columns,
    }


# ── Stage flags ──
@dataclass(frozen=True)
class EnrichmentConfig:
    """تحكّم بأي مراحل تُشغَّل."""
    add_simulators: bool = True      # 18 sim_*
    add_wall_depth: bool = True      # 8 wall_*
    add_iceberg: bool = True         # 5 iceberg_*
    add_session_mapping: bool = True # zones + 12 levels
    add_bell_pairs: bool = True      # 9 bp_*
    skip_missing_cols: bool = True
    verbose: bool = False


DEFAULT_CONFIG = EnrichmentConfig()


# ══════════════════════════════════════════════════════════════════
# Main enrichment function
# ══════════════════════════════════════════════════════════════════

def enrich_features(
    df: pd.DataFrame,
    config: EnrichmentConfig = DEFAULT_CONFIG,
    mbp: Optional[pd.DataFrame] = None,
    mbo: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """يأخذ DataFrame (مخرج prepare_day_trading) ويثري بـ V19.2 features.

    Parameters
    ----------
    df : DataFrame مع الأعمدة الأساسية (close, ts_event, ...)
    config : EnrichmentConfig يحدد ماذا يُضاف
    mbp : optional - MBP-10 trade-level data للـ wall_depth_simulator
    mbo : optional - MBO trade-level data للـ iceberg_simulator
        لو mbp/mbo مش متوفرين، الـ stages المعتمدة عليهم تتخطّى
        (بشرط config.skip_missing_cols=True).

    Returns
    -------
    DataFrame مع الأعمدة الجديدة مضافة. الـ df الأصلي لا يتغيّر.
    """
    out = df.copy()

    try:
        v19 = _import_v19_2()
    except ImportError as e:
        if config.skip_missing_cols:
            if config.verbose:
                print(f"⚠️  V19.2 import failed: {e} — تخطّي enrichment.")
            return out
        raise

    if config.add_simulators:
        if config.verbose:
            print("🔮 إضافة V19.2 simulators (18 outputs)...")
        try:
            out = v19["run_all_simulators"](out)
        except (KeyError, ValueError) as e:
            if not config.skip_missing_cols:
                raise
            if config.verbose:
                print(f"   ⚠️  simulators تخطّي: {e}")

    if config.add_wall_depth:
        if mbp is None:
            if config.verbose:
                print("🧱 wall_depth يحتاج mbp DataFrame — تخطّي.")
        else:
            if config.verbose:
                print("🧱 إضافة wall_depth (8 outputs)...")
            try:
                out = v19["simulate_wall_depth"](out, mbp)
            except (KeyError, ValueError) as e:
                if not config.skip_missing_cols:
                    raise
                if config.verbose:
                    print(f"   ⚠️  wall_depth تخطّي: {e}")

    if config.add_iceberg:
        if mbo is None:
            if config.verbose:
                print("🧊 iceberg يحتاج mbo DataFrame — تخطّي.")
        else:
            if config.verbose:
                print("🧊 إضافة iceberg (5 outputs)...")
            try:
                out = v19["simulate_iceberg"](out, mbo)
            except (KeyError, ValueError) as e:
                if not config.skip_missing_cols:
                    raise
                if config.verbose:
                    print(f"   ⚠️  iceberg تخطّي: {e}")

    if config.add_session_mapping:
        if config.verbose:
            print("🌐 إضافة session zones + 12 levels...")
        try:
            out = v19["map_sessions"](out)
        except (KeyError, ValueError, AttributeError, TypeError) as e:
            if not config.skip_missing_cols:
                raise
            if config.verbose:
                print(f"   ⚠️  session_mapping تخطّي: {e}")

    if config.add_bell_pairs:
        if config.verbose:
            print("⚛️  إضافة Bell pairs (9 entangled features)...")
        try:
            out = v19["apply_bell_pairs"](out, skip_missing=config.skip_missing_cols)
        except (KeyError, ValueError) as e:
            if not config.skip_missing_cols:
                raise
            if config.verbose:
                print(f"   ⚠️  bell_pairs تخطّي: {e}")

    if config.verbose:
        added = len(out.columns) - len(df.columns)
        print(f"\n✅ enrichment تم: {len(df.columns)} → {len(out.columns)} عمود (+{added})")

    return out


def list_added_columns(config: EnrichmentConfig = DEFAULT_CONFIG) -> list[str]:
    """الأعمدة المتوقع إضافتها بناءً على config (للتشخيص)."""
    try:
        v19 = _import_v19_2()
    except ImportError:
        return []

    cols: list[str] = []
    if config.add_simulators:
        cols.extend(v19["SIMULATOR_OUTPUTS"])
    if config.add_wall_depth:
        cols.extend(v19["WALL_DEPTH_OUTPUTS"])
    if config.add_iceberg:
        cols.extend(v19["ICEBERG_OUTPUTS"])
    if config.add_session_mapping:
        # zones + 12 levels × 5 events ≈ context columns
        cols.append("zone_full")
        for level in ["PDH", "PDL", "PD_50", "PWH", "PWL", "PW_50",
                      "asia_H", "asia_L", "london_H", "london_L", "ny_H", "ny_L"]:
            cols.append(f"{level}_event")
    if config.add_bell_pairs:
        cols.extend(v19["get_bell_pair_columns"]())
    return cols


def feature_count_summary(config: EnrichmentConfig = DEFAULT_CONFIG) -> dict[str, int]:
    """عدد الـ features المتوقع لكل stage (للتشخيص)."""
    try:
        v19 = _import_v19_2()
    except ImportError:
        return {"error": -1}

    counts = {}
    if config.add_simulators:
        counts["simulators"] = len(v19["SIMULATOR_OUTPUTS"])
    if config.add_wall_depth:
        counts["wall_depth"] = len(v19["WALL_DEPTH_OUTPUTS"])
    if config.add_iceberg:
        counts["iceberg"] = len(v19["ICEBERG_OUTPUTS"])
    if config.add_session_mapping:
        counts["session_zones"] = 1
        counts["session_levels"] = 12  # *_event columns
    if config.add_bell_pairs:
        counts["bell_pairs"] = len(v19["get_bell_pair_columns"]())
    counts["total_added"] = sum(counts.values())
    return counts


__all__ = [
    "EnrichmentConfig",
    "DEFAULT_CONFIG",
    "enrich_features",
    "list_added_columns",
    "feature_count_summary",
]
