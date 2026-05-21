"""
edge_scanner_v2.py — Phase A: Vectorized version of edge_scanner.py
══════════════════════════════════════════════════════════════════════

التحسين الرئيسي:
  الـ original يستدعي _apply_combo داخل الـ nested loop:
    18 zones × 12 levels × 5 events × 33 combos = 35,640 استدعاء

  v2 يـ pre-compute combo masks مرة واحدة (33 استدعاء فقط):
    - global_combo_masks[combo_name] = bool array of len(df)
    - cell membership = bool AND بين zone_mask & event_mask
    - sub_mask = cell_mask & global_combo_masks[combo_name]

  النتيجة المتوقعة: 5-20× speedup على stage-1 scan.
  النتائج المطابقة 1:1 للـ original.

Public API يطابق الأصلي:
  scan_edges_v2(df, horizons, symbol, confidence, ...) → dict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_specs import get_market_spec, derive_edge_criteria
from statistics_module import (
    benjamini_hochberg, permutation_test_edge,
    make_3way_split, edge_statistics, validate_edge_pipeline,
)
from edge_scanner import (
    SIMULATOR_FILTERS,
    FILTER_COMBOS,
    LEVEL_NAMES,
    EVENT_TYPES,
)


# ══════════════════════════════════════════════════════════════════
# Pre-computation helpers — هذا هو الـ speedup الرئيسي
# ══════════════════════════════════════════════════════════════════

def _precompute_combo_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """يحسب bool mask للـ combo على full df مرة واحدة.

    قبل: _apply_combo يُستدعى داخل الـ inner loop 35,640 مرة.
    بعد: 33 استدعاء فقط (واحد لكل combo) — كل filter يُحسب مرة.
    """
    n = len(df)
    # Step 1: حساب filter masks مرة واحدة لكل filter (27 فقط)
    filter_masks: dict[str, np.ndarray] = {}
    for fk, (col, op, thr) in SIMULATOR_FILTERS.items():
        if col not in df.columns:
            filter_masks[fk] = np.zeros(n, dtype=bool)
            continue
        s = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
        if op == ">":
            filter_masks[fk] = s > thr
        elif op == "<":
            filter_masks[fk] = s < thr
        else:
            filter_masks[fk] = np.zeros(n, dtype=bool)

    # Step 2: مزج filters للحصول على combo mask (AND بين filters)
    combo_masks: dict[str, np.ndarray] = {}
    for combo_name, direction, filts in FILTER_COMBOS:
        if not filts:
            combo_masks[combo_name] = np.zeros(n, dtype=bool)
            continue
        # AND مع كل filter
        mask = np.ones(n, dtype=bool)
        valid = True
        for fk in filts:
            if fk not in filter_masks:
                valid = False
                break
            mask &= filter_masks[fk]
        combo_masks[combo_name] = mask if valid else np.zeros(n, dtype=bool)
    return combo_masks


def _precompute_event_masks(
    df: pd.DataFrame,
    level_names: list[str],
    event_types: list[str],
) -> dict[tuple[str, str], np.ndarray]:
    """يحسب bool mask لكل (level, event) pair مرة واحدة.

    قبل: مقارنة `df[event_col] == event` تنفّذ داخل الـ loop.
    بعد: 12 × 5 = 60 array مُحسبت مسبقاً.
    """
    n = len(df)
    masks: dict[tuple[str, str], np.ndarray] = {}
    for level in level_names:
        ev_col = f"{level}_event"
        if ev_col not in df.columns:
            for event in event_types:
                masks[(level, event)] = np.zeros(n, dtype=bool)
            continue
        col_vals = df[ev_col].to_numpy()
        for event in event_types:
            masks[(level, event)] = (col_vals == event)
    return masks


def _precompute_zone_masks(
    df: pd.DataFrame,
    zones: list[str],
) -> dict[str, np.ndarray]:
    """zone membership كـ bool arrays."""
    col_vals = df["zone_full"].to_numpy()
    return {z: (col_vals == z) for z in zones}


def _precompute_fwd_rets(
    df: pd.DataFrame,
    horizons: list[int],
) -> dict[int, np.ndarray]:
    """numpy arrays للـ fwd_ret_h لتفادي تكرار `.to_numpy()`."""
    return {h: df[f"fwd_ret_{h}"].to_numpy() for h in horizons}


def _ensure_fwd_rets_inplace(df: pd.DataFrame, horizons: list[int]) -> None:
    """يضمن وجود أعمدة fwd_ret_h (نسخة من scan_edges الأصلية)."""
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    for h in horizons:
        col = f"fwd_ret_{h}"
        if col not in df.columns:
            arr = np.full(len(c), np.nan)
            if len(c) > h:
                arr[:-h] = (c[h:] - c[:-h]) / c[:-h]
            df[col] = arr


# ══════════════════════════════════════════════════════════════════
# Main scan — vectorized version
# ══════════════════════════════════════════════════════════════════

def scan_edges_v2(
    df: pd.DataFrame,
    horizons: list[int] = [3, 6, 12],
    symbol: str = "6B",
    confidence: str = "medium",
    run_permutation: bool = True,
    n_permutations: int = 1000,
    verbose: bool = True,
) -> dict:
    """نسخة vectorized من scan_edges. النتائج مطابقة 1:1 للـ original.

    التحسينات (Phase A):
      ① pre-compute filter masks مرة واحدة (27 فقط بدل 35,640)
      ② pre-compute (level, event) masks (60 array)
      ③ pre-compute zone masks
      ④ numpy bool AND للـ cell membership (بدل pandas slicing)
      ⑤ numpy indexing مباشر للـ rets/days (بدل DataFrame indexing)
    """
    df = df.copy()
    if "ts_event" in df.columns:
        df = df.sort_values("ts_event").reset_index(drop=True)
    df["day"] = pd.to_datetime(df["ts_event"]).dt.date

    _ensure_fwd_rets_inplace(df, horizons)

    spec = get_market_spec(symbol)
    n_train_days = df["day"].nunique() // 2
    criteria = derive_edge_criteria(spec, n_train_days, confidence)

    max_horizon = max(horizons)
    purge_bars = max_horizon
    split = make_3way_split(
        n=len(df),
        train_ratio=criteria["train_ratio"],
        validation_ratio=criteria["validation_ratio"],
        purge_bars=purge_bars,
    )

    df_train = df.iloc[split.train_idx].reset_index(drop=True)
    df_val = df.iloc[split.validation_idx].reset_index(drop=True)
    df_hold = df.iloc[split.holdout_idx].reset_index(drop=True)

    if verbose:
        print(f"🔍 Edge Scanner V19.2 (vectorized) — symbol={symbol}, confidence={confidence}")
        print(f"   3-way split:")
        print(f"     train     : {split.n_train:,} ({split.n_train/len(df):.0%})")
        print(f"     validation: {split.n_validation:,} ({split.n_validation/len(df):.0%})")
        print(f"     holdout   : {split.n_holdout:,} ({split.n_holdout/len(df):.0%})")
        print(f"     purge     : {purge_bars} bars × 2")
        print(f"   transaction cost: {criteria['round_trip_cost']:.2f} pips/trade")
        print(f"   criteria: t_train≥{criteria['min_t_stat']}, "
              f"t_oos≥{criteria['min_t_oos']}, wr≥{criteria['min_wr']:.0%}, "
              f"min_avg≥{criteria['min_avg_pips']:.1f}p")

    zones_all = [z for z in df["zone_full"].unique() if z != "off_session"]

    # ── Phase A pre-computation ───────────────────────────────────────────
    combo_masks_tr = _precompute_combo_masks(df_train)
    event_masks_tr = _precompute_event_masks(df_train, LEVEL_NAMES, EVENT_TYPES)
    zone_masks_tr = _precompute_zone_masks(df_train, zones_all)
    fwd_rets_tr = _precompute_fwd_rets(df_train, horizons)
    days_tr_arr = df_train["day"].to_numpy()

    all_candidates_stage1 = []
    cells_scanned = 0

    for zone in zones_all:
        zone_mask = zone_masks_tr[zone]
        if zone_mask.sum() < 30:
            continue
        for level in LEVEL_NAMES:
            ev_col = f"{level}_event"
            if ev_col not in df.columns:
                continue
            for event in EVENT_TYPES:
                cell_mask = zone_mask & event_masks_tr[(level, event)]
                cell_n = int(cell_mask.sum())
                if cell_n < criteria["min_n"]:
                    continue
                cells_scanned += 1
                for combo_name, direction, filts in FILTER_COMBOS:
                    sub_mask = cell_mask & combo_masks_tr[combo_name]
                    if int(sub_mask.sum()) < criteria["min_n"]:
                        continue
                    for h in horizons:
                        rets_tr = fwd_rets_tr[h][sub_mask]
                        days_tr = days_tr_arr[sub_mask]
                        st_tr = edge_statistics(rets_tr, days_tr, direction)
                        if not st_tr.get("valid", False):
                            continue
                        if st_tr["n_days"] < criteria["min_days_recur"]:
                            continue
                        all_candidates_stage1.append({
                            "zone": zone, "level": level, "event": event,
                            "combo": combo_name, "direction": direction,
                            "filters": filts, "horizon": h,
                            "stats_train": st_tr,
                            "p_value": st_tr["p_value"],
                        })

    if verbose:
        print(f"\n   مرحلة 1 (مسح train): {cells_scanned} خلية، "
              f"{len(all_candidates_stage1)} مرشّح أولي")

    if not all_candidates_stage1:
        return _empty_result(criteria, cells_scanned, split)

    # ── Stage 2: FDR ──────────────────────────────────────────────────────
    p_values = np.array([c["p_value"] for c in all_candidates_stage1])
    rejected, fdr_threshold = benjamini_hochberg(p_values, criteria["fdr_alpha"])
    candidates_stage2 = [c for c, r in zip(all_candidates_stage1, rejected) if r]
    if verbose:
        print(f"   مرحلة 2 (FDR @ α={criteria['fdr_alpha']}): "
              f"{len(candidates_stage2)} اجتاز (p_threshold={fdr_threshold:.4f})")

    # ── Stage 3: Permutation ──────────────────────────────────────────────
    candidates_stage3 = []
    if run_permutation and candidates_stage2:
        for c in candidates_stage2:
            zm = zone_masks_tr[c["zone"]]
            em = event_masks_tr[(c["level"], c["event"])]
            cm = combo_masks_tr[c["combo"]]
            sub_mask = zm & em & cm
            rets = fwd_rets_tr[c["horizon"]][sub_mask]
            perm = permutation_test_edge(rets, c["direction"], n_permutations=n_permutations)
            c["permutation"] = perm
            if perm["p_value"] < 0.10:
                candidates_stage3.append(c)
        if verbose:
            print(f"   مرحلة 3 (Permutation): {len(candidates_stage3)} اجتاز")
    else:
        candidates_stage3 = candidates_stage2

    # ── Stage 4: Validation (مكرر الـ pre-computation على df_val) ──────────
    combo_masks_val = _precompute_combo_masks(df_val)
    event_masks_val = _precompute_event_masks(df_val, LEVEL_NAMES, EVENT_TYPES)
    zone_masks_val = _precompute_zone_masks(df_val, zones_all)
    fwd_rets_val = _precompute_fwd_rets(df_val, horizons)
    days_val_arr = df_val["day"].to_numpy()

    candidates_stage4 = []
    for c in candidates_stage3:
        if c["zone"] not in zone_masks_val:
            continue
        zm = zone_masks_val[c["zone"]]
        em = event_masks_val.get((c["level"], c["event"]))
        cm = combo_masks_val.get(c["combo"])
        if em is None or cm is None:
            continue
        sub_mask = zm & em & cm
        if int(sub_mask.sum()) < criteria["min_n_test"]:
            continue
        rets_val = fwd_rets_val[c["horizon"]][sub_mask]
        days_val = days_val_arr[sub_mask]
        st_val = edge_statistics(rets_val, c["direction"], days=None) if False else edge_statistics(rets_val, days_val, c["direction"])
        if not st_val.get("valid", False):
            continue
        c["stats_validation"] = st_val
        if validate_edge_pipeline(c["stats_train"], st_val, criteria, c["direction"]):
            candidates_stage4.append(c)
    if verbose:
        print(f"   مرحلة 4 (Validation): {len(candidates_stage4)} اجتاز")

    # ── Stage 5: Holdout (لمسة واحدة) ─────────────────────────────────────
    combo_masks_hold = _precompute_combo_masks(df_hold)
    event_masks_hold = _precompute_event_masks(df_hold, LEVEL_NAMES, EVENT_TYPES)
    zone_masks_hold = _precompute_zone_masks(df_hold, zones_all)
    fwd_rets_hold = _precompute_fwd_rets(df_hold, horizons)
    days_hold_arr = df_hold["day"].to_numpy()

    candidates_stage5 = []
    for c in candidates_stage4:
        if c["zone"] not in zone_masks_hold:
            continue
        zm = zone_masks_hold[c["zone"]]
        em = event_masks_hold.get((c["level"], c["event"]))
        cm = combo_masks_hold.get(c["combo"])
        if em is None or cm is None:
            continue
        sub_mask = zm & em & cm
        if int(sub_mask.sum()) < criteria["min_n_test"]:
            c["stats_holdout"] = {"valid": False, "n": int(sub_mask.sum())}
            continue
        rets_hold = fwd_rets_hold[c["horizon"]][sub_mask]
        days_hold = days_hold_arr[sub_mask]
        st_hold = edge_statistics(rets_hold, days_hold, c["direction"])
        c["stats_holdout"] = st_hold
        if st_hold.get("valid", False):
            candidates_stage5.append(c)
    if verbose:
        print(f"   مرحلة 5 (Holdout): {len(candidates_stage5)} اجتاز")

    return {
        "candidates": candidates_stage5,
        "criteria": criteria,
        "diagnostics": {
            "cells_scanned": cells_scanned,
            "stage1_candidates": len(all_candidates_stage1),
            "stage2_passed_fdr": len(candidates_stage2),
            "stage3_passed_permutation": len(candidates_stage3),
            "stage4_passed_validation": len(candidates_stage4),
            "stage5_passed_holdout": len(candidates_stage5),
        },
        "split_info": {
            "n_train": split.n_train,
            "n_validation": split.n_validation,
            "n_holdout": split.n_holdout,
        },
    }


def _empty_result(criteria, cells_scanned, split):
    return {
        "candidates": [], "criteria": criteria,
        "diagnostics": {
            "cells_scanned": cells_scanned,
            "stage1_candidates": 0,
            "stage2_passed_fdr": 0,
            "stage3_passed_permutation": 0,
            "stage4_passed_validation": 0,
            "stage5_passed_holdout": 0,
        },
        "split_info": {
            "n_train": split.n_train,
            "n_validation": split.n_validation,
            "n_holdout": split.n_holdout,
        },
    }


__all__ = [
    "scan_edges_v2",
    "_precompute_combo_masks",
    "_precompute_event_masks",
    "_precompute_zone_masks",
    "_precompute_fwd_rets",
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Vectorized edge scanner (Phase A)")
    ap.add_argument("--input", required=True, help="mapped.parquet")
    ap.add_argument("--output", default="edge_candidates_v2.json")
    ap.add_argument("--symbol", default="6B")
    ap.add_argument("--confidence", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--horizons", type=int, nargs="+", default=[3, 6, 12])
    ap.add_argument("--no-permutation", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    result = scan_edges_v2(
        df,
        horizons=args.horizons,
        symbol=args.symbol,
        confidence=args.confidence,
        run_permutation=not args.no_permutation,
    )

    out = Path(args.output)
    out.write_text(json.dumps(result, default=str, indent=2, ensure_ascii=False))
    print(f"\n✅ {len(result['candidates'])} alphas → {out}")
