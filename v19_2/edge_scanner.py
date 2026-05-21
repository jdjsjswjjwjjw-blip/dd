"""
edge_scanner.py — البحث عن edges بصرامة علمية كاملة (V19.2)
══════════════════════════════════════════════════════════════════════

تحسينات V19.2:
  ① 3-Way Time Split: train (60%) + validation (20%) + holdout (20%)
  ② Benjamini-Hochberg FDR (بدل Bonferroni)
  ③ Permutation Test (إضافياً)
  ④ Transaction Costs Integration
  ⑤ Single Source of Truth (market_specs)
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


SIMULATOR_FILTERS = {
    "depth_pressure_pos":  ("sim_depth_pressure",   ">",  0.20),
    "depth_pressure_neg":  ("sim_depth_pressure",   "<", -0.20),
    "informed_high":       ("sim_informed_prob",    ">",  0.50),
    "absorb_gate_open":    ("sim_absorb_intensity", "<",  0.40),
    "flow_dir_up":         ("sim_flow_direction",   ">",  0.5),
    "flow_dir_down":       ("sim_flow_direction",   "<", -0.5),
    "depth_imb_pos":       ("sim_depth_imbalance",  ">",  0.05),
    "depth_imb_neg":       ("sim_depth_imbalance",  "<", -0.05),
    "wall_growing":        ("sim_wall_growth",      ">",  0.10),
    "wall_shrinking":      ("sim_wall_growth",      "<", -0.10),
    "wall_consumed_high":  ("sim_wall_consumed",    ">",  0.30),
    "wall_solid":          ("sim_wall_consumed",    "<",  0.15),
    "wall_persistent":     ("sim_wall_persist",     ">",  0.70),
    "wall_approaching":    ("sim_wall_shift",       ">",  0.20),
    "wall_receding":       ("sim_wall_shift",       "<", -0.20),
    "bid_wall_near":       ("sim_wall_bid_level",   "<",  3.0),
    "ask_wall_near":       ("sim_wall_ask_level",   "<",  3.0),
    "bid_wall_big":        ("sim_wall_bid_size",    ">",  0.65),
    "ask_wall_big":        ("sim_wall_ask_size",    ">",  0.65),
    "iceberg_present":     ("sim_iceberg_prob",       ">",  0.60),
    "iceberg_strong":      ("sim_iceberg_prob",       ">",  0.75),
    "iceberg_buy":         ("sim_iceberg_side",       ">",  0.5),
    "iceberg_sell":        ("sim_iceberg_side",       "<", -0.5),
    "iceberg_replenish":   ("sim_iceberg_replenish",  ">",  0.50),
    "iceberg_stealth":     ("sim_iceberg_stealth",    ">",  0.30),
    "iceberg_powerful":    ("sim_iceberg_strength",   ">",  0.60),
}


def _build_filter_combos() -> list[tuple]:
    combos = []
    long_base = [
        ("dp_pos",                ["depth_pressure_pos"]),
        ("dp_pos+informed",       ["depth_pressure_pos", "informed_high"]),
        ("dp_pos+gate",           ["depth_pressure_pos", "absorb_gate_open"]),
        ("dp_pos+informed+gate",  ["depth_pressure_pos", "informed_high", "absorb_gate_open"]),
        ("flow_up+informed",      ["flow_dir_up", "informed_high"]),
        ("imb_pos+informed",      ["depth_imb_pos", "informed_high"]),
        ("bid_wall_building",     ["wall_growing", "bid_wall_near"]),
        ("bid_wall_solid+dp",     ["wall_solid", "bid_wall_near", "depth_pressure_pos"]),
        ("ask_wall_consumed",     ["wall_consumed_high", "ask_wall_near"]),
        ("bid_wall_big+informed", ["bid_wall_big", "informed_high"]),
        ("iceberg_buy_replenish", ["iceberg_buy", "iceberg_replenish"]),
        ("iceberg_buy+bid_wall",  ["iceberg_buy", "bid_wall_near", "iceberg_replenish"]),
        ("iceberg_buy+dp",        ["iceberg_buy", "depth_pressure_pos", "iceberg_present"]),
    ]
    for name, filts in long_base:
        combos.append((f"LONG_{name}", 1, filts))
    short_base = [
        ("dp_neg",                ["depth_pressure_neg"]),
        ("dp_neg+informed",       ["depth_pressure_neg", "informed_high"]),
        ("dp_neg+gate",           ["depth_pressure_neg", "absorb_gate_open"]),
        ("dp_neg+informed+gate",  ["depth_pressure_neg", "informed_high", "absorb_gate_open"]),
        ("flow_down+informed",    ["flow_dir_down", "informed_high"]),
        ("imb_neg+informed",      ["depth_imb_neg", "informed_high"]),
        ("ask_wall_building",     ["wall_growing", "ask_wall_near"]),
        ("ask_wall_solid+dp",     ["wall_solid", "ask_wall_near", "depth_pressure_neg"]),
        ("bid_wall_consumed",     ["wall_consumed_high", "bid_wall_near"]),
        ("ask_wall_big+informed", ["ask_wall_big", "informed_high"]),
        ("iceberg_sell_replenish",["iceberg_sell", "iceberg_replenish"]),
        ("iceberg_sell+ask_wall", ["iceberg_sell", "ask_wall_near", "iceberg_replenish"]),
        ("iceberg_sell+dp",       ["iceberg_sell", "depth_pressure_neg", "iceberg_present"]),
    ]
    for name, filts in short_base:
        combos.append((f"SHORT_{name}", -1, filts))
    return combos


FILTER_COMBOS = _build_filter_combos()


def _apply_combo(df: pd.DataFrame, filters: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for fk in filters:
        if fk not in SIMULATOR_FILTERS:
            return pd.Series(False, index=df.index)
        col, op, thr = SIMULATOR_FILTERS[fk]
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if op == ">":
            mask &= (s > thr)
        elif op == "<":
            mask &= (s < thr)
    return mask


LEVEL_NAMES = [
    "PDH", "PDL", "PD_50", "PWH", "PWL", "PW_50",
    "asia_H", "asia_L", "london_H", "london_L", "ny_H", "ny_L",
]
EVENT_TYPES = ["touch", "break", "reject", "failed_break", "near"]


def scan_edges(
    df: pd.DataFrame,
    horizons: list[int] = [3, 6, 12],
    symbol: str = "6B",
    confidence: str = "medium",
    run_permutation: bool = True,
    n_permutations: int = 1000,
    verbose: bool = True,
) -> dict:
    """
    البحث الكامل عن edges مع كل الحمايات.

    Pipeline:
      ① مسح train → جمع كل المرشّحات بـ p-values
      ② FDR (Benjamini-Hochberg) → تحكّم بـ false discovery rate
      ③ Permutation test → تأكيد إحصائي بدون افتراض توزيع
      ④ Validation set → فلترة (ضبط مسموح)
      ⑤ Holdout → اختبار نهائي (لمسة واحدة)
    """
    df = df.copy()
    if "ts_event" in df.columns:
        df = df.sort_values("ts_event").reset_index(drop=True)
    df["day"] = pd.to_datetime(df["ts_event"]).dt.date

    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    for h in horizons:
        col = f"fwd_ret_{h}"
        if col not in df.columns:
            arr = np.full(len(c), np.nan)
            if len(c) > h:
                arr[:-h] = (c[h:] - c[:-h]) / c[:-h]
            df[col] = arr

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

    df_train = df.iloc[split.train_idx].copy()
    df_val   = df.iloc[split.validation_idx].copy()
    df_hold  = df.iloc[split.holdout_idx].copy()

    if verbose:
        print(f"🔍 Edge Scanner V19.2 — symbol={symbol}, confidence={confidence}")
        print(f"   3-way split:")
        print(f"     train     : {split.n_train:,} ({split.n_train/len(df):.0%})")
        print(f"     validation: {split.n_validation:,} ({split.n_validation/len(df):.0%})")
        print(f"     holdout   : {split.n_holdout:,} ({split.n_holdout/len(df):.0%})")
        print(f"     purge     : {purge_bars} bars × 2")
        print(f"   transaction cost: {criteria['round_trip_cost']:.2f} pips/trade")
        print(f"   criteria: t_train≥{criteria['min_t_stat']}, "
              f"t_oos≥{criteria['min_t_oos']}, wr≥{criteria['min_wr']:.0%}, "
              f"min_avg≥{criteria['min_avg_pips']:.1f}p")

    zones = [z for z in df["zone_full"].unique() if z != "off_session"]
    all_candidates_stage1 = []
    cells_scanned = 0

    for zone in zones:
        zt = df_train[df_train["zone_full"] == zone]
        if len(zt) < 30:
            continue
        for level in LEVEL_NAMES:
            ev_col = f"{level}_event"
            if ev_col not in df.columns:
                continue
            for event in EVENT_TYPES:
                cell_tr = zt[zt[ev_col] == event]
                if len(cell_tr) < criteria["min_n"]:
                    continue
                cells_scanned += 1
                for combo_name, direction, filts in FILTER_COMBOS:
                    mask = _apply_combo(cell_tr, filts)
                    sub_tr = cell_tr[mask]
                    if len(sub_tr) < criteria["min_n"]:
                        continue
                    for h in horizons:
                        rets_tr = sub_tr[f"fwd_ret_{h}"].to_numpy()
                        days_tr = sub_tr["day"].to_numpy()
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

    # مرحلة 2: FDR
    p_values = np.array([c["p_value"] for c in all_candidates_stage1])
    rejected, fdr_threshold = benjamini_hochberg(p_values, criteria["fdr_alpha"])
    candidates_stage2 = [c for c, r in zip(all_candidates_stage1, rejected) if r]
    if verbose:
        print(f"   مرحلة 2 (FDR @ α={criteria['fdr_alpha']}): "
              f"{len(candidates_stage2)} اجتاز (p_threshold={fdr_threshold:.4f})")

    # مرحلة 3: Permutation
    candidates_stage3 = []
    if run_permutation and candidates_stage2:
        for c in candidates_stage2:
            zt = df_train[df_train["zone_full"] == c["zone"]]
            cell_tr = zt[zt[f"{c['level']}_event"] == c["event"]]
            mask = _apply_combo(cell_tr, c["filters"])
            sub_tr = cell_tr[mask]
            rets = sub_tr[f"fwd_ret_{c['horizon']}"].to_numpy()
            perm = permutation_test_edge(rets, c["direction"], n_permutations=n_permutations)
            c["permutation"] = perm
            if perm["p_value"] < 0.10:
                candidates_stage3.append(c)
        if verbose:
            print(f"   مرحلة 3 (Permutation): {len(candidates_stage3)} اجتاز")
    else:
        candidates_stage3 = candidates_stage2

    # مرحلة 4: Validation
    candidates_stage4 = []
    for c in candidates_stage3:
        zv = df_val[df_val["zone_full"] == c["zone"]]
        cell_val = zv[zv[f"{c['level']}_event"] == c["event"]]
        mask_val = _apply_combo(cell_val, c["filters"])
        sub_val = cell_val[mask_val]
        if len(sub_val) < criteria["min_n_test"]:
            continue
        rets_val = sub_val[f"fwd_ret_{c['horizon']}"].to_numpy()
        days_val = sub_val["day"].to_numpy()
        st_val = edge_statistics(rets_val, days_val, c["direction"])
        if not st_val.get("valid", False):
            continue
        if np.sign(st_val["avg_pips"]) != np.sign(c["stats_train"]["avg_pips"]):
            continue
        if abs(st_val["t_stat"]) < criteria["min_t_oos"]:
            continue
        c["stats_validation"] = st_val
        candidates_stage4.append(c)
    if verbose:
        print(f"   مرحلة 4 (Validation): {len(candidates_stage4)} اجتاز")

    # مرحلة 5: Holdout (الحكم النهائي)
    final_candidates = []
    for c in candidates_stage4:
        zh = df_hold[df_hold["zone_full"] == c["zone"]]
        cell_hold = zh[zh[f"{c['level']}_event"] == c["event"]]
        mask_hold = _apply_combo(cell_hold, c["filters"])
        sub_hold = cell_hold[mask_hold]
        if len(sub_hold) < criteria["min_n_test"]:
            continue
        rets_hold = sub_hold[f"fwd_ret_{c['horizon']}"].to_numpy()
        days_hold = sub_hold["day"].to_numpy()
        st_hold = edge_statistics(rets_hold, days_hold, c["direction"])
        if not st_hold.get("valid", False):
            continue
        validation = validate_edge_pipeline(
            c["stats_train"], c.get("stats_validation"), st_hold, criteria
        )
        c["stats_holdout"] = st_hold
        c["validation_report"] = validation
        if validation["passed"]:
            c["edge_id"] = (
                f"{c['zone']}__{c['level']}_{c['event']}__"
                f"{c['combo']}__h{c['horizon']}"
            )
            final_candidates.append(c)

    if verbose:
        print(f"   مرحلة 5 (Holdout): {len(final_candidates)} edge نهائي ✅")

    final_candidates.sort(
        key=lambda c: abs(c["stats_holdout"]["t_stat"]) * c["stats_holdout"]["wr"],
        reverse=True,
    )

    return {
        "candidates": final_candidates,
        "criteria": criteria,
        "diagnostics": {
            "cells_scanned": cells_scanned,
            "stage1_candidates":         len(all_candidates_stage1),
            "stage2_passed_fdr":         len(candidates_stage2),
            "stage3_passed_permutation": len(candidates_stage3),
            "stage4_passed_validation":  len(candidates_stage4),
            "stage5_passed_holdout":     len(final_candidates),
            "fdr_p_threshold":           float(fdr_threshold),
        },
        "market_spec": {
            "symbol": spec.symbol,
            "tick_size": spec.tick_size,
            "tick_value": spec.tick_value,
            "round_trip_cost_pips": spec.round_trip_cost_pips(),
        },
        "split_info": {
            "n_train":      split.n_train,
            "n_validation": split.n_validation,
            "n_holdout":    split.n_holdout,
            "purge_bars":   split.purge_bars,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Edge Scanner V19.2")
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", default="edge_candidates.json")
    ap.add_argument("--horizons", default="3,6,12")
    ap.add_argument("--symbol",  default="6B")
    ap.add_argument("--confidence", default="medium",
                    choices=["loose", "medium", "strict"])
    ap.add_argument("--no-permutation", action="store_true")
    ap.add_argument("--n-permutations", type=int, default=1000)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    print(f"📥 {len(df):,} rows | {len(df.columns)} cols")

    horizons = [int(x) for x in args.horizons.split(",")]
    result = scan_edges(
        df, horizons=horizons,
        symbol=args.symbol, confidence=args.confidence,
        run_permutation=not args.no_permutation,
        n_permutations=args.n_permutations,
    )

    def _clean(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_clean(x) for x in obj]
        return obj

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(_clean(result), f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 محفوظ: {args.output}")
    print(f"\n📊 Diagnostics:")
    for k, v in result["diagnostics"].items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
