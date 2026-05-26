"""
cluster_engine.py — المستوى ⑤ من نظام اكتشاف الـ Edge
══════════════════════════════════════════════════════════════════════

الدور: يأخذ edge_candidates ويُنتج alphas مميّزة غير مكررة.

المشكلة التي يحلها:
  edge_scanner قد يُخرج 50 edge، لكن كثيراً منها:
    - نفس الفكرة في خلايا متجاورة
    - مترابطة (تدخل نفس الصفقات تقريباً)
    - تكرار يضخّم الثقة زيفاً

ما يفعله:
  ① يجمّع الـ edges المتشابهة منطقياً (نفس combo/direction/level)
  ② يحسب الترابط الفعلي بين إشارات الدخول
  ③ يحذف المترابط (corr > عتبة) — يُبقي الأقوى
  ④ يُنتج alphas مميّزة + تقرير

المخرج: distinct_alphas.json

الاستخدام:
  python cluster_engine.py \\
    --candidates edge_candidates.json \\
    --data mapped.parquet \\
    --output distinct_alphas.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# عتبات التجميع
# ══════════════════════════════════════════════════════════════════

CLUSTER_CONFIG = {
    "corr_threshold":   0.70,   # فوقها = edges مكررة
    "min_overlap_jaccard": 0.50,  # تداخل صفقات فوقه = مكرر
}


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — التجميع المنطقي
# ══════════════════════════════════════════════════════════════════

def _logical_group_key(edge: dict) -> str:
    """
    مفتاح التجميع المنطقي.
    edges بنفس (direction + combo + level + event) = نفس الفكرة
    حتى لو في zones مختلفة.
    """
    return (
        f"{edge['entry']['direction']}__"
        f"{edge['entry']['combo']}__"
        f"{edge['cell']['level']}_{edge['cell']['event']}"
    )


def group_edges_logically(candidates: list[dict]) -> dict[str, list[dict]]:
    """يجمّع الـ edges حسب الفكرة المنطقية."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for edge in candidates:
        key = _logical_group_key(edge)
        groups[key].append(edge)
    return dict(groups)


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — حساب إشارات الدخول الفعلية
# ══════════════════════════════════════════════════════════════════

# نُعيد بناء فلاتر edge_scanner هنا للاستقلالية
SIMULATOR_FILTERS = {
    "depth_pressure_pos": ("sim_depth_pressure", ">",  0.20),
    "depth_pressure_neg": ("sim_depth_pressure", "<", -0.20),
    "informed_high":      ("sim_informed_prob",  ">",  0.50),
    "absorb_gate_open":   ("sim_absorb_intensity","<", 0.40),
    "flow_dir_up":        ("sim_flow_direction",  ">",  0.5),
    "flow_dir_down":      ("sim_flow_direction",  "<", -0.5),
    "depth_imb_pos":      ("sim_depth_imbalance", ">",  0.05),
    "depth_imb_neg":      ("sim_depth_imbalance", "<", -0.05),
    # الجدران العميقة
    "wall_growing":       ("sim_wall_growth",    ">",  0.10),
    "wall_shrinking":     ("sim_wall_growth",    "<", -0.10),
    "wall_consumed_high": ("sim_wall_consumed",  ">",  0.30),
    "wall_consumed_low":  ("sim_wall_consumed",  "<",  0.10),
    "wall_solid":         ("sim_wall_persist",   ">",  0.70),
    "wall_approaching":   ("sim_wall_shift",     ">",  0.15),
    "wall_retreating":    ("sim_wall_shift",     "<", -0.15),
    "bid_wall_near":      ("sim_wall_bid_level", "<",  3.0),
    "ask_wall_near":      ("sim_wall_ask_level", "<",  3.0),
    "bid_wall_big":       ("sim_wall_bid_size",  ">",  0.65),
    "ask_wall_big":       ("sim_wall_ask_size",  ">",  0.65),
    # Iceberg المؤسسي
    "iceberg_present":    ("sim_iceberg_prob",      ">",  0.60),
    "iceberg_strong":     ("sim_iceberg_prob",      ">",  0.75),
    "iceberg_buy":        ("sim_iceberg_side",      ">",  0.5),
    "iceberg_sell":       ("sim_iceberg_side",      "<", -0.5),
    "iceberg_replenish":  ("sim_iceberg_replenish", ">",  0.50),
    "iceberg_stealth":    ("sim_iceberg_stealth",   ">",  0.30),
    "iceberg_powerful":   ("sim_iceberg_strength",  ">",  0.60),
}


def _entry_signal(df: pd.DataFrame, edge: dict) -> pd.Series:
    """
    يبني سلسلة boolean: متى يُفعَّل هذا الـ edge.
    = (الخلية الصحيحة) AND (تركيبة الفلاتر).
    """
    cell = edge["cell"]
    ev_col = f"{cell['level']}_event"

    sig = pd.Series(True, index=df.index)
    # شرط الخلية
    if "zone_full" in df.columns:
        sig &= (df["zone_full"] == cell["zone"])
    if ev_col in df.columns:
        sig &= (df[ev_col] == cell["event"])

    # تركيبة الفلاتر
    for fk in edge["entry"]["filters"]:
        if fk not in SIMULATOR_FILTERS:
            continue
        col, op, thr = SIMULATOR_FILTERS[fk]
        if col not in df.columns:
            sig &= False
            continue
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        sig &= (s > thr) if op == ">" else (s < thr)

    return sig


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — قياس الترابط بين edges
# ══════════════════════════════════════════════════════════════════

def _signal_overlap(sig_a: pd.Series, sig_b: pd.Series) -> float:
    """
    Jaccard overlap بين إشارتي دخول.
    = |A ∩ B| / |A ∪ B|  ∈ [0,1]
    عالٍ = الـ edges تدخل نفس الصفقات.
    """
    a = sig_a.to_numpy(bool)
    b = sig_b.to_numpy(bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — اختيار الممثّل الأقوى لكل مجموعة
# ══════════════════════════════════════════════════════════════════

def _edge_score(edge: dict) -> float:
    """
    درجة قوة الـ edge للترتيب.
    يستخدم stats_train (إذا متاح) أو stats (للتوافق العكسي).
    """
    # دعم كلا الصيغتين: الجديدة (train/test) والقديمة
    s = edge.get("stats_train") or edge.get("stats")
    if not s:
        return 0.0
    
    score = (
        abs(s["t_stat"]) * 1.0 +
        (s["wr"] - 0.5) * 10.0 +
        s["avg_pips"] * 0.3 +
        (1.0 if s["stable"] else -2.0) +
        min(s["days"], 23) * 0.05
    )
    
    # مكافأة إضافية لو نجح على test (out-of-sample)
    st = edge.get("stats_test")
    if st and abs(st["t_stat"]) > 1.0:
        # نفس الإشارة على test = ثقة عالية
        score += min(abs(st["t_stat"]), 3.0) * 0.5
    
    return score


def select_distinct_alphas(
    candidates: list[dict],
    df: pd.DataFrame,
    corr_threshold: float = 0.70,
) -> tuple[list[dict], dict]:
    """
    العملية الكاملة:
      ① تجميع منطقي
      ② اختيار ممثّل لكل مجموعة (الأقوى)
      ③ حذف الممثّلين المترابطين فعلياً
    يرجع: (alphas مميّزة, تقرير)
    """
    if not candidates:
        return [], {"note": "لا مرشّحات"}

    # ① تجميع منطقي
    groups = group_edges_logically(candidates)

    # ② ممثّل لكل مجموعة = الأقوى
    representatives = []
    for key, edges in groups.items():
        edges_sorted = sorted(edges, key=_edge_score, reverse=True)
        best = edges_sorted[0]
        best["_group_key"] = key
        best["_group_size"] = len(edges)
        representatives.append(best)

    representatives.sort(key=_edge_score, reverse=True)

    # ③ حذف المترابط — greedy: نأخذ الأقوى، نحذف ما يترابط معه
    signals = {}
    for rep in representatives:
        signals[rep["edge_id"]] = _entry_signal(df, rep)

    kept = []
    dropped = []
    for rep in representatives:
        rep_id = rep["edge_id"]
        rep_sig = signals[rep_id]
        is_dup = False
        for k in kept:
            ov = _signal_overlap(rep_sig, signals[k["edge_id"]])
            if ov > corr_threshold:
                is_dup = True
                dropped.append({
                    "edge_id": rep_id,
                    "duplicate_of": k["edge_id"],
                    "overlap": round(ov, 3),
                })
                break
        if not is_dup:
            kept.append(rep)

    report = {
        "n_candidates": len(candidates),
        "n_logical_groups": len(groups),
        "n_representatives": len(representatives),
        "n_distinct_alphas": len(kept),
        "n_dropped_correlated": len(dropped),
        "dropped_detail": dropped,
    }

    return kept, report


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster Engine — Edge System Layer 5")
    ap.add_argument("--candidates", required=True, help="edge_candidates.json")
    ap.add_argument("--data",       required=True, help="mapped.parquet (لحساب الترابط)")
    ap.add_argument("--output",     default="distinct_alphas.json")
    ap.add_argument("--corr-threshold", type=float, default=0.70)
    args = ap.parse_args()

    with open(args.candidates, "r", encoding="utf-8") as f:
        cand_data = json.load(f)
    candidates = cand_data.get("candidates", [])
    print(f"📥 {len(candidates)} مرشّح edge")

    df = pd.read_parquet(args.data)
    print(f"📥 داتا: {len(df):,} rows")

    print("\n🧬 Cluster Engine — تجميع وإزالة التكرار...")
    alphas, report = select_distinct_alphas(candidates, df, args.corr_threshold)

    print(f"  ✅ {report['n_logical_groups']} مجموعة منطقية")
    print(f"  ✅ {report['n_distinct_alphas']} alpha مميّزة")
    print(f"  ✅ حُذف {report['n_dropped_correlated']} مترابط")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "report": report,
            "n_alphas": len(alphas),
            "alphas": alphas,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 محفوظ: {args.output}")
    if alphas:
        print(f"\n🏆 أقوى alphas مميّزة:")
        for a in alphas[:5]:
            s = a.get("stats_train") or a.get("stats")
            st = a.get("stats_test")
            print(f"   {a['edge_id']}")
            print(f"      TRAIN: n={s['n']} WR={s['wr']:.0%} "
                  f"t={s['t_stat']:.2f} (group of {a.get('_group_size',1)})")
            if st:
                print(f"      TEST:  n={st['n']} WR={st['wr']:.0%} "
                      f"t={st['t_stat']:.2f}")


if __name__ == "__main__":
    main()
