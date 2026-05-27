"""
tools/aggregate_walk_forward.py
═══════════════════════════════════════════════════════════════════════════════
Aggregate per-fold metrics from a walk-forward run into a single summary.

Reads <folds_dir>/fold_*/metrics.json and produces:
  • mean / std of hit_rate (rule baseline vs hybrid filtered)
  • per-fold lift (hybrid hit − baseline hit)
  • stability score (fraction of folds with positive lift)
  • aggregated trade count

Result written to a JSON file + printed as a readable table.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as stats
from pathlib import Path


def _load_fold(metrics_path: Path) -> dict | None:
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return None


def _fmt_pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:+.{digits}f}%" if v is not None else "  N/A"


def aggregate(folds_dir: Path) -> dict:
    fold_files = sorted(folds_dir.glob("fold_*/metrics.json"),
                        key=lambda p: int(re.search(r"fold_(\d+)", str(p)).group(1)))
    if not fold_files:
        return {"error": f"no fold_*/metrics.json found in {folds_dir}"}

    folds = []
    for fp in fold_files:
        m = _load_fold(fp)
        if m is not None:
            folds.append(m)

    rule_hits = []
    hyb_hits = []
    lifts = []
    rule_n = []
    hyb_n = []

    per_fold_table = []
    for f in folds:
        rb = f.get("rule_baseline", {}) or {}
        hf = f.get("hybrid_filtered", {}) or {}
        r_hit = rb.get("hit_rate")
        h_hit = hf.get("hit_rate")
        lift = f.get("lift_hit_rate")
        if r_hit is not None:
            rule_hits.append(r_hit)
            rule_n.append(rb.get("n_events", 0))
        if h_hit is not None:
            hyb_hits.append(h_hit)
            hyb_n.append(hf.get("n_events", 0))
        if lift is not None:
            lifts.append(lift)
        per_fold_table.append({
            "fold": f.get("fold_id", "?"),
            "train_period": f.get("train_period"),
            "test_period": f.get("test_period"),
            "rule_hit": r_hit,
            "rule_n": rb.get("n_events"),
            "hyb_hit": h_hit,
            "hyb_n": hf.get("n_events"),
            "lift": lift,
        })

    def _agg(xs: list[float]) -> dict:
        if not xs:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": float(stats.fmean(xs)),
            "std": float(stats.pstdev(xs)) if len(xs) > 1 else 0.0,
            "min": float(min(xs)),
            "max": float(max(xs)),
        }

    pos_lift = sum(1 for l in lifts if l is not None and l > 0)
    summary = {
        "n_folds": len(folds),
        "rule_baseline_hit_rate": _agg(rule_hits),
        "hybrid_filtered_hit_rate": _agg(hyb_hits),
        "lift_hit_rate": _agg(lifts),
        "stability_pct_positive_lift": pos_lift / max(len(lifts), 1),
        "total_rule_events": sum(rule_n),
        "total_hybrid_events": sum(hyb_n),
        "folds": per_fold_table,
    }
    return summary


def print_summary(s: dict):
    if "error" in s:
        print(f"❌ {s['error']}")
        return

    print("════════════════════════════════════════════════════════════════")
    print(f"  WALK-FORWARD SUMMARY  ({s['n_folds']} folds)")
    print("════════════════════════════════════════════════════════════════")
    print()
    print("Per-fold:")
    print(f"  {'fold':<4} {'train':<18} {'test':<18} "
          f"{'rule_hit':>10} {'rule_n':>7} "
          f"{'hyb_hit':>10} {'hyb_n':>7} {'lift':>8}")
    for f in s["folds"]:
        print(
            f"  {f['fold']!s:<4} "
            f"{(f['train_period'] or '?'):<18} "
            f"{(f['test_period'] or '?'):<18} "
            f"{_fmt_pct(f['rule_hit'])!s:>10} "
            f"{(f['rule_n'] or 0):>7} "
            f"{_fmt_pct(f['hyb_hit'])!s:>10} "
            f"{(f['hyb_n'] or 0):>7} "
            f"{_fmt_pct(f['lift'])!s:>8}"
        )
    print()
    print("Aggregate:")
    rb = s["rule_baseline_hit_rate"]
    hb = s["hybrid_filtered_hit_rate"]
    lf = s["lift_hit_rate"]
    if rb["mean"] is not None:
        print(f"  Rule baseline hit:   mean={_fmt_pct(rb['mean'], 2)}  "
              f"std={_fmt_pct(rb['std'], 2)}  "
              f"min={_fmt_pct(rb['min'])}  max={_fmt_pct(rb['max'])}")
    if hb["mean"] is not None:
        print(f"  Hybrid filt hit:     mean={_fmt_pct(hb['mean'], 2)}  "
              f"std={_fmt_pct(hb['std'], 2)}  "
              f"min={_fmt_pct(hb['min'])}  max={_fmt_pct(hb['max'])}")
    if lf["mean"] is not None:
        print(f"  Lift (hyb - rule):   mean={_fmt_pct(lf['mean'], 2)}  "
              f"std={_fmt_pct(lf['std'], 2)}  "
              f"min={_fmt_pct(lf['min'])}  max={_fmt_pct(lf['max'])}")
    print(f"  Folds with +lift:    "
          f"{int(s['stability_pct_positive_lift'] * s['n_folds'])}/{s['n_folds']}  "
          f"({_fmt_pct(s['stability_pct_positive_lift'], 0)})")
    print(f"  Total events:        rule={s['total_rule_events']}  "
          f"hybrid_filtered={s['total_hybrid_events']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    folds_dir = Path(args.folds_dir)
    s = aggregate(folds_dir)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(s, f, indent=2, default=float)
    print_summary(s)
    print()
    print(f"💾 written: {out_path}")


if __name__ == "__main__":
    main()
