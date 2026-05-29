"""
tools/ssl_ablation_comparison.py
═══════════════════════════════════════════════════════════════════════════════
Paired SSL ablation comparison — quantifies whether the SSL backbone
actually contributes to downstream trading performance.

Method
──────
Per fold, run the hybrid model TWICE under identical conditions:
  1. WITH SSL: normal run (SSL embeddings used)
  2. WITHOUT SSL: same run but `--ablate-ssl` zeros the embeddings

The model architecture (`ssl_embed_dim` etc.) stays identical between
the two runs — only the information content of the SSL channel
differs. The per-fold delta (with − without) is the SSL contribution.

Across folds, aggregate with:
  - mean delta and std delta
  - paired t-statistic on the deltas (one-sample t-test vs zero)
  - sign consistency (how many folds have positive delta)

Verdict cascade
───────────────
INSUFFICIENT_PAIRS  < min_pairs paired folds available
SSL_USEFUL          mean(delta_sharpe) > 0 AND |t-stat| ≥ 2.0
                    AND sign_consistency ≥ 0.70
SSL_NEUTRAL         |mean(delta_sharpe)| ≤ tolerance OR |t-stat| < 2.0
SSL_HARMFUL         mean(delta_sharpe) < 0 AND |t-stat| ≥ 2.0
                    (the SSL is actively confusing the model — refit)

Usage
─────
    # Run each fold twice
    for FOLD in 1 2 3 4 5 6 7 8; do
        python tools/run_walk_forward_fold_enhanced.py \\
            --fold-id $FOLD --output-dir runs/with_ssl/fold_$FOLD ...
        python tools/run_walk_forward_fold_enhanced.py \\
            --fold-id $FOLD --output-dir runs/no_ssl/fold_$FOLD \\
            --ablate-ssl ...
    done

    # Compare
    python tools/ssl_ablation_comparison.py \\
        --with-ssl-dir runs/with_ssl \\
        --no-ssl-dir   runs/no_ssl \\
        --output       runs/ssl_ablation

Outputs
───────
<output>/ssl_ablation_summary.json     per-fold deltas + verdict
<output>/ssl_ablation_report.txt       human-readable
<output>/per_fold_deltas.csv           one row per paired fold
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_PAIRS = 3                   # below this → INSUFFICIENT_PAIRS
T_STAT_THRESHOLD = 2.0          # rough 95 % CI cutoff
SIGN_CONSISTENCY_THRESHOLD = 0.70
NEUTRAL_TOLERANCE = 0.05        # |delta| ≤ this → neutral


# ══════════════════════════════════════════════════════════════════════════════
# Per-fold metric extraction
# ══════════════════════════════════════════════════════════════════════════════
def _extract_metric(d: dict, key: str, default: float = 0.0) -> float:
    """Try several common locations for a single scalar metric."""
    if not isinstance(d, dict):
        return float(default)
    if key in d and isinstance(d[key], (int, float)):
        return float(d[key])
    for nest in ("hybrid_filtered", "rule_baseline", "metrics", "holdout"):
        sub = d.get(nest)
        if isinstance(sub, dict) and key in sub and isinstance(
            sub[key], (int, float)
        ):
            return float(sub[key])
    return float(default)


def extract_per_fold_metrics(metrics: dict) -> dict[str, float]:
    """Pull the four metrics we'll compare across the with/without pair."""
    return {
        "hit_rate":   _extract_metric(metrics, "hit_rate"),
        "sharpe":     _extract_metric(metrics, "annualized_sharpe",
                                       _extract_metric(metrics, "sharpe")),
        "lift":       _extract_metric(metrics, "lift_hit_rate"),
        "max_drawdown": _extract_metric(metrics, "max_drawdown",
                                         _extract_metric(metrics, "max_drawdown_pct")),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pair discovery
# ══════════════════════════════════════════════════════════════════════════════
def _load_fold_metrics(folds_dir: Path) -> dict[str, dict]:
    """Return {fold_name: metrics_dict} for every subdirectory with metrics.json."""
    out: dict[str, dict] = {}
    for sub in sorted(folds_dir.iterdir() if folds_dir.exists() else []):
        if not sub.is_dir():
            continue
        m_path = sub / "metrics.json"
        if not m_path.exists():
            continue
        try:
            out[sub.name] = json.loads(m_path.read_text())
        except (ValueError, OSError):
            continue
    return out


def discover_pairs(
    with_ssl_dir: Path, no_ssl_dir: Path,
) -> list[tuple[str, dict, dict]]:
    """Return list of (fold_name, with_metrics, without_metrics) tuples.

    Only folds present in BOTH directories produce a pair.
    """
    with_folds = _load_fold_metrics(with_ssl_dir)
    no_folds = _load_fold_metrics(no_ssl_dir)
    pairs: list[tuple[str, dict, dict]] = []
    for name in sorted(set(with_folds) & set(no_folds)):
        pairs.append((name, with_folds[name], no_folds[name]))
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════════
def paired_t_statistic(deltas: list[float] | np.ndarray) -> float:
    """One-sample t-statistic on the deltas (null: mean = 0).

    Returns 0.0 when the input is too small or has zero variance —
    in either case the verdict cascade treats this as 'no evidence'."""
    arr = np.asarray(deltas, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std <= 1e-12:
        return 0.0
    return mean / (std / math.sqrt(n))


def sign_consistency(deltas: list[float] | np.ndarray) -> float:
    """Fraction of deltas with the same sign as the mean."""
    arr = np.asarray(deltas, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    mean_sign = 1.0 if float(np.mean(arr)) >= 0 else -1.0
    matching = np.sum(np.sign(arr) == mean_sign)
    return float(matching) / float(len(arr))


# ══════════════════════════════════════════════════════════════════════════════
# Verdict cascade
# ══════════════════════════════════════════════════════════════════════════════
def compute_verdict(
    deltas_sharpe: list[float],
    min_pairs: int = MIN_PAIRS,
    t_threshold: float = T_STAT_THRESHOLD,
    sign_threshold: float = SIGN_CONSISTENCY_THRESHOLD,
    neutral_tolerance: float = NEUTRAL_TOLERANCE,
) -> tuple[str, dict]:
    """Pick a verdict from the per-fold delta_sharpe list."""
    diag: dict = {
        "n_pairs": int(len(deltas_sharpe)),
        "t_threshold": float(t_threshold),
        "sign_threshold": float(sign_threshold),
    }
    if len(deltas_sharpe) < min_pairs:
        diag["min_pairs"] = int(min_pairs)
        return "INSUFFICIENT_PAIRS", diag

    arr = np.asarray(deltas_sharpe, dtype=np.float64)
    mean_delta = float(np.mean(arr))
    t = paired_t_statistic(arr)
    sign_c = sign_consistency(arr)
    diag.update({
        "mean_delta_sharpe": mean_delta,
        "t_statistic": float(t),
        "sign_consistency": float(sign_c),
    })

    if abs(mean_delta) <= neutral_tolerance or abs(t) < t_threshold:
        return "SSL_NEUTRAL", diag
    if mean_delta > 0 and sign_c >= sign_threshold:
        return "SSL_USEFUL", diag
    if mean_delta < 0:
        return "SSL_HARMFUL", diag
    return "SSL_NEUTRAL", diag


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration + I/O
# ══════════════════════════════════════════════════════════════════════════════
def build_per_fold_table(
    pairs: list[tuple[str, dict, dict]],
) -> pd.DataFrame:
    """One row per paired fold with the deltas."""
    rows: list[dict] = []
    for name, m_with, m_without in pairs:
        e_with = extract_per_fold_metrics(m_with)
        e_without = extract_per_fold_metrics(m_without)
        rows.append({
            "fold": name,
            "sharpe_with":    e_with["sharpe"],
            "sharpe_without": e_without["sharpe"],
            "delta_sharpe":   e_with["sharpe"] - e_without["sharpe"],
            "hit_with":       e_with["hit_rate"],
            "hit_without":    e_without["hit_rate"],
            "delta_hit":      e_with["hit_rate"] - e_without["hit_rate"],
            "lift_with":      e_with["lift"],
            "lift_without":   e_without["lift"],
            "delta_lift":     e_with["lift"] - e_without["lift"],
            "dd_with":        e_with["max_drawdown"],
            "dd_without":     e_without["max_drawdown"],
            "delta_dd":       e_with["max_drawdown"] - e_without["max_drawdown"],
        })
    return pd.DataFrame(rows)


def write_report(
    path: Path, verdict: str, diag: dict, table: pd.DataFrame,
) -> None:
    lines = ["═" * 70, "SSL ABLATION COMPARISON", "═" * 70,
             f"Verdict        : {verdict}",
             f"N paired folds : {diag.get('n_pairs', 0)}",
             ""]
    if "mean_delta_sharpe" in diag:
        lines.extend([
            f"Mean Δ sharpe  : {diag['mean_delta_sharpe']:+.4f}",
            f"t-statistic    : {diag['t_statistic']:+.3f}  "
            f"(threshold ±{diag['t_threshold']:.1f})",
            f"Sign consistency: {diag['sign_consistency']:.1%}  "
            f"(threshold ≥ {diag['sign_threshold']:.0%})",
            "",
        ])

    if not table.empty:
        lines.append("Per-fold deltas:")
        lines.append("  fold              "
                     "Δ_sharpe   Δ_hit    Δ_lift   Δ_dd")
        for _, r in table.iterrows():
            lines.append(
                f"  {str(r['fold']):<16}  "
                f"{r['delta_sharpe']:+.4f}   "
                f"{r['delta_hit']:+.4f}  "
                f"{r['delta_lift']:+.4f}  "
                f"{r['delta_dd']:+.4f}"
            )
        lines.append("")

    if verdict == "SSL_USEFUL":
        lines.append("→ SSL adds measurable, consistent value. Keep training.")
    elif verdict == "SSL_NEUTRAL":
        lines.append("→ No evidence SSL adds value. Consider Head Freezing or")
        lines.append("  Task Weight Pruning before deploying.")
    elif verdict == "SSL_HARMFUL":
        lines.append("→ SSL actively hurts. Gradient Orthogonalization or refit.")
    else:
        lines.append("→ Not enough paired folds to draw a conclusion.")

    lines.append("═" * 70)
    path.write_text("\n".join(lines) + "\n")


def run_comparison(
    with_ssl_dir: Path,
    no_ssl_dir: Path,
    output_dir: Path,
    min_pairs: int = MIN_PAIRS,
    neutral_tolerance: float = NEUTRAL_TOLERANCE,
) -> dict:
    pairs = discover_pairs(with_ssl_dir, no_ssl_dir)
    table = build_per_fold_table(pairs)
    deltas = table["delta_sharpe"].tolist() if not table.empty else []
    verdict, diag = compute_verdict(
        deltas, min_pairs=min_pairs, neutral_tolerance=neutral_tolerance,
    )

    summary = {
        "verdict": verdict,
        "diag": diag,
        "with_ssl_dir": str(with_ssl_dir),
        "no_ssl_dir": str(no_ssl_dir),
        "per_fold": table.to_dict(orient="records"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ssl_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    if not table.empty:
        table.to_csv(output_dir / "per_fold_deltas.csv", index=False)
    write_report(
        output_dir / "ssl_ablation_report.txt",
        verdict=verdict, diag=diag, table=table,
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--with-ssl-dir", required=True, type=Path)
    p.add_argument("--no-ssl-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    p.add_argument("--neutral-tolerance", type=float, default=NEUTRAL_TOLERANCE)
    args = p.parse_args()

    summary = run_comparison(
        args.with_ssl_dir, args.no_ssl_dir, args.output,
        min_pairs=args.min_pairs,
        neutral_tolerance=args.neutral_tolerance,
    )
    diag = summary["diag"]
    print(f"verdict           : {summary['verdict']}")
    print(f"paired folds      : {diag.get('n_pairs', 0)}")
    if "mean_delta_sharpe" in diag:
        print(f"mean Δ sharpe     : {diag['mean_delta_sharpe']:+.4f}")
        print(f"t-statistic       : {diag['t_statistic']:+.3f}")
        print(f"sign consistency  : {diag['sign_consistency']:.1%}")
    print(f"output            : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
