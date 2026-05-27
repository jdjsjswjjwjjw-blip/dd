"""
tools/audit_feature_redundancy.py
═══════════════════════════════════════════════════════════════════════════════
Empirical feature-redundancy audit — measures the "Feature Noise / Redundancy"
problem from docs/PIPELINE_ISSUES_AUDIT.md Issue #1.

The premise: if your model is given Imbalance_L1, Imbalance_L2, and
Book_Pressure simultaneously, you're forcing it to disentangle the same
underlying information three times. The model "succeeds" in training by
memorizing the high-dim correlation pattern, but in live trading any drift
breaks the memorization and predictions collapse.

This tool quantifies the problem on YOUR feature parquet:

  1. Compute the absolute Pearson correlation matrix of all numeric
     (non-leakage) feature columns
  2. Identify HIGHLY CORRELATED PAIRS  (|r| >= threshold, default 0.95)
  3. Cluster features into "redundancy groups" via connected-components
     of the correlation graph (one cluster = features that are all >=
     threshold correlated with at least one other member transitively)
  4. Recommend a REDUCED FEATURE SET — one representative per cluster
     (the feature with the highest mean absolute correlation to the
     target return, or just the first member if no target is given)
  5. Emit a CSV with the full pairwise correlation matrix, plus a JSON
     summary with the redundancy clusters and the recommended drop list

Usage:
  python tools/audit_feature_redundancy.py \\
      --features combined_6m/day_trading_features.parquet \\
      --output audit_results/redundancy \\
      --threshold 0.95 \\
      --target-col target_ret_24    # optional, used to pick the
                                    # best representative per cluster

Output:
  <output>/correlation_matrix.csv        full N×N matrix
  <output>/redundancy_summary.json       per-cluster summary + drop list
  <output>/redundancy_report.txt         human-readable summary
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# Mirror train_hybrid.py's leakage blocklist so we don't audit label-derived
# columns (they correlate with everything by construction).
LEAKAGE_PATTERNS = [
    "event_flag", "event_direction", "event_score", "is_event",
    "path_outcome", "bias_label", "label_confidence", "label_end_ts",
    "label_horizon_steps", "forward_return", "trade_duration",
    "soft_label", "neutral_reason", "event_label_tier",
    "train_event_flag", "is_train_slice", "is_holdout_slice",
    "is_purged_slice", "dataset_slice", "signal_quality",
    "target_ret_", "target_valid_",
]


def _is_leakage(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in LEAKAGE_PATTERNS)


def _select_audit_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that aren't leakage-prone."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if not _is_leakage(c)]


def compute_correlation_matrix(
    df: pd.DataFrame, columns: list[str],
) -> pd.DataFrame:
    """Compute |Pearson correlation| matrix. NaN-safe."""
    # Replace inf with NaN, then drop columns that are all-NaN
    sub = df[columns].replace([np.inf, -np.inf], np.nan)
    # Standardize so we can use simple matrix math (more stable than
    # pandas .corr() on very wide frames)
    valid_cols = [c for c in columns if sub[c].notna().sum() > 10]
    sub = sub[valid_cols].fillna(0.0)
    # Numerical guard for constant columns
    std = sub.std(axis=0)
    nonconst = std[std > 1e-9].index.tolist()
    sub = sub[nonconst]
    return sub.corr().abs()


def find_redundancy_clusters(
    corr: pd.DataFrame, threshold: float,
) -> list[list[str]]:
    """Union-Find on the correlation graph: two features in the same
    cluster if there's a PATH of >= threshold correlations between them."""
    cols = list(corr.columns)
    parent = {c: c for c in cols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    arr = corr.to_numpy()
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i, j] >= threshold:
                union(cols[i], cols[j])

    groups: dict[str, list[str]] = defaultdict(list)
    for c in cols:
        groups[find(c)].append(c)
    # Return only non-trivial clusters
    return [sorted(g) for g in groups.values() if len(g) > 1]


def pick_cluster_representative(
    cluster: list[str], df: pd.DataFrame, target_col: str | None,
) -> str:
    """Pick the most predictive feature in a cluster.

    If target_col is given, return the feature with the highest absolute
    Pearson correlation to that target. Otherwise return the first member
    (alphabetical sort already applied upstream).
    """
    if target_col is None or target_col not in df.columns:
        return cluster[0]
    target = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    best = cluster[0]
    best_corr = 0.0
    for c in cluster:
        col = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if col.std() <= 1e-9:
            continue
        r = abs(col.corr(target))
        if not np.isnan(r) and r > best_corr:
            best_corr = r
            best = c
    return best


def audit(
    features_path: Path,
    output_dir: Path,
    threshold: float = 0.95,
    target_col: str | None = None,
    sample_rows: int | None = None,
) -> dict:
    """Run the full redundancy audit. Returns a summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {features_path}")
    df = pd.read_parquet(features_path)
    if sample_rows and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=0).reset_index(drop=True)
        print(f"  Sampled {len(df):,} rows for speed (full file has more)")
    print(f"  {len(df):,} rows × {df.shape[1]} columns")

    audit_cols = _select_audit_columns(df)
    print(f"  Audit set: {len(audit_cols)} non-leakage numeric columns")

    print(f"\nComputing correlation matrix (this is the heavy step)...")
    corr = compute_correlation_matrix(df, audit_cols)
    print(f"  Matrix shape: {corr.shape}")

    # Save full matrix
    corr_path = output_dir / "correlation_matrix.csv"
    corr.to_csv(corr_path)
    print(f"  💾 {corr_path}")

    # Find clusters
    print(f"\nFinding redundancy clusters (threshold = {threshold})...")
    clusters = find_redundancy_clusters(corr, threshold)
    n_redundant_features = sum(len(c) for c in clusters)
    n_clusters = len(clusters)
    print(f"  {n_clusters} clusters, {n_redundant_features} redundant features")

    # Pick a representative for each cluster, drop the rest
    drop_list: list[str] = []
    cluster_summary = []
    for c in clusters:
        rep = pick_cluster_representative(c, df, target_col)
        drops = [x for x in c if x != rep]
        drop_list.extend(drops)
        cluster_summary.append({
            "size": len(c),
            "members": c,
            "representative": rep,
            "would_drop": drops,
        })

    # Compute a "redundancy score" — fraction of features that are
    # at-or-above threshold with at least one other
    redundant_ratio = n_redundant_features / max(len(corr.columns), 1)

    summary = {
        "features_file": str(features_path),
        "n_rows": int(len(df)),
        "n_columns_total": int(df.shape[1]),
        "n_columns_audited": int(len(audit_cols)),
        "n_columns_after_constancy_filter": int(len(corr.columns)),
        "threshold": threshold,
        "target_col": target_col,
        "n_redundancy_clusters": n_clusters,
        "n_redundant_features": n_redundant_features,
        "redundant_ratio": redundant_ratio,
        "n_to_drop": len(drop_list),
        "n_after_dedup": int(len(corr.columns)) - len(drop_list),
        "drop_list": sorted(drop_list),
        "clusters": cluster_summary,
    }
    json_path = output_dir / "redundancy_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  💾 {json_path}")

    # Human-readable report
    lines = []
    lines.append("═══════════════════════════════════════════════════════════════")
    lines.append(f"  FEATURE REDUNDANCY AUDIT  (threshold = {threshold})")
    lines.append("═══════════════════════════════════════════════════════════════")
    lines.append("")
    lines.append(f"  Features file:       {features_path}")
    lines.append(f"  Rows audited:        {len(df):,}")
    lines.append(f"  Audited columns:     {len(corr.columns)}")
    lines.append(f"  Redundancy clusters: {n_clusters}")
    lines.append(f"  Redundant features:  {n_redundant_features} "
                  f"({100 * redundant_ratio:.1f}%)")
    lines.append(f"  Would drop:          {len(drop_list)} → {len(corr.columns) - len(drop_list)} features after dedup")
    lines.append("")
    if clusters:
        lines.append("Top 10 largest clusters (representative ← members):")
        for c in cluster_summary[:10]:
            lines.append(f"")
            lines.append(f"  cluster size {c['size']}, keep '{c['representative']}', drop {len(c['would_drop'])}:")
            for m in c["members"]:
                marker = "←" if m == c["representative"] else " "
                lines.append(f"    {marker} {m}")
    else:
        lines.append("  (no clusters found at this threshold)")
    lines.append("")
    report_text = "\n".join(lines)
    report_path = output_dir / "redundancy_report.txt"
    report_path.write_text(report_text)
    print(f"  💾 {report_path}")
    print()
    print(report_text)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True,
                   help="Path to features parquet (day_trading_features.parquet)")
    p.add_argument("--output", required=True, help="Output directory for results")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Correlation threshold (0..1) above which two features "
                        "are considered redundant. Default 0.95.")
    p.add_argument("--target-col", default=None,
                   help="Optional target column for picking the best cluster "
                        "representative (e.g., target_ret_24).")
    p.add_argument("--sample-rows", type=int, default=None,
                   help="Subsample rows for speed (default: use all)")
    args = p.parse_args()
    audit(
        features_path=Path(args.features),
        output_dir=Path(args.output),
        threshold=args.threshold,
        target_col=args.target_col,
        sample_rows=args.sample_rows,
    )


if __name__ == "__main__":
    sys.exit(main())
