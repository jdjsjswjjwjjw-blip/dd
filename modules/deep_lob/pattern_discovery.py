"""
modules/deep_lob/pattern_discovery.py
──────────────────────────────────────
Discover new patterns from a trained Hierarchical LOB Transformer.

3 methods:
    1. Attention Weight Analysis — which orders matter for predictions?
    2. Embedding Cluster Discovery — group similar bars in embedding space
    3. Counter-factual Analysis — which orders changed the prediction?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class DiscoveredPattern:
    """A pattern discovered from the model."""

    pattern_id: str
    method: str                    # 'attention' | 'cluster' | 'counter_factual'
    score: float                   # importance score
    n_examples: int
    centroid_embedding: Optional[np.ndarray] = None
    examples: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "method": self.method,
            "score": float(self.score),
            "n_examples": int(self.n_examples),
            "metadata": dict(self.metadata),
            "examples": list(self.examples),
        }


# ════════════════════════════════════════════════════════════════════════════
# Method 1: Attention-based discovery
# ════════════════════════════════════════════════════════════════════════════


def discover_critical_orders_from_attention(
    attentions: list[torch.Tensor],
    order_features: torch.Tensor,
    order_masks: torch.Tensor,
    top_k: int = 10,
) -> list[dict]:
    """For each bar in batch, find the K most-attended orders.

    These are the orders that the model "found important".

    Parameters
    ----------
    attentions : list of (B, N, N) tensors — one per transformer layer
    order_features : (B, N, 7)
    order_masks : (B, N) bool
    top_k : how many critical orders per bar

    Returns
    -------
    list of dicts (one per bar in batch):
        {
            "critical_indices": list[int],
            "attention_scores": list[float],
            "features": list[7-tuple],
        }
    """
    if not attentions:
        return []

    # Average attention across layers
    avg_attn = torch.stack(attentions).mean(dim=0)  # (B, N, N)

    # Sum incoming attention per order (column-wise)
    incoming_attention = avg_attn.sum(dim=1)  # (B, N) — how much each order is attended to

    B, N = incoming_attention.shape
    results = []

    for b in range(B):
        # Mask invalid
        scores = incoming_attention[b].clone()
        scores[~order_masks[b]] = -1.0

        # Top-K
        k = min(top_k, int(order_masks[b].sum()))
        top_scores, top_idx = scores.topk(k)

        bar_result = {
            "critical_indices": top_idx.tolist(),
            "attention_scores": top_scores.tolist(),
            "features": order_features[b, top_idx].cpu().numpy().tolist(),
        }
        results.append(bar_result)

    return results


def discover_event_assignments(
    event_assignment: torch.Tensor,
    order_features: torch.Tensor,
    order_masks: torch.Tensor,
    top_orders_per_event: int = 5,
) -> list[dict]:
    """For each event type, identify dominant order patterns.

    Parameters
    ----------
    event_assignment : (B, E, N) — soft assignment matrix
    order_features : (B, N, 7)
    order_masks : (B, N)

    Returns
    -------
    list of dicts per bar:
        {
            "event_id": int,
            "dominant_orders": list[indices],
            "weights": list[float],
        }
    """
    B, E, N = event_assignment.shape
    results = []

    for b in range(B):
        bar_result = {"event_assignments": []}
        for e in range(E):
            scores = event_assignment[b, e].clone()
            scores[~order_masks[b]] = 0.0
            k = min(top_orders_per_event, int(order_masks[b].sum()))
            top_scores, top_idx = scores.topk(k)

            bar_result["event_assignments"].append({
                "event_id": e,
                "dominant_orders": top_idx.tolist(),
                "weights": top_scores.tolist(),
                "features": order_features[b, top_idx].cpu().numpy().tolist(),
            })
        results.append(bar_result)

    return results


# ════════════════════════════════════════════════════════════════════════════
# Method 2: Embedding cluster discovery
# ════════════════════════════════════════════════════════════════════════════


def discover_clusters_from_embeddings(
    embeddings: np.ndarray,
    outcomes: np.ndarray,
    min_cluster_size: int = 20,
    method: str = "kmeans",
    n_clusters: int = 20,
) -> list[DiscoveredPattern]:
    """Cluster bars by embedding + analyze outcome stats per cluster.

    Clusters with high win-rate / high Sharpe = discovered patterns.

    Parameters
    ----------
    embeddings : (N, D) — shared embeddings from model
    outcomes : (N,) — returns or labels
    min_cluster_size : minimum bars to keep a cluster
    method : 'kmeans' | 'dbscan' | 'hdbscan'
    n_clusters : for kmeans

    Returns
    -------
    list of DiscoveredPattern
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    if method == "kmeans":
        labels = _simple_kmeans(embeddings, n_clusters)
    elif method == "dbscan":
        try:
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=0.5, min_samples=min_cluster_size).fit_predict(embeddings)
        except ImportError:
            labels = _simple_kmeans(embeddings, n_clusters)
    else:
        labels = _simple_kmeans(embeddings, n_clusters)

    patterns = []
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # DBSCAN noise
            continue
        mask = labels == cluster_id
        if mask.sum() < min_cluster_size:
            continue
        cluster_outcomes = outcomes[mask]
        wr = float((cluster_outcomes > 0).mean())
        mean_outcome = float(cluster_outcomes.mean())
        std_outcome = float(cluster_outcomes.std())
        sharpe = mean_outcome / std_outcome if std_outcome > 1e-9 else 0.0

        centroid = embeddings[mask].mean(axis=0)
        score = abs(sharpe) + abs(wr - 0.5)

        patterns.append(DiscoveredPattern(
            pattern_id=f"cluster_{int(cluster_id)}",
            method="cluster",
            score=score,
            n_examples=int(mask.sum()),
            centroid_embedding=centroid,
            metadata={
                "win_rate": wr,
                "mean_outcome": mean_outcome,
                "std_outcome": std_outcome,
                "sharpe": sharpe,
                "cluster_id": int(cluster_id),
            },
        ))

    patterns.sort(key=lambda p: p.score, reverse=True)
    return patterns


def _simple_kmeans(
    X: np.ndarray, k: int, max_iter: int = 100, seed: int = 42,
) -> np.ndarray:
    """Lloyd's K-means (no sklearn dependency)."""
    rng = np.random.RandomState(seed)
    n, d = X.shape
    if n == 0 or k == 0:
        return np.zeros(n, dtype=int)
    k = min(k, n)
    idx = rng.choice(n, k, replace=False)
    centers = X[idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        diff = X[:, None, :] - centers[None, :, :]
        dists = np.sum(diff ** 2, axis=-1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = X[mask].mean(axis=0)
    return labels


# ════════════════════════════════════════════════════════════════════════════
# Method 3: Counter-factual analysis
# ════════════════════════════════════════════════════════════════════════════


def counter_factual_importance(
    model,
    order_features: torch.Tensor,
    order_masks: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    target_output: str = "direction_logits",
) -> torch.Tensor:
    """For each order, measure how much removing it changes the prediction.

    Parameters
    ----------
    model : HierarchicalLOBTransformer
    order_features : (B, T, N, 7)
    order_masks : (B, T, N)
    context : (B, n_context) optional
    target_output : which output to measure importance for

    Returns
    -------
    importance : (B, T, N) — change in output norm if each order is removed
    """
    model.eval()
    B, T, N, _ = order_features.shape

    with torch.no_grad():
        # Baseline prediction
        out_base = model(order_features, order_masks, context=context)
        base_tensor = getattr(out_base, target_output)
        if base_tensor.ndim == 1:
            base_tensor = base_tensor.unsqueeze(-1)

        importances = torch.zeros(B, T, N, device=order_features.device)

        for t in range(T):
            for i in range(N):
                # Skip already-masked orders
                if not order_masks[:, t, i].any():
                    continue

                # Create counterfactual: remove order i in time t
                cf_masks = order_masks.clone()
                cf_masks[:, t, i] = False

                out_cf = model(order_features, cf_masks, context=context)
                cf_tensor = getattr(out_cf, target_output)
                if cf_tensor.ndim == 1:
                    cf_tensor = cf_tensor.unsqueeze(-1)

                # L2 distance between base and counterfactual
                diff = (base_tensor - cf_tensor).norm(dim=-1)  # (B,)
                importances[:, t, i] = diff

    return importances
