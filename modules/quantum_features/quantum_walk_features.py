"""
modules/quantum_features/quantum_walk_features.py
─────────────────────────────────────────────────
Sprint 2 (التقرير 6.4): Quantum walk على market topology.

Continuous-time quantum walk:
    |ψ(t)⟩ = e^{−iHt} |ψ(0)⟩

حيث H = adjacency matrix من graph.

في الـ market:
    - Nodes: market regimes / price levels / sessions
    - Edges: تشابه إحصائي (kNN)
    - Walk: يولد features تعكس topology

quantum-inspired (real-valued).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_topology_adjacency(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_neighbors: int = 4,
    max_nodes: int = 1000,
) -> np.ndarray | None:
    """Build kNN adjacency matrix من feature space.

    Parameters
    ----------
    df : DataFrame
    feature_cols : أعمدة numeric للـ distance. لو None → كل numeric.
    n_neighbors : عدد nearest neighbors
    max_nodes : لو len(df) > max_nodes نأخذ sample

    Returns
    -------
    adjacency : (n, n) symmetric matrix in {0, 1}. أو None لو لا توجد features.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
    if not feature_cols:
        return None

    X = df[feature_cols].fillna(0.0).values.astype(np.float64)
    n = X.shape[0]
    if n == 0:
        return None
    if n > max_nodes:
        idx = np.linspace(0, n - 1, max_nodes).astype(int)
        X = X[idx]
        n = X.shape[0]

    # Pair-wise Euclidean distance
    diff = X[:, None, :] - X[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # kNN adjacency
    adj = np.zeros((n, n))
    k = min(n_neighbors, n - 1)
    if k <= 0:
        return adj
    for i in range(n):
        order = np.argsort(dists[i])
        # skip self (order[0])
        nearest = order[1 : k + 1]
        adj[i, nearest] = 1.0
    return (adj + adj.T) / 2.0


def quantum_walk_step(adjacency: np.ndarray, state: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """One step of CT quantum walk (real-valued approximation).

    |ψ(t+1)⟩ = (I + γA) |ψ(t)⟩  followed by normalization.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    s = np.asarray(state, dtype=np.float64)
    new_state = s + gamma * (adj @ s)
    norm = np.linalg.norm(new_state)
    if norm > 1e-12:
        new_state = new_state / norm
    return new_state


def compute_quantum_walk_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_steps: int = 3,
    n_neighbors: int = 4,
    gamma: float = 0.5,
    max_nodes: int = 1000,
) -> dict[str, np.ndarray]:
    """Compute quantum-walk-based features.

    Returns
    -------
    dict {f"qwalk_step_{i}": amplitudes}.
    Length of each amplitude array == len(df) (broadcasted back if sampled).
    """
    adj = build_topology_adjacency(
        df, feature_cols=feature_cols, n_neighbors=n_neighbors, max_nodes=max_nodes
    )
    if adj is None or adj.size == 0:
        return {}
    n_nodes = adj.shape[0]
    # Initial state: uniform superposition
    state = np.full(n_nodes, 1.0 / np.sqrt(n_nodes))

    features: dict[str, np.ndarray] = {}
    for step in range(1, n_steps + 1):
        state = quantum_walk_step(adj, state, gamma=gamma)
        # Broadcast back to len(df) if sampled
        if n_nodes < len(df):
            full = np.repeat(state, len(df) // n_nodes + 1)[: len(df)]
        else:
            full = state[: len(df)] if n_nodes > len(df) else state
        features[f"qwalk_step_{step}"] = full
    return features


def add_quantum_walk_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_steps: int = 3,
    n_neighbors: int = 4,
    inplace: bool = False,
) -> pd.DataFrame:
    """يضيف qwalk features للـ DataFrame."""
    feats = compute_quantum_walk_features(
        df, feature_cols=feature_cols, n_steps=n_steps, n_neighbors=n_neighbors
    )
    result = df if inplace else df.copy()
    for k, v in feats.items():
        if len(v) == len(result):
            result[k] = v
    return result
