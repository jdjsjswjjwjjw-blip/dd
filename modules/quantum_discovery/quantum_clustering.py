"""
modules/quantum_discovery/quantum_clustering.py
───────────────────────────────────────────────
Sprint 3 (Phase C - التقرير 6.4): Quantum clustering (بدل K-means).

K-means عشوائي → quantum-inspired clustering:
    1. Build similarity matrix S (kernel-based)
    2. Spectral decomposition (eigenvectors of Laplacian)
    3. Cluster على الـ embedding بـ k-means أو tight grouping

الـ "quantum" aspect:
    - Eigenvectors هم quantum modes
    - Each cluster = ground state of a particular Hamiltonian
    - Soft membership = quantum superposition

التقرير: ⚛ quantum_discovery/quantum_clustering.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuantumClusteringConfig:
    """إعدادات الـ clustering."""

    n_clusters: int = 5
    n_eigenvectors: int = 5
    similarity_metric: str = "gaussian"  # 'gaussian' | 'cosine' | 'dot'
    gaussian_sigma: float | None = None  # None → auto
    max_iter: int = 100
    seed: int = 42


def _similarity_matrix(
    X: np.ndarray, metric: str = "gaussian", sigma: float | None = None
) -> np.ndarray:
    """Compute similarity matrix S."""
    arr = np.asarray(X, dtype=np.float64)
    n = arr.shape[0]
    if metric == "gaussian":
        # Euclidean distance squared
        diff = arr[:, None, :] - arr[None, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        if sigma is None:
            # auto sigma = median of distances
            sigma = float(np.sqrt(np.median(dists_sq[dists_sq > 0]))) if (dists_sq > 0).any() else 1.0
        return np.exp(-dists_sq / (2.0 * max(sigma ** 2, 1e-12)))
    if metric == "cosine":
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        safe = np.where(norms < 1e-12, 1.0, norms)
        normed = arr / safe
        return (normed @ normed.T + 1.0) / 2.0  # ∈ [0, 1]
    if metric == "dot":
        s = arr @ arr.T
        s_min = s.min()
        s_max = s.max()
        rng = s_max - s_min
        if rng < 1e-12:
            return np.ones_like(s)
        return (s - s_min) / rng
    raise ValueError(f"Unknown metric: {metric}")


def _laplacian(S: np.ndarray) -> np.ndarray:
    """Normalized Laplacian: L = I - D^(-1/2) S D^(-1/2)."""
    d = S.sum(axis=1)
    d_inv_sqrt = np.where(d > 1e-12, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(S.shape[0]) - D_inv_sqrt @ S @ D_inv_sqrt
    # Symmetrize
    return (L + L.T) / 2.0


def _spectral_embedding(L: np.ndarray, n_components: int) -> np.ndarray:
    """First n_components eigenvectors of L (excluding the trivial constant)."""
    n = L.shape[0]
    # Symmetric → use eigh
    w, V = np.linalg.eigh(L)
    # Sort ascending, skip the first (≈ 0)
    order = np.argsort(w)
    start = 1
    end = min(start + n_components, n)
    return V[:, order[start:end]]


def _kmeans_simple(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 42) -> np.ndarray:
    """Lloyd's algorithm — simple K-means."""
    rng = np.random.RandomState(seed)
    n, d = X.shape
    if n == 0 or k == 0:
        return np.zeros(n, dtype=int)
    k = min(k, n)
    idx = rng.choice(n, k, replace=False)
    centers = X[idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        diff = X[:, None, :] - centers[None, :, :]
        dists = np.sum(diff ** 2, axis=-1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centers
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = X[mask].mean(axis=0)
    return labels


def quantum_cluster(
    X: np.ndarray, config: QuantumClusteringConfig | None = None,
) -> dict:
    """Spectral (quantum-inspired) clustering.

    Returns
    -------
    dict مع:
        labels : (N,) cluster id (0..k-1)
        embedding : (N, n_eigenvectors)
        eigenvalues : (n_eigenvectors,)
    """
    if config is None:
        config = QuantumClusteringConfig()

    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    n = arr.shape[0]
    if n == 0:
        return {
            "labels": np.zeros(0, dtype=int),
            "embedding": np.zeros((0, config.n_eigenvectors)),
            "eigenvalues": np.zeros(config.n_eigenvectors),
        }

    S = _similarity_matrix(arr, metric=config.similarity_metric, sigma=config.gaussian_sigma)
    L = _laplacian(S)
    emb = _spectral_embedding(L, config.n_eigenvectors)
    # Normalize rows of embedding for stability
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    emb_norm = emb / norms

    labels = _kmeans_simple(emb_norm, config.n_clusters, max_iter=config.max_iter, seed=config.seed)

    # Eigenvalues of L (for diagnostics)
    eigvals = np.sort(np.linalg.eigvalsh(L))[: config.n_eigenvectors + 1]

    return {
        "labels": labels,
        "embedding": emb_norm,
        "eigenvalues": eigvals,
        "n_clusters_found": int(np.unique(labels).size),
    }


def cluster_alphas_by_returns(
    returns: np.ndarray,
    n_clusters: int = 5,
    config: QuantumClusteringConfig | None = None,
) -> dict:
    """Cluster N alphas حسب returns correlation.

    Parameters
    ----------
    returns : (T, N) historical returns
    n_clusters : هدف K

    Returns
    -------
    dict (مثل quantum_cluster) مع labels.shape = (N,)
    """
    if config is None:
        config = QuantumClusteringConfig(n_clusters=n_clusters)
    else:
        config.n_clusters = n_clusters

    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("returns must be 2D (T, N)")
    # transpose: each alpha = row
    return quantum_cluster(arr.T, config=config)
