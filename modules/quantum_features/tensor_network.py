"""
modules/quantum_features/tensor_network.py
──────────────────────────────────────────
Sprint 2 (التقرير 6.4): Full feature tensor (MPS-like representation).

Matrix Product States (MPS) في الكم:
    |ψ⟩ = Σ_{i₁,...,iₙ} A^{i₁}_α₁ A^{i₂}_{α₁α₂} ... A^{iₙ}_{αₙ₋₁} |i₁...iₙ⟩

النسخة الـ classical: outer-product بين feature groups
تكوّن عالي الأبعاد (high-order interactions structurally embedded).

استخدام للـ DL: يبني tensor (N, d₁, d₂, ..., dₖ) للـ CNN multi-head.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _minmax_per_col(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-column min-max normalization to [0, 1]."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        arr = arr[:, None]
    vmin = arr.min(axis=0, keepdims=True)
    vmax = arr.max(axis=0, keepdims=True)
    rng = vmax - vmin
    rng = np.where(rng < eps, 1.0, rng)
    return (arr - vmin) / rng


def build_outer_product_tensor(arrays: list[np.ndarray]) -> np.ndarray:
    """Outer product بين arrays.

    Each array: (N, kᵢ).
    Output: (N, k₁, k₂, ..., kₘ)
    """
    if not arrays:
        return np.zeros((0,), dtype=np.float64)
    tensor = np.asarray(arrays[0], dtype=np.float64)
    if tensor.ndim == 1:
        tensor = tensor[:, None]
    for arr in arrays[1:]:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        # tensor shape (N, k₁, k₂, ...); arr shape (N, kᵢ)
        # → (N, ..., 1) * (N, 1, ..., kᵢ)
        tensor = tensor[..., np.newaxis] * arr.reshape(arr.shape[0], *([1] * (tensor.ndim - 1)), arr.shape[1])
    return tensor


def build_feature_tensor(
    df: pd.DataFrame,
    feature_groups: list[list[str]],
    normalize: bool = True,
    skip_empty: bool = True,
) -> np.ndarray:
    """Builds MPS-like tensor من feature groups.

    Parameters
    ----------
    df : DataFrame
    feature_groups : list of column-name lists. Each group = "dimension".
    normalize : min-max normalize per column
    skip_empty : skip groups with no columns present in df

    Returns
    -------
    tensor : np.ndarray shape (N, k₁, k₂, ..., kₘ)
    """
    arrays = []
    for group in feature_groups:
        present = [c for c in group if c in df.columns]
        if not present:
            if skip_empty:
                continue
            raise KeyError(f"No columns found for group: {group}")
        arr = df[present].fillna(0.0).values.astype(np.float64)
        if normalize:
            arr = _minmax_per_col(arr)
        arrays.append(arr)
    if not arrays:
        return np.zeros((len(df), 0), dtype=np.float64)
    return build_outer_product_tensor(arrays)


def flatten_tensor(tensor: np.ndarray) -> np.ndarray:
    """Flatten (N, d₁, ..., dₖ) → (N, d₁*...*dₖ) for use as ML features."""
    arr = np.asarray(tensor, dtype=np.float64)
    if arr.ndim <= 1:
        return arr
    return arr.reshape(arr.shape[0], -1)


def mps_truncate(tensor: np.ndarray, bond_dim: int = 4) -> np.ndarray:
    """SVD-based truncation (MPS approximation).

    Reduces (N, d₁*...*dₖ) → (N, bond_dim) via SVD.
    Keeps the top bond_dim singular components.

    Parameters
    ----------
    tensor : (N, d₁, ..., dₖ) أو (N, M)
    bond_dim : عدد الـ singular values to keep
    """
    arr = np.asarray(tensor, dtype=np.float64)
    if arr.ndim < 2:
        return arr
    flat = arr if arr.ndim == 2 else flatten_tensor(arr)
    n_features = flat.shape[1]
    if n_features == 0:
        return np.zeros((flat.shape[0], 0), dtype=np.float64)
    k = min(bond_dim, min(flat.shape))
    if k == 0:
        return np.zeros((flat.shape[0], 0), dtype=np.float64)
    U, S, Vt = np.linalg.svd(flat, full_matrices=False)
    return U[:, :k] * S[:k]


def tensor_correlation(tensor: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Correlation بين كل tensor entry والـ target.

    Returns:
        shape (d₁, d₂, ..., dₖ) من correlations.
    """
    arr = np.asarray(tensor, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if arr.ndim < 2:
        return np.array([])
    flat = flatten_tensor(arr)
    target_centered = target - target.mean()
    target_std = target.std()
    if target_std < 1e-12:
        return np.zeros(flat.shape[1])
    cors = np.zeros(flat.shape[1])
    for i in range(flat.shape[1]):
        col = flat[:, i]
        col_std = col.std()
        if col_std < 1e-12:
            cors[i] = 0.0
            continue
        col_centered = col - col.mean()
        cors[i] = float(np.mean(col_centered * target_centered) / (col_std * target_std))
    rest_shape = arr.shape[1:]
    return cors.reshape(rest_shape)
