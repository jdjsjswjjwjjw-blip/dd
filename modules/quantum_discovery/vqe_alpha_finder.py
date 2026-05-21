"""
modules/quantum_discovery/vqe_alpha_finder.py
─────────────────────────────────────────────
Sprint 3 (Phase C - التقرير 6.4): Variational Quantum Eigensolver.

VQE في الكم: يبحث عن lowest eigenvalue (ground state) لـ Hamiltonian H.
هنا نطبق الـ classical analog:

    H = correlation matrix أو covariance من alpha returns
    eigenvectors = "principal alpha modes"

    eigenvalue الأصغر (negative) = الـ direction الأعلى negative correlation
    → potential alpha (uncorrelated to market modes).

VQE-inspired flow:
    1. Build H من historical alpha returns
    2. Find smallest eigenvalue via power iteration (cheap)
    3. Map eigenvector to alpha selection (top-k by |weight|)

التقرير: ⚛ quantum_dl/variational_quantum_eigen.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VQEConfig:
    """إعدادات VQE."""

    max_iter: int = 200
    tolerance: float = 1e-6
    seed: int = 42
    top_k: int = 10
    # If True, use smallest eigenvalue (most "independent" alpha mode);
    # else largest (dominant mode).
    smallest: bool = True


def build_hamiltonian(returns: np.ndarray) -> np.ndarray:
    """Hamiltonian = correlation matrix من returns.

    Parameters
    ----------
    returns : (T, N) array من historical alpha returns

    Returns
    -------
    H : (N, N) correlation matrix
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    # Standardize columns
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z = (arr - mu) / sd
    T = max(arr.shape[0], 1)
    return (z.T @ z) / T


def power_iteration(
    matrix: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Largest eigenvalue + eigenvector via power iteration."""
    arr = np.asarray(matrix, dtype=np.float64)
    n = arr.shape[0]
    rng = np.random.RandomState(seed)
    v = rng.randn(n)
    v = v / max(np.linalg.norm(v), 1e-12)
    lam_prev = 0.0
    for it in range(max_iter):
        Av = arr @ v
        norm = np.linalg.norm(Av)
        if norm < 1e-12:
            return 0.0, v
        v_new = Av / norm
        lam = float(v_new.T @ arr @ v_new)
        if abs(lam - lam_prev) < tol:
            return lam, v_new
        v = v_new
        lam_prev = lam
    return lam_prev, v


def inverse_power_iteration(
    matrix: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
    shift: float = 1e-3,
) -> tuple[float, np.ndarray]:
    """Smallest eigenvalue via inverse power iteration.

    Solves (A - σI) v_{k+1} = v_k iteratively.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    n = arr.shape[0]
    rng = np.random.RandomState(seed)
    v = rng.randn(n)
    v = v / max(np.linalg.norm(v), 1e-12)
    eye = np.eye(n)
    shifted = arr - shift * eye
    try:
        inv = np.linalg.pinv(shifted)
    except np.linalg.LinAlgError:
        return 0.0, v
    lam_prev = 0.0
    for it in range(max_iter):
        w = inv @ v
        norm = np.linalg.norm(w)
        if norm < 1e-12:
            break
        v_new = w / norm
        lam = float(v_new.T @ arr @ v_new)
        if abs(lam - lam_prev) < tol:
            return lam, v_new
        v = v_new
        lam_prev = lam
    return lam_prev, v


def vqe_find_alpha_mode(
    returns: np.ndarray,
    config: VQEConfig | None = None,
) -> dict:
    """يجد "alpha mode" via VQE-inspired eigenvalue search.

    Parameters
    ----------
    returns : (T, N) — T time steps, N alphas

    Returns
    -------
    dict مع:
        eigenvalue : float
        eigenvector : (N,) normalized
        top_k_indices : array من alphas الأهم
        top_k_weights : array من |weights|
        hamiltonian : (N, N)
    """
    if config is None:
        config = VQEConfig()

    H = build_hamiltonian(returns)
    if config.smallest:
        lam, v = inverse_power_iteration(
            H, max_iter=config.max_iter, tol=config.tolerance, seed=config.seed,
        )
    else:
        lam, v = power_iteration(
            H, max_iter=config.max_iter, tol=config.tolerance, seed=config.seed,
        )

    # Top-k by |weight|
    abs_v = np.abs(v)
    order = np.argsort(-abs_v)
    top_k = config.top_k if config.top_k > 0 else len(v)
    top_idx = order[:top_k]

    return {
        "eigenvalue": float(lam),
        "eigenvector": v,
        "top_k_indices": top_idx,
        "top_k_weights": v[top_idx],
        "hamiltonian_shape": H.shape,
    }
