"""
modules/quantum_discovery/qaoa_optimizer.py
───────────────────────────────────────────
Sprint 3 (Phase C - التقرير 6.4): Quantum Approximate Optimization Algorithm.

QAOA يحل مشاكل combinatorial optimization (مثل MaxCut, portfolio selection).
هنا نطبق quantum-inspired classical analog:

    1. Cost function C(z): يحسب جودة الـ subset
    2. Mixer M(γ): يخلط الـ candidates
    3. Layer p مرات

أمثلة:
    - اختيار subset من alphas بأقل correlation وأعلى Sharpe
    - portfolio weights optimization مع constraints

التقرير: ⚛ quantum_dl/variational_quantum_eigen.py (VQE/QAOA family).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class QAOAConfig:
    """إعدادات QAOA."""

    p: int = 3  # عدد الـ layers
    n_starts: int = 5  # multiple starts (للـ classical search)
    max_iter: int = 100
    learning_rate: float = 0.1
    seed: int = 42


def _objective_max_sharpe(
    subset_mask: np.ndarray,
    sharpe: np.ndarray,
    corr_matrix: np.ndarray | None = None,
    corr_penalty: float = 0.5,
    max_size_penalty: float = 0.0,
    target_size: int = 10,
) -> float:
    """Default objective: maximize Σ sharpe[i] - penalty * Σ |corr[i,j]|."""
    if subset_mask.sum() == 0:
        return -1e9
    score = float(np.sum(sharpe[subset_mask]))
    if corr_matrix is not None:
        sub_corr = corr_matrix[np.ix_(subset_mask, subset_mask)]
        # Off-diagonal sum
        n = sub_corr.shape[0]
        if n > 1:
            off_diag = np.sum(np.abs(sub_corr)) - n  # diag = 1 each
            score -= corr_penalty * float(off_diag) / max(1, n * (n - 1))
    if max_size_penalty > 0:
        size_diff = abs(int(subset_mask.sum()) - target_size)
        score -= max_size_penalty * size_diff
    return float(score)


def qaoa_select_subset(
    sharpe: np.ndarray,
    corr_matrix: np.ndarray | None = None,
    target_size: int = 10,
    config: QAOAConfig | None = None,
    objective: Callable | None = None,
) -> tuple[np.ndarray, float, dict]:
    """QAOA-inspired subset selection.

    Parameters
    ----------
    sharpe : (N,) array من sharpe values
    corr_matrix : optional (N, N) correlation matrix
    target_size : عدد الـ subset elements المطلوب
    config : QAOAConfig
    objective : custom objective function (subset_mask, sharpe, corr) → float

    Returns
    -------
    best_subset_mask : boolean (N,)
    best_value : float
    diagnostics : dict
    """
    if config is None:
        config = QAOAConfig()
    if objective is None:
        objective = _objective_max_sharpe

    sharpe = np.asarray(sharpe, dtype=np.float64)
    N = sharpe.size
    if N == 0:
        return np.zeros(0, dtype=bool), 0.0, {"reason": "empty"}

    rng = np.random.RandomState(config.seed)
    best_subset = np.zeros(N, dtype=bool)
    best_value = -np.inf
    history = []

    # Multi-start "classical analog" of QAOA:
    # نختار random subsets ونحسّن iteratively باستخدام mixer-like moves
    for start in range(config.n_starts):
        # Initial: top target_size by sharpe + random perturbation
        order = np.argsort(-sharpe)
        subset = np.zeros(N, dtype=bool)
        subset[order[:target_size]] = True
        # perturb: flip a few random bits
        n_flips = max(1, target_size // 4)
        flip_idx = rng.choice(N, n_flips, replace=False)
        subset[flip_idx] = ~subset[flip_idx]

        value = objective(subset, sharpe, corr_matrix, target_size=target_size)

        # Local search (mixer-inspired)
        for _ in range(config.max_iter):
            improved = False
            # Try flipping each bit
            for i in range(N):
                new_subset = subset.copy()
                new_subset[i] = ~new_subset[i]
                new_value = objective(new_subset, sharpe, corr_matrix, target_size=target_size)
                if new_value > value:
                    subset = new_subset
                    value = new_value
                    improved = True
                    break
            if not improved:
                break

        if value > best_value:
            best_value = value
            best_subset = subset.copy()
        history.append(value)

    return best_subset, best_value, {
        "n_starts": config.n_starts,
        "history": history,
        "best_size": int(best_subset.sum()),
    }


def qaoa_portfolio_weights(
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 1.0,
    target_size: int = 10,
    config: QAOAConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """QAOA-inspired portfolio: maximize μᵀw - λ wᵀΣw subject to ‖w‖₀ ≤ target_size.

    Uses subset selection (binary) + closed-form Markowitz solution على الـ subset.
    """
    returns = np.asarray(returns, dtype=np.float64)
    cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
    N = returns.size

    # Pre-screen by Sharpe-like score
    diag = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))
    sharpe_proxy = returns / diag

    # Build correlation matrix for QAOA
    corr = cov_matrix / np.outer(diag, diag)

    best_subset, best_value, diag_info = qaoa_select_subset(
        sharpe_proxy,
        corr_matrix=corr,
        target_size=target_size,
        config=config,
    )

    # Hard-enforce target_size cap: take top-target_size by sharpe within best_subset
    weights = np.zeros(N)
    if best_subset.sum() == 0:
        return weights, {"reason": "empty_subset", **diag_info}
    sub_indices = np.where(best_subset)[0]
    if len(sub_indices) > target_size:
        # Keep the top target_size by sharpe_proxy
        ranked = sub_indices[np.argsort(-sharpe_proxy[sub_indices])][:target_size]
        sub_indices = np.sort(ranked)
    sub_idx = sub_indices
    sub_returns = returns[sub_idx]
    sub_cov = cov_matrix[np.ix_(sub_idx, sub_idx)]

    # w_sub = (Σ_sub)⁻¹ μ_sub / (risk_aversion * 2)
    try:
        inv_cov = np.linalg.pinv(sub_cov + 1e-8 * np.eye(len(sub_idx)))
        w_sub = inv_cov @ sub_returns / (2.0 * risk_aversion)
    except np.linalg.LinAlgError:
        w_sub = sub_returns / np.sum(np.abs(sub_returns) + 1e-12)

    # Normalize: ‖w‖₁ = 1
    w_sub = w_sub / max(np.sum(np.abs(w_sub)), 1e-12)
    weights[sub_idx] = w_sub

    return weights, {
        "subset_value": best_value,
        "subset_indices": sub_idx.tolist(),
        **diag_info,
    }
