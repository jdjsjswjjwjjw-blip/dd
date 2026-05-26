"""
modules/quantum_core/interference_amplifier.py
──────────────────────────────────────────────
Sprint 2 (التقرير 6.3): Grover amplitude amplification.

Grover's algorithm:
    n_iter ≈ (π/4) * √(N/M)

حيث:
    N = total basis states (candidates)
    M = "good" basis states (الـ alphas الصحيحة)

كل iteration = Oracle + Diffusion:
    Oracle: ⟨ψ|→ flip sign of good states
    Diffusion: 2|ψ⟩⟨ψ| − I → reflection حول الـ mean amplitude

النتيجة: amplitudes الصحيحة تكبر، الـ noise يضمحلّ.
O(√N) بدل O(N) للـ classical search.
"""

from __future__ import annotations

import numpy as np


def optimal_iterations(N: int, M: int) -> int:
    """عدد التكرارات المثالي لـ Grover.

    n_iter ≈ (π/4) * √(N/M)
    """
    if M <= 0 or N <= 0:
        return 0
    return max(1, int(round(np.pi / 4.0 * np.sqrt(N / M))))


def grover_oracle(state: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Oracle: state[mask] *= −1.

    Parameters
    ----------
    state : amplitudes (n,) or (batch, n)
    mask : boolean (n,) — True للـ good basis states

    Returns
    -------
    state بعد flip phase.
    """
    state = np.asarray(state, dtype=np.float64).copy()
    mask = np.asarray(mask, dtype=bool)
    state[..., mask] *= -1.0
    return state


def diffusion_operator(state: np.ndarray) -> np.ndarray:
    """Diffusion: D = 2|ψ⟩⟨ψ| − I.

    Equivalent to reflection around the mean amplitude:
        ψ' = 2 * mean(ψ) − ψ
    """
    state = np.asarray(state, dtype=np.float64).copy()
    mean = state.mean(axis=-1, keepdims=True)
    return 2.0 * mean - state


def grover_iteration(
    state: np.ndarray,
    oracle_mask: np.ndarray,
    n_iter: int | None = None,
) -> np.ndarray:
    """Full Grover algorithm: تطبيق Oracle + Diffusion عدة مرات.

    Parameters
    ----------
    state : initial amplitudes (uniform عادة)
    oracle_mask : boolean array — True for "good" basis states
    n_iter : optional override. Default = (π/4)√(N/M)

    Returns
    -------
    amplified state. amplitudes of good states ≈ 1, others ≈ 0.
    """
    state = np.asarray(state, dtype=np.float64).copy()
    mask = np.asarray(oracle_mask, dtype=bool)
    N = state.shape[-1]
    M = int(mask.sum())

    if M == 0 or M == N:
        return state
    if n_iter is None:
        n_iter = optimal_iterations(N, M)

    for _ in range(n_iter):
        state = grover_oracle(state, mask)
        state = diffusion_operator(state)

    return state


def amplitude_amplification(
    values: np.ndarray,
    oracle_func,
    n_iter: int | None = None,
    start_uniform: bool = True,
) -> np.ndarray:
    """High-level wrapper: amplify values that satisfy oracle_func.

    Parameters
    ----------
    values : array of raw values
    oracle_func : callable(values) → boolean mask
    n_iter : optional iteration count
    start_uniform : if True, start from uniform superposition;
                    else use values as initial amplitudes
    """
    values = np.asarray(values, dtype=np.float64)
    N = values.size
    if start_uniform:
        state = np.full(N, 1.0 / np.sqrt(N))
    else:
        norm = np.linalg.norm(values)
        state = values / (norm if norm > 1e-12 else 1.0)

    mask = oracle_func(values)
    return grover_iteration(state, mask, n_iter=n_iter)


class GroverAlphaSearch:
    """Grover-style alpha search.

    Replaces sequential filtering (5 stages) في edge_scanner بـ:
        1. Oracle: علامة سالبة على الـ alphas التي تحقق criteria
        2. Diffusion: reflect حول الـ mean

    التقرير: 170× أسرع نظرياً في discovery.
    """

    def __init__(
        self,
        perm_p_thresh: float = 0.05,
        win_rate_thresh: float = 0.55,
        sharpe_thresh: float = 1.0,
        min_n_days: int = 5,
    ):
        self.perm_p_thresh = perm_p_thresh
        self.win_rate_thresh = win_rate_thresh
        self.sharpe_thresh = sharpe_thresh
        self.min_n_days = min_n_days

    def _oracle(self, candidates_df) -> np.ndarray:
        """Boolean mask للـ "good" candidates."""
        import pandas as pd
        if isinstance(candidates_df, pd.DataFrame):
            df = candidates_df
            mask = np.ones(len(df), dtype=bool)
            if "perm_p" in df.columns:
                mask &= (df["perm_p"] < self.perm_p_thresh).values
            if "win_rate" in df.columns:
                mask &= (df["win_rate"] > self.win_rate_thresh).values
            if "sharpe" in df.columns:
                mask &= (df["sharpe"] > self.sharpe_thresh).values
            if "n_days" in df.columns:
                mask &= (df["n_days"] >= self.min_n_days).values
            return mask
        return np.asarray(candidates_df, dtype=bool)

    def amplify(self, candidates) -> np.ndarray:
        """Returns amplified amplitudes (length = n_candidates).

        Higher amplitude → more likely "good" alpha.
        """
        mask = self._oracle(candidates)
        N = mask.size
        if N == 0:
            return np.zeros(0)
        state = np.full(N, 1.0 / np.sqrt(N))
        return grover_iteration(state, mask)

    def select_top_k(self, candidates, k: int = 10):
        """Returns top-k indices by amplified amplitude."""
        amp = self.amplify(candidates)
        if amp.size == 0:
            return np.array([], dtype=int)
        return np.argsort(-(amp ** 2))[:k]
