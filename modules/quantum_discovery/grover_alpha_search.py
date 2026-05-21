"""
modules/quantum_discovery/grover_alpha_search.py
────────────────────────────────────────────────
Sprint 3 (Phase C - التقرير 6.5): Grover-based alpha discovery.

يستبدل sequential filtering (5 stages في edge_scanner) بـ:
    1. توليد candidates (vectorized via edge_scanner_v2)
    2. Grover oracle: علامة على alphas تحقق criteria
    3. Diffusion: عكس حول الـ mean → amplification

التقرير: O(√N) بدل O(N)، 170× أسرع نظرياً.

العملياً: edge_scanner_v2 يبقى الـ candidate generator (vectorized، سريع).
الـ Grover هنا يفعّل الـ scoring/ranking امبيرياً.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from modules.quantum_core.interference_amplifier import (
    grover_iteration,
    optimal_iterations,
)
from modules.quantum_core.measurement_engine import (
    born_probabilities,
    confidence_from_state,
)


@dataclass
class GroverDiscoveryConfig:
    """إعدادات Grover-based alpha search."""

    # Oracle criteria
    perm_p_max: float = 0.05
    win_rate_min: float = 0.55
    sharpe_min: float = 1.0
    n_days_min: int = 5
    n_trades_min: int = 10

    # Grover hyperparameters
    n_iter: int | None = None  # default = optimal (π/4)√(N/M)
    n_iter_cap: int = 50  # safety cap
    custom_oracle: Callable[[pd.DataFrame], np.ndarray] | None = None

    # Output
    top_k: int = 15
    min_confidence: float = 0.05


@dataclass
class GroverSearchResult:
    """نتائج البحث."""

    selected_alphas: pd.DataFrame
    amplitudes: np.ndarray
    n_iter: int
    n_candidates: int
    n_marked: int
    confidence: float
    diagnostics: dict = field(default_factory=dict)


def _default_oracle(candidates: pd.DataFrame, config: GroverDiscoveryConfig) -> np.ndarray:
    """Boolean mask: True للـ candidates التي تحقق criteria."""
    mask = np.ones(len(candidates), dtype=bool)
    if "perm_p" in candidates.columns:
        mask &= (candidates["perm_p"] < config.perm_p_max).values
    if "p_value" in candidates.columns and "perm_p" not in candidates.columns:
        mask &= (candidates["p_value"] < config.perm_p_max).values
    if "win_rate" in candidates.columns:
        mask &= (candidates["win_rate"] >= config.win_rate_min).values
    if "sharpe" in candidates.columns:
        mask &= (candidates["sharpe"] >= config.sharpe_min).values
    if "n_days" in candidates.columns:
        mask &= (candidates["n_days"] >= config.n_days_min).values
    if "n_trades" in candidates.columns:
        mask &= (candidates["n_trades"] >= config.n_trades_min).values
    return mask


def grover_alpha_search(
    candidates: pd.DataFrame,
    config: GroverDiscoveryConfig | None = None,
) -> GroverSearchResult:
    """Grover-based alpha discovery.

    Parameters
    ----------
    candidates : DataFrame مع candidates + metrics (perm_p, win_rate, ...)
    config : GroverDiscoveryConfig

    Returns
    -------
    GroverSearchResult
    """
    if config is None:
        config = GroverDiscoveryConfig()

    N = len(candidates)
    if N == 0:
        return GroverSearchResult(
            selected_alphas=candidates.iloc[:0].copy(),
            amplitudes=np.zeros(0),
            n_iter=0,
            n_candidates=0,
            n_marked=0,
            confidence=0.0,
            diagnostics={"reason": "empty_candidates"},
        )

    # Oracle mask
    if config.custom_oracle is not None:
        mask = config.custom_oracle(candidates)
    else:
        mask = _default_oracle(candidates, config)
    M = int(mask.sum())

    if M == 0:
        return GroverSearchResult(
            selected_alphas=candidates.iloc[:0].copy(),
            amplitudes=np.zeros(N),
            n_iter=0,
            n_candidates=N,
            n_marked=0,
            confidence=0.0,
            diagnostics={"reason": "no_candidates_pass_oracle"},
        )

    # Uniform superposition start
    state = np.full(N, 1.0 / np.sqrt(N))

    # Grover iterations
    n_iter = config.n_iter if config.n_iter is not None else optimal_iterations(N, M)
    n_iter = min(n_iter, config.n_iter_cap)
    amplified = grover_iteration(state, mask, n_iter=n_iter)

    # Born probabilities → ranking
    p = born_probabilities(amplified)
    conf = float(confidence_from_state(amplified))

    # Select top-K, filter by oracle (لا نأخذ شيء فشل في الـ oracle حتى لو amplitude عالية)
    order = np.argsort(-p)
    # First pass: keep only those that passed oracle
    keep_idx = [i for i in order if mask[i]]
    selected_idx = keep_idx[: config.top_k]
    selected = candidates.iloc[selected_idx].copy()
    selected["grover_amplitude"] = amplified[selected_idx]
    selected["grover_probability"] = p[selected_idx]

    return GroverSearchResult(
        selected_alphas=selected,
        amplitudes=amplified,
        n_iter=n_iter,
        n_candidates=N,
        n_marked=M,
        confidence=conf,
        diagnostics={
            "n_iter_optimal": optimal_iterations(N, M),
            "n_iter_used": n_iter,
            "marked_total_prob": float(p[mask].sum()),
            "max_amplitude": float(np.max(np.abs(amplified))),
            "uniform_amplitude": 1.0 / np.sqrt(N),
        },
    )


def grover_from_edge_scanner(
    df: pd.DataFrame,
    config: GroverDiscoveryConfig | None = None,
    run_permutation: bool = True,
    n_permutations: int = 500,
    verbose: bool = True,
    horizons: list[int] | None = None,
) -> GroverSearchResult:
    """End-to-end: edge_scanner_v2 → candidates → Grover ranking.

    يفترض أن v19_2 متاح في sys.path.
    """
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    v19_path = os.path.join(_ROOT, "v19_2")
    if v19_path not in sys.path:
        sys.path.insert(0, v19_path)
    from edge_scanner_v2 import scan_edges_v2

    if horizons is None:
        horizons = [3, 6, 12]

    res = scan_edges_v2(
        df, horizons=horizons,
        run_permutation=run_permutation,
        n_permutations=n_permutations,
        verbose=verbose,
    )

    # candidates للـ Grover: validation pass + holdout pass
    if "validation_candidates" in res:
        candidates = pd.DataFrame(res["validation_candidates"])
    elif "candidates" in res:
        candidates = pd.DataFrame(res["candidates"])
    else:
        candidates = pd.DataFrame()

    return grover_alpha_search(candidates, config=config)
