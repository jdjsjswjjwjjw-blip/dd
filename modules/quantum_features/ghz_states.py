"""
modules/quantum_features/ghz_states.py
──────────────────────────────────────
Sprint 2 (التقرير 6.4): 3+ particle entangled features.

GHZ state: |GHZ⟩ = (|000⟩ + |111⟩)/√2

Used for high-order correlations:
    - 3-feature institutional activity (wall + iceberg + informed)
    - 3-feature intensity (depth + volume + volatility)
    - 3-feature spatial-temporal context (zone + level + trend)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.quantum_core.entanglement_gates import ghz_state


STANDARD_GHZ_TRIPLETS = [
    {
        "name": "wall_iceberg_informed",
        "cols": ["wall_persist_score", "sim_iceberg_strength", "sim_informed_prob"],
        "description": "ثلاثي نشاط مؤسسي قوي",
    },
    {
        "name": "depth_volume_volatility",
        "cols": ["sim_depth_pressure", "volume_zscore", "atr_14"],
        "description": "intensity ثلاثي للحركة",
    },
    {
        "name": "zone_level_trend",
        "cols": ["zone_id_numeric", "level_distance", "kalman_trend"],
        "description": "spatial-temporal context",
    },
    {
        "name": "absorption_orderflow_sweep",
        "cols": ["sim_absorption", "sim_orderflow", "sim_sweep"],
        "description": "exhaustion 3-particle",
    },
    {
        "name": "buy_pressure_triplet",
        "cols": ["sim_buy_pressure", "sim_buy_inertia", "sim_buy_continuation"],
        "description": "buy momentum 3-particle",
    },
]


def _minmax_normalize(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize array to [0, 1]."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr
    vmin = arr.min()
    vmax = arr.max()
    rng = vmax - vmin
    if rng < eps:
        return np.zeros_like(arr)
    return (arr - vmin) / rng


def compute_ghz_features(
    df: pd.DataFrame,
    triplets: list[dict] | None = None,
    skip_missing: bool = True,
    fillna: float = 0.0,
) -> dict[str, np.ndarray]:
    """يحسب GHZ features من DataFrame.

    Parameters
    ----------
    df : DataFrame مع feature columns
    triplets : list of dicts كل dict له {name, cols, description}.
               Default = STANDARD_GHZ_TRIPLETS
    skip_missing : إذا الـ cols غير موجودة، تخطّى. لو False يرفع KeyError.
    fillna : value لـ NaN replacement قبل GHZ

    Returns
    -------
    dict {f"ghz_{name}": np.ndarray}
    """
    if triplets is None:
        triplets = STANDARD_GHZ_TRIPLETS
    out: dict[str, np.ndarray] = {}
    for trip in triplets:
        name = trip["name"]
        cols = trip["cols"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            if skip_missing:
                continue
            raise KeyError(f"Missing GHZ columns for {name!r}: {missing}")
        feats = []
        for c in cols:
            v = df[c].fillna(fillna).values.astype(np.float64)
            feats.append(_minmax_normalize(v))
        out[f"ghz_{name}"] = ghz_state(feats)
    return out


def add_ghz_features(
    df: pd.DataFrame,
    triplets: list[dict] | None = None,
    skip_missing: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """يضيف GHZ features كـ DataFrame columns."""
    features = compute_ghz_features(df, triplets=triplets, skip_missing=skip_missing)
    if inplace:
        result = df
    else:
        result = df.copy()
    for k, v in features.items():
        result[k] = v
    return result
