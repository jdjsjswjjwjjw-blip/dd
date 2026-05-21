"""
modules/statistical_validation_layer.py — Phase 4: V19.2 discovery قبل ML training.
══════════════════════════════════════════════════════════════════════

الفكرة (من التقرير، المرحلة 4):
  قبل تدريب الـ ML على كل الـ 138 feature، نشغّل V19.2 discovery:
    - edge_scanner_v2 يكتشف 5-15 alpha موثوقة
    - statistics_module يطبّق FDR + permutation + 3-way split
    - النتيجة: قائمة edges احصائياً موثوقة (validated)

  الـ ML تدريب يبدأ بـ:
    - filter mask على الـ training data (rows المنطبقة على الـ alphas)
    - signal-to-noise أعلى 10× (5,000 sample 'محددة' بدل 12K مختلطة)

API:
    discover_alphas(df, symbol, confidence) -> AlphaSet
    apply_alpha_filter(df, alpha_set) -> filtered_df + mask
    AlphaSet.summary() -> dict (للتشخيص)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_V19_2 = os.path.join(os.path.dirname(_HERE), "v19_2")
if os.path.isdir(_V19_2) and _V19_2 not in sys.path:
    sys.path.insert(0, _V19_2)


@dataclass
class AlphaSet:
    """نتيجة discovery: قائمة edges موثوقة + معايير."""
    candidates: list[dict] = field(default_factory=list)
    criteria: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    split_info: dict = field(default_factory=dict)

    @classmethod
    def from_scan_result(cls, result: dict) -> "AlphaSet":
        return cls(
            candidates=result.get("candidates", []),
            criteria=result.get("criteria", {}),
            diagnostics=result.get("diagnostics", {}),
            split_info=result.get("split_info", {}),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "n_alphas": len(self.candidates),
            "diagnostics": self.diagnostics,
            "split": self.split_info,
            "directions": {
                "long":  sum(1 for c in self.candidates if c.get("direction") == 1),
                "short": sum(1 for c in self.candidates if c.get("direction") == -1),
            },
        }

    def is_valid(self) -> bool:
        """فيها alphas موثوقة (stage 5 passed)؟"""
        return len(self.candidates) > 0


def discover_alphas(
    df: pd.DataFrame,
    symbol: str = "6B",
    confidence: str = "medium",
    horizons: list[int] = [3, 6, 12],
    run_permutation: bool = True,
    use_vectorized: bool = True,
    verbose: bool = False,
) -> AlphaSet:
    """يشغّل V19.2 edge_scanner ويرجع alphas موثوقة.

    Parameters
    ----------
    df : DataFrame بمخرج enrich_features (مع sim_* + zone_full + *_event)
    symbol : market symbol (e.g. "6B")
    confidence : "low" | "medium" | "high"
    horizons : list of forward-return horizons
    run_permutation : لو True يشغّل permutation test (أبطأ لكن أصرم)
    use_vectorized : لو True يستخدم scan_edges_v2 (Phase A)، False = original

    Returns
    -------
    AlphaSet مع candidates التي نجحت في كل الـ 5 stages.
    """
    if use_vectorized:
        from edge_scanner_v2 import scan_edges_v2 as scan_fn
    else:
        from edge_scanner import scan_edges as scan_fn

    result = scan_fn(
        df,
        horizons=horizons,
        symbol=symbol,
        confidence=confidence,
        run_permutation=run_permutation,
        verbose=verbose,
    )
    return AlphaSet.from_scan_result(result)


def apply_alpha_filter(
    df: pd.DataFrame,
    alpha_set: AlphaSet,
    return_mask: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """يطبّق alphas المُكتشَفة على DataFrame كـ filter.

    يبني bool mask = OR للـ rows اللي تنطبق على أي alpha من الـ set.
    النتيجة: 'محدد' rows فقط (الباقي يُلغى).

    Parameters
    ----------
    df : DataFrame مع zone_full, *_event, sim_*
    alpha_set : AlphaSet من discover_alphas
    return_mask : لو True يرجع (filtered_df, mask)

    Returns
    -------
    filtered DataFrame (rows اللي ضمن alphas)، أو tuple (df, mask).
    """
    if not alpha_set.is_valid():
        empty = np.zeros(len(df), dtype=bool)
        if return_mask:
            return df.iloc[empty].copy(), empty
        return df.iloc[empty].copy()

    # نستخدم نفس الـ pre-computed masks logic من edge_scanner_v2
    from edge_scanner_v2 import (
        _precompute_combo_masks,
        _precompute_event_masks,
        _precompute_zone_masks,
    )
    from edge_scanner import LEVEL_NAMES, EVENT_TYPES

    zones = list({c["zone"] for c in alpha_set.candidates})
    combo_masks = _precompute_combo_masks(df)
    event_masks = _precompute_event_masks(df, LEVEL_NAMES, EVENT_TYPES)
    zone_masks = _precompute_zone_masks(df, zones)

    # OR all alpha masks
    n = len(df)
    union_mask = np.zeros(n, dtype=bool)
    for c in alpha_set.candidates:
        z = zone_masks.get(c["zone"])
        em = event_masks.get((c["level"], c["event"]))
        cm = combo_masks.get(c["combo"])
        if z is None or em is None or cm is None:
            continue
        union_mask |= (z & em & cm)

    filtered = df.iloc[union_mask].copy()
    if return_mask:
        return filtered, union_mask
    return filtered


def signal_to_noise_estimate(
    df: pd.DataFrame,
    alpha_set: AlphaSet,
    horizon: int = 6,
) -> dict[str, float]:
    """يقدّر signal-to-noise قبل/بعد الـ alpha filter.

    يحسب |mean(fwd_ret)| / std(fwd_ret) على:
      - all rows (baseline)
      - alpha-filtered rows (selective)
    """
    col = f"fwd_ret_{horizon}"
    if col not in df.columns:
        return {"error": -1.0}

    rets_all = df[col].dropna().to_numpy()
    snr_all = abs(rets_all.mean()) / max(rets_all.std(), 1e-9)

    filtered = apply_alpha_filter(df, alpha_set)
    if len(filtered) == 0:
        return {
            "snr_all": float(snr_all),
            "snr_filtered": 0.0,
            "ratio": 0.0,
            "n_all": len(rets_all),
            "n_filtered": 0,
        }
    rets_f = filtered[col].dropna().to_numpy()
    snr_f = abs(rets_f.mean()) / max(rets_f.std(), 1e-9)
    return {
        "snr_all": float(snr_all),
        "snr_filtered": float(snr_f),
        "ratio": float(snr_f / max(snr_all, 1e-9)),
        "n_all": int(len(rets_all)),
        "n_filtered": int(len(rets_f)),
    }


__all__ = [
    "AlphaSet",
    "discover_alphas",
    "apply_alpha_filter",
    "signal_to_noise_estimate",
]
