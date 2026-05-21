"""
quantum/bell_pairs.py — Phase B: Entangled Features (9 Bell pairs)
══════════════════════════════════════════════════════════════════════

من التقرير 6.2 + 6.5:
  Bell state: |Φ+⟩ = (|00⟩ + |11⟩) / √2

  نبني correlations مدمجة structurally بدل ترك CNN يكتشفها وحده.
  features 138 مستقل → 30 entangled (4.6× أقل).
  DL parameters: 500K → 100K (5× أقل).
  Convergence: 50 epoch → 15 epoch.

ملاحظة: هذا quantum-inspired (classical implementation).
لا hardware كمي مطلوب — كل العمليات على numpy/pandas.

الفكرة:
  Bell pair formula:
      joint  = a * b                 # كلاهما مرتفع
      anti   = (1-a) * (1-b)         # كلاهما منخفض
      mixed  = a*(1-b) + (1-a)*b     # مختلطان
      result = joint + anti - mixed

  - يصل +1 عند التوافق (كلاهما high أو low)
  - يصل -1 عند التضاد (واحد high والآخر low)
  - constructive interference عند التوافق
  - destructive interference عند التضاد

API:
    bell_pair(a, b)            → np.ndarray in [-1, +1]
    normalize_to_unit(x)       → tanh-based mapping إلى [0, 1]
    apply_bell_pairs(df, pairs) → df مع الأعمدة الجديدة bp_*
    cnot_gate(control, target) → CNOT-like gate
    STANDARD_BELL_PAIRS        → 9 pairs المقترحة في التقرير
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# Core Bell pair formula
# ══════════════════════════════════════════════════════════════════

def normalize_to_unit(x: np.ndarray | pd.Series, method: str = "tanh") -> np.ndarray:
    """ينقل قيم إلى [0, 1] لأن الـ Bell formula يفترض probabilities.

    method:
      "tanh"     — (tanh(x) + 1) / 2 → بشكل ناعم
      "sigmoid"  — 1 / (1 + exp(-x))
      "clip"     — clip(x, 0, 1) — لو القيم أصلاً في [-1,1] نحوّلها بـ (x+1)/2
      "minmax"   — (x - min) / (max - min) per-array
    """
    arr = np.asarray(x, dtype=np.float64)
    if method == "tanh":
        return (np.tanh(arr) + 1.0) * 0.5
    if method == "sigmoid":
        return 1.0 / (1.0 + np.exp(-arr))
    if method == "clip":
        # if values in [-1, 1] → map to [0, 1]
        if arr.min() < -0.001:
            return (arr + 1.0) * 0.5
        return np.clip(arr, 0.0, 1.0)
    if method == "minmax":
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi - lo < 1e-12:
            return np.full_like(arr, 0.5)
        return (arr - lo) / (hi - lo)
    raise ValueError(f"unknown normalization method: {method!r}")


def bell_pair(
    a: np.ndarray | pd.Series,
    b: np.ndarray | pd.Series,
    normalize: bool = True,
    method: str = "clip",
) -> np.ndarray:
    """
    Bell-pair entanglement formula:
        |ψ⟩ ~ joint + anti - mixed

    لـ a, b في [0, 1]:
        result ∈ [-1, +1]
        +1 عند التوافق التام (a≈b)
        -1 عند التضاد التام (a + b ≈ 1, واحد ≈ 0 والآخر ≈ 1)
        0 عند الـ uncorrelated random

    Parameters
    ----------
    a, b : array-like (n,)
    normalize : bool
        لو True، يطبّق normalize_to_unit أولاً (default).
    method : str
        طريقة الـ normalization.
    """
    if normalize:
        a_n = normalize_to_unit(a, method=method)
        b_n = normalize_to_unit(b, method=method)
    else:
        a_n = np.asarray(a, dtype=np.float64)
        b_n = np.asarray(b, dtype=np.float64)

    if a_n.shape != b_n.shape:
        raise ValueError(f"shape mismatch: {a_n.shape} vs {b_n.shape}")

    joint = a_n * b_n                       # كلاهما مرتفع
    anti = (1.0 - a_n) * (1.0 - b_n)        # كلاهما منخفض
    mixed = a_n * (1.0 - b_n) + (1.0 - a_n) * b_n  # مختلطان
    return joint + anti - mixed


def cnot_gate(
    control: np.ndarray | pd.Series,
    target: np.ndarray | pd.Series,
    threshold: float = 0.5,
    amplification: float = 1.5,
    deflation: float = 0.5,
    normalize_control: bool = True,
    normalize_method: str = "clip",
) -> np.ndarray:
    """
    CNOT-like gate: target يتغيّر IFF control في حالة معينة (> threshold).

    من التقرير 6.2:
        if control > THRESHOLD:
            target *= AMPLIFICATION
        else:
            target *= DEFLATION
    """
    if normalize_control:
        c = normalize_to_unit(control, method=normalize_method)
    else:
        c = np.asarray(control, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    return np.where(c > threshold, t * amplification, t * deflation)


# ══════════════════════════════════════════════════════════════════
# Bell pairs configuration
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BellPairSpec:
    """مواصفات Bell pair واحد."""
    name: str               # output column name (bp_<name>)
    feat_a: str             # column name لـ a
    feat_b: str             # column name لـ b
    meaning: str = ""       # وصف للتشخيص
    method: str = "clip"    # طريقة الـ normalization


# الـ 9 Bell pairs المُقترَحة في التقرير 6.5
# تُستخدم أعمدة V19.2 من feature_simulators + wall_depth_simulator + iceberg_simulator
STANDARD_BELL_PAIRS: list[BellPairSpec] = [
    # BP1: Wall ⊗ Iceberg → نشاط مؤسسي حقيقي
    BellPairSpec(
        name="wall_iceberg",
        feat_a="sim_wall_persist",
        feat_b="sim_iceberg_strength",
        meaning="نشاط مؤسسي حقيقي vs noise",
    ),
    # BP2: Depth ⊗ Informed → directional flow
    BellPairSpec(
        name="depth_informed",
        feat_a="sim_depth_pressure",
        feat_b="sim_informed_prob",
        meaning="directional flow strength",
    ),
    # BP3: Sweep ⊗ Absorb → exhaustion signals
    BellPairSpec(
        name="sweep_absorb",
        feat_a="sim_sweep_signal",
        feat_b="sim_absorb_intensity",
        meaning="exhaustion signals",
    ),
    # BP4: Regime ⊗ Horizon → timing context
    # نستخدم volatility_regime كـ proxy للـ regime
    BellPairSpec(
        name="regime_volatility",
        feat_a="sim_volatility_regime",
        feat_b="sim_flow_consistency",
        meaning="timing context (regime × flow stability)",
    ),
    # BP5: Wall_growth ⊗ Iceberg_replenish → accumulation
    BellPairSpec(
        name="wall_growth_iceberg_replenish",
        feat_a="sim_wall_growth",
        feat_b="sim_iceberg_replenish",
        meaning="institutional accumulation",
    ),
    # BP6: Wall_real ⊗ Wall_consumed → wall validity
    BellPairSpec(
        name="wall_real_consumed",
        feat_a="sim_wall_real",
        feat_b="sim_wall_consumed",
        meaning="real wall being consumed (not spoofing)",
    ),
    # BP7: Flow_direction ⊗ Depth_imbalance → directional pressure
    BellPairSpec(
        name="flow_depth",
        feat_a="sim_flow_direction",
        feat_b="sim_depth_imbalance",
        meaning="directional pressure alignment",
    ),
    # BP8: Iceberg_side ⊗ Informed_direction → directional institutions
    BellPairSpec(
        name="iceberg_informed",
        feat_a="sim_iceberg_side",
        feat_b="sim_informed_direction",
        meaning="directional institutional activity",
    ),
    # BP9: Liquidity ⊗ Data_quality → signal trustworthiness
    BellPairSpec(
        name="liquidity_quality",
        feat_a="sim_liquidity_state",
        feat_b="sim_data_quality",
        meaning="signal trustworthiness (high liq + good data)",
    ),
]


def apply_bell_pairs(
    df: pd.DataFrame,
    pairs: list[BellPairSpec] = STANDARD_BELL_PAIRS,
    prefix: str = "bp_",
    skip_missing: bool = True,
) -> pd.DataFrame:
    """يطبّق list من Bell pairs ويضيف أعمدة bp_<name> للـ df.

    Parameters
    ----------
    df : DataFrame مع أعمدة sim_* (من feature_simulators).
    pairs : قائمة BellPairSpec للتطبيق.
    prefix : prefix للأعمدة الناتجة (default "bp_").
    skip_missing : لو True، يتخطّى pairs بأعمدة مفقودة بدلاً من رفع error.

    Returns
    -------
    DataFrame مع الأعمدة الجديدة مضافة.
    """
    out = df.copy()
    for spec in pairs:
        if spec.feat_a not in df.columns or spec.feat_b not in df.columns:
            if skip_missing:
                continue
            raise KeyError(
                f"Bell pair {spec.name!r}: columns missing "
                f"(a={spec.feat_a!r}, b={spec.feat_b!r})"
            )
        col_name = f"{prefix}{spec.name}"
        out[col_name] = bell_pair(
            df[spec.feat_a].to_numpy(),
            df[spec.feat_b].to_numpy(),
            normalize=True,
            method=spec.method,
        ).astype(np.float32)
    return out


def get_bell_pair_columns(prefix: str = "bp_") -> list[str]:
    """أسماء أعمدة الـ Bell pairs الناتجة بالترتيب."""
    return [f"{prefix}{spec.name}" for spec in STANDARD_BELL_PAIRS]


__all__ = [
    "BellPairSpec",
    "STANDARD_BELL_PAIRS",
    "normalize_to_unit",
    "bell_pair",
    "cnot_gate",
    "apply_bell_pairs",
    "get_bell_pair_columns",
]
