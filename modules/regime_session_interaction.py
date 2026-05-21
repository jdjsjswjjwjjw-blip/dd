"""
modules/regime_session_interaction.py
─────────────────────────────────────
Sprint 12 (Regime Analysis Report v2, التعديل ②):
Regime × Session Interaction — يميّز "trending في London" عن "trending في Asia".

═════════════════════════════════════════════════════════════════════════════
الفكرة العلمية (Statistical Hypothesis Testing)
═════════════════════════════════════════════════════════════════════════════

Null hypothesis (H₀):
    P(direction | regime) = P(direction | regime, session)
    أي: الـ session لا يضيف معلومات على الـ regime.

Alternative hypothesis (H₁):
    P(direction | regime, session) ≠ P(direction | regime)
    الـ session يُغيّر سلوك الـ regime جوهرياً.

Test الإحصائي:
    Chi-square test of independence على contingency table
    (regime × session) vs label distribution.

Effect size:
    Cramér's V = √(χ² / (n × min(r-1, c-1)))
    V > 0.10 = weak، V > 0.30 = moderate، V > 0.50 = strong association.

═════════════════════════════════════════════════════════════════════════════
المرجع التشريحي (التقرير ص 9):
═════════════════════════════════════════════════════════════════════════════

    "ابق على 3 regimes لكن أضف session_zone كـ feature إضافي.
     النموذج يتعلم: trending في London ≠ trending في Asia."

دعم بحثي:
    - Asia session: manipulation-driven → false breakouts dominant
      (Bouchaud, "Trades, Quotes and Prices", 2018)
    - London session: institutional flow → trending أقوى
      (Lyons, "The Microstructure Approach to Exchange Rates", 2001)
    - NY: news-driven volatile bursts
      (Andersen et al., "Real-Time Price Discovery", 2003)

═════════════════════════════════════════════════════════════════════════════
الـ Cardinality Math
═════════════════════════════════════════════════════════════════════════════

    regimes:     3 (trending, ranging, volatile)
    sessions:    16 zones (4 sessions × 4 quartiles)
    combinations: 3 × 16 = 48

Minimum support per cell (rule of thumb):
    n_min = max(200, 0.5% × N_total)

    => للـ 12K bars: n_min = max(200, 60) = 200 samples per cell
    => للـ 100K bars: n_min = max(200, 500) = 500 samples per cell

Rare combinations (< n_min) تُجمّع في 'other' لمنع overfitting.

═════════════════════════════════════════════════════════════════════════════
API
═════════════════════════════════════════════════════════════════════════════

    >>> from modules.regime_session_interaction import (
    ...     add_regime_session_features,
    ...     compute_interaction_statistics,
    ...     validate_combination_support,
    ... )
    >>> df_enriched = add_regime_session_features(
    ...     df,
    ...     regime_col='regime_label',
    ...     session_col='zone_full',
    ...     min_support=200,
    ... )
    >>> stats = compute_interaction_statistics(df_enriched, target_col='bias_label')
    >>> print(f"Cramér's V = {stats['cramers_v']:.3f}")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_REGIMES: tuple[str, ...] = ("trending", "ranging", "volatile")
RARE_LABEL: str = "other"  # bucket for under-supported combinations

# Cramér's V effect-size benchmarks (Cohen 1988)
CRAMERS_V_WEAK = 0.10
CRAMERS_V_MODERATE = 0.30
CRAMERS_V_STRONG = 0.50


@dataclass
class InteractionStatistics:
    """نتائج التحقق الإحصائي للـ regime × session interaction."""

    chi_square: float
    p_value: float
    degrees_of_freedom: int
    cramers_v: float
    effect_size_class: str  # 'negligible' | 'weak' | 'moderate' | 'strong'
    n_combinations: int
    n_rare: int
    min_cell_count: int
    median_cell_count: int
    contingency_shape: tuple[int, int]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chi_square": float(self.chi_square),
            "p_value": float(self.p_value),
            "degrees_of_freedom": int(self.degrees_of_freedom),
            "cramers_v": float(self.cramers_v),
            "effect_size_class": self.effect_size_class,
            "n_combinations": int(self.n_combinations),
            "n_rare": int(self.n_rare),
            "min_cell_count": int(self.min_cell_count),
            "median_cell_count": int(self.median_cell_count),
            "contingency_shape": [int(self.contingency_shape[0]),
                                  int(self.contingency_shape[1])],
            "diagnostics": dict(self.diagnostics),
        }


# ════════════════════════════════════════════════════════════════════════════
# Core: build interaction label
# ════════════════════════════════════════════════════════════════════════════


def build_regime_session_label(
    df: pd.DataFrame,
    regime_col: str = "regime_label",
    session_col: str = "zone_full",
    separator: str = "_x_",
    na_placeholder: str = "na",
) -> pd.Series:
    """يبني regime × session label كـ categorical string.

    Parameters
    ----------
    df : DataFrame مع regime_col و session_col
    regime_col : اسم column الـ regime
    session_col : اسم column الـ session zone
    separator : الفاصل بين الـ tokens (default "_x_" لتمييزه عن underscore الطبيعي)
    na_placeholder : value لـ NaN replacement

    Returns
    -------
    Series من strings: e.g. "trending_x_T_REG_LONDON_Q1"

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "regime_label": ["trending", "ranging", "trending"],
    ...     "zone_full": ["T_REG_LONDON_Q1", "T_REG_ASIA_Q2", "T_REG_NY_Q1"],
    ... })
    >>> build_regime_session_label(df).tolist()
    ['trending_x_T_REG_LONDON_Q1', 'ranging_x_T_REG_ASIA_Q2', 'trending_x_T_REG_NY_Q1']
    """
    if regime_col not in df.columns:
        raise KeyError(f"regime_col '{regime_col}' غير موجود")
    if session_col not in df.columns:
        raise KeyError(f"session_col '{session_col}' غير موجود")

    regime = df[regime_col].fillna(na_placeholder).astype(str)
    session = df[session_col].fillna(na_placeholder).astype(str)
    return regime + separator + session


# ════════════════════════════════════════════════════════════════════════════
# Validation: combination support
# ════════════════════════════════════════════════════════════════════════════


def validate_combination_support(
    df: pd.DataFrame,
    regime_col: str = "regime_label",
    session_col: str = "zone_full",
    min_support: int | None = None,
) -> dict[str, Any]:
    """يقيّم coverage لكل combination قبل بناء الـ feature.

    Parameters
    ----------
    df : DataFrame
    regime_col, session_col : أعمدة الـ interaction
    min_support : حد أدنى للـ counts. إذا None → max(200, 0.5% × N).

    Returns
    -------
    dict مع:
        n_total : عدد الـ rows
        n_combinations : عدد الـ unique combinations
        min_support : العتبة المُستخدمة
        rare_combinations : list of strings تحت العتبة
        support_distribution : dict {combo: count}
        coverage_pct : النسبة المئوية للـ rows في combinations valid
    """
    n_total = len(df)
    if n_total == 0:
        return {
            "n_total": 0, "n_combinations": 0, "min_support": 0,
            "rare_combinations": [], "support_distribution": {},
            "coverage_pct": 0.0,
        }

    if min_support is None:
        min_support = max(200, int(0.005 * n_total))

    label = build_regime_session_label(df, regime_col, session_col)
    counts = label.value_counts()

    rare_mask = counts < min_support
    rare = counts[rare_mask].index.tolist()
    valid = counts[~rare_mask].index.tolist()

    valid_rows = int(counts[~rare_mask].sum())
    coverage = 100.0 * valid_rows / n_total if n_total > 0 else 0.0

    return {
        "n_total": int(n_total),
        "n_combinations": int(len(counts)),
        "min_support": int(min_support),
        "rare_combinations": [str(c) for c in rare],
        "valid_combinations": [str(c) for c in valid],
        "support_distribution": {str(k): int(v) for k, v in counts.items()},
        "coverage_pct": float(coverage),
    }


# ════════════════════════════════════════════════════════════════════════════
# Statistical test: chi-square + Cramér's V
# ════════════════════════════════════════════════════════════════════════════


def _classify_effect_size(v: float) -> str:
    if v < CRAMERS_V_WEAK:
        return "negligible"
    if v < CRAMERS_V_MODERATE:
        return "weak"
    if v < CRAMERS_V_STRONG:
        return "moderate"
    return "strong"


def compute_interaction_statistics(
    df: pd.DataFrame,
    target_col: str = "bias_label",
    regime_col: str = "regime_label",
    session_col: str = "zone_full",
    min_support: int | None = None,
) -> InteractionStatistics:
    """يحسب chi-square + Cramér's V للـ interaction.

    Hypothesis test:
        H₀: regime_session label is independent of target
        H₁: regime_session label predicts target

    Implementation: pure numpy chi-square (no scipy dependency).

    Parameters
    ----------
    df : DataFrame
    target_col : العمود الـ target (مثلاً bias_label)
    regime_col, session_col : أعمدة الـ interaction
    min_support : عتبة الـ rare combinations

    Returns
    -------
    InteractionStatistics dataclass
    """
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' غير موجود")

    label = build_regime_session_label(df, regime_col, session_col)
    target = df[target_col]

    # Contingency table
    table = pd.crosstab(label, target)
    n = int(table.values.sum())

    if n == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return InteractionStatistics(
            chi_square=0.0, p_value=1.0, degrees_of_freedom=0,
            cramers_v=0.0, effect_size_class="negligible",
            n_combinations=int(table.shape[0]),
            n_rare=0,
            min_cell_count=0,
            median_cell_count=0,
            contingency_shape=table.shape,
            diagnostics={"reason": "insufficient_dimensions"},
        )

    observed = table.values.astype(np.float64)
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    expected = (row_totals @ col_totals) / n

    # Chi-square statistic
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_sq_terms = np.where(expected > 0,
                                (observed - expected) ** 2 / expected,
                                0.0)
    chi_sq = float(chi_sq_terms.sum())

    # Degrees of freedom
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)

    # p-value via chi-square survival (manual, no scipy)
    # Wilson-Hilferty approximation: convert χ² to standard normal
    # then use normal CDF
    if dof > 0 and chi_sq > 0:
        # Wilson-Hilferty: z = ((chi²/dof)^(1/3) - (1 - 2/(9*dof))) / sqrt(2/(9*dof))
        z = ((chi_sq / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) \
            / np.sqrt(2.0 / (9.0 * dof))
        # p-value = 1 - Phi(z) ≈ via erfc
        from math import erfc, sqrt
        p_value = 0.5 * erfc(z / sqrt(2))
    else:
        p_value = 1.0

    # Cramér's V
    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
    if n > 0 and min_dim > 0:
        cramers_v = float(np.sqrt(chi_sq / (n * min_dim)))
        cramers_v = min(cramers_v, 1.0)
    else:
        cramers_v = 0.0

    # Support stats
    support = validate_combination_support(df, regime_col, session_col, min_support)
    cell_counts = np.array(list(support["support_distribution"].values()))

    return InteractionStatistics(
        chi_square=chi_sq,
        p_value=float(np.clip(p_value, 0.0, 1.0)),
        degrees_of_freedom=int(dof),
        cramers_v=cramers_v,
        effect_size_class=_classify_effect_size(cramers_v),
        n_combinations=int(table.shape[0]),
        n_rare=len(support["rare_combinations"]),
        min_cell_count=int(cell_counts.min()) if cell_counts.size > 0 else 0,
        median_cell_count=int(np.median(cell_counts)) if cell_counts.size > 0 else 0,
        contingency_shape=table.shape,
        diagnostics={
            "min_support_threshold": support["min_support"],
            "coverage_pct": support["coverage_pct"],
            "target_categories": int(table.shape[1]),
        },
    )


# ════════════════════════════════════════════════════════════════════════════
# Public API: add features to DataFrame
# ════════════════════════════════════════════════════════════════════════════


def add_regime_session_features(
    df: pd.DataFrame,
    regime_col: str = "regime_label",
    session_col: str = "zone_full",
    out_col: str = "regime_session",
    min_support: int | None = None,
    bucket_rare: bool = True,
    inplace: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """يضيف regime_session interaction column.

    Parameters
    ----------
    df : DataFrame
    regime_col, session_col : أعمدة المصدر
    out_col : اسم الـ column الجديد (default "regime_session")
    min_support : عتبة rare combinations. None → auto (max(200, 0.5%))
    bucket_rare : إذا True، rare combinations → "other"
    inplace : modify الـ df مباشرة
    verbose : print statistics

    Returns
    -------
    DataFrame مع column جديد out_col (categorical).

    Notes
    -----
    الـ column يكون من نوع `pd.Categorical` لتوافق CatBoost.
    لو regime_col أو session_col غير موجود → يطلق KeyError.
    """
    if regime_col not in df.columns:
        raise KeyError(
            f"regime_col '{regime_col}' غير موجود في الـ DataFrame. "
            f"تأكد إن prepare_day_trading شغّال مع regime detection."
        )
    if session_col not in df.columns:
        raise KeyError(
            f"session_col '{session_col}' غير موجود. "
            f"تأكد إن feature_enrichment(add_session_mapping=True) شغّال."
        )

    label = build_regime_session_label(df, regime_col, session_col)

    if bucket_rare:
        n_total = len(df)
        if min_support is None:
            min_support = max(200, int(0.005 * n_total))
        counts = label.value_counts()
        rare = set(counts[counts < min_support].index)
        if rare:
            label = label.where(~label.isin(rare), other=RARE_LABEL)
            if verbose:
                print(f"   [regime_session] bucketed {len(rare)} rare → '{RARE_LABEL}'")

    result = df if inplace else df.copy()
    result[out_col] = pd.Categorical(label)

    if verbose:
        n_unique = result[out_col].nunique()
        print(f"   [regime_session] added '{out_col}' with {n_unique} unique categories")

    return result


# ════════════════════════════════════════════════════════════════════════════
# Monte Carlo: stability test
# ════════════════════════════════════════════════════════════════════════════


def monte_carlo_interaction_stability(
    df: pd.DataFrame,
    target_col: str = "bias_label",
    regime_col: str = "regime_label",
    session_col: str = "zone_full",
    n_iterations: int = 100,
    bootstrap_frac: float = 0.7,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte Carlo: يتحقّق من الـ Cramér's V يبقى مستقر تحت bootstrap.

    لو الـ V يتذبذب كثيراً = الـ effect مش robust.
    لو ضيق = الـ interaction reliable.

    Parameters
    ----------
    df : DataFrame
    n_iterations : عدد الـ bootstrap samples
    bootstrap_frac : نسبة sample size

    Returns
    -------
    dict مع mean, std, 95% CI من cramers_v + percentile distribution
    """
    rng = np.random.RandomState(seed)
    n = len(df)
    sample_size = max(1, int(n * bootstrap_frac))

    v_values = []
    for it in range(n_iterations):
        idx = rng.choice(n, sample_size, replace=True)
        # reset_index لتجنّب duplicate-label conflict في pd.crosstab
        sub = df.iloc[idx].reset_index(drop=True)
        try:
            stats = compute_interaction_statistics(
                sub, target_col, regime_col, session_col,
            )
            v_values.append(stats.cramers_v)
        except (KeyError, ValueError):
            continue

    if not v_values:
        return {"reason": "all_iterations_failed", "n_iterations": n_iterations}

    arr = np.array(v_values)
    return {
        "n_iterations": int(n_iterations),
        "n_successful": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci_95_lo": float(np.percentile(arr, 2.5)),
        "ci_95_hi": float(np.percentile(arr, 97.5)),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "stability_class": (
            "stable" if arr.std() < 0.05 else
            "moderate" if arr.std() < 0.15 else
            "unstable"
        ),
    }


# ════════════════════════════════════════════════════════════════════════════
# High-level helper for pipeline integration
# ════════════════════════════════════════════════════════════════════════════


def enrich_with_regime_session(
    df: pd.DataFrame,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One-shot integration: validate + add + return diagnostics.

    Returns
    -------
    (df_enriched, diagnostics_dict)
    """
    # Check columns exist (graceful skip)
    if "regime_label" not in df.columns or "zone_full" not in df.columns:
        if verbose:
            print("   [regime_session] skipped: missing regime_label or zone_full")
        return df, {"skipped": True, "reason": "missing_columns"}

    support = validate_combination_support(df)
    enriched = add_regime_session_features(df, verbose=verbose)

    diagnostics = {"support": support}
    if "bias_label" in df.columns:
        try:
            stats = compute_interaction_statistics(enriched)
            diagnostics["statistics"] = stats.to_dict()
            if verbose:
                print(f"   [regime_session] χ²={stats.chi_square:.2f}, "
                      f"V={stats.cramers_v:.3f} ({stats.effect_size_class})")
        except Exception as exc:
            diagnostics["statistics"] = {"error": str(exc)}

    return enriched, diagnostics
