"""
tests/test_regime_session_interaction.py — Sprint 12 tests.

يحرس regime × session interaction (Regime Analysis Report v2 ②).

تغطية:
    1. API correctness (shape, types, values)
    2. Statistical validity (chi-square, Cramér's V)
    3. Edge cases (rare combinations, empty df, missing cols)
    4. Monte Carlo stability
    5. Integration مع feature_enrichment
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _make_synthetic_df(
    n: int = 1000,
    seed: int = 42,
    embed_interaction: bool = True,
) -> pd.DataFrame:
    """يولّد DataFrame مع regime + session + (اختياري) interaction signal.

    لو embed_interaction=True:
        - trending_x_LONDON: 70% LONG
        - trending_x_ASIA: 30% LONG (counter)
        - ranging_x_*: 50%
    """
    rng = np.random.RandomState(seed)
    regimes = rng.choice(["trending", "ranging", "volatile"], n, p=[0.6, 0.35, 0.05])
    sessions = rng.choice(
        ["LONDON_Q1", "LONDON_Q2", "ASIA_Q1", "ASIA_Q2", "NY_Q1", "NY_Q2"],
        n,
    )

    if embed_interaction:
        # Strong signal: trending_x_LONDON → LONG, trending_x_ASIA → SHORT
        bias = []
        for r, s in zip(regimes, sessions):
            if r == "trending" and "LONDON" in s:
                p = rng.choice([1, -1, 0], p=[0.70, 0.15, 0.15])
            elif r == "trending" and "ASIA" in s:
                p = rng.choice([1, -1, 0], p=[0.20, 0.65, 0.15])
            elif r == "ranging":
                p = rng.choice([1, -1, 0], p=[0.30, 0.30, 0.40])
            else:
                p = rng.choice([1, -1, 0])
            bias.append(p)
        bias = np.array(bias)
    else:
        bias = rng.choice([1, -1, 0], n)

    return pd.DataFrame({
        "regime_label": regimes,
        "zone_full": sessions,
        "bias_label": bias,
    })


class TestBuildLabel(unittest.TestCase):
    """API: build_regime_session_label."""

    def test_basic_concatenation(self):
        from modules.regime_session_interaction import build_regime_session_label
        df = pd.DataFrame({
            "regime_label": ["trending", "ranging"],
            "zone_full": ["LONDON_Q1", "ASIA_Q2"],
        })
        out = build_regime_session_label(df)
        self.assertEqual(out.tolist(),
                         ["trending_x_LONDON_Q1", "ranging_x_ASIA_Q2"])

    def test_custom_separator(self):
        from modules.regime_session_interaction import build_regime_session_label
        df = pd.DataFrame({
            "regime_label": ["trending"],
            "zone_full": ["LONDON"],
        })
        out = build_regime_session_label(df, separator="||")
        self.assertEqual(out.iloc[0], "trending||LONDON")

    def test_nan_handling(self):
        from modules.regime_session_interaction import build_regime_session_label
        df = pd.DataFrame({
            "regime_label": ["trending", None],
            "zone_full": [None, "LONDON"],
        })
        out = build_regime_session_label(df, na_placeholder="MISSING")
        self.assertIn("MISSING", out.iloc[0])
        self.assertIn("MISSING", out.iloc[1])

    def test_raises_on_missing_column(self):
        from modules.regime_session_interaction import build_regime_session_label
        df = pd.DataFrame({"regime_label": ["a"]})
        with self.assertRaises(KeyError):
            build_regime_session_label(df, session_col="missing_col")


class TestSupportValidation(unittest.TestCase):
    """validate_combination_support."""

    def test_no_rare_when_uniform(self):
        from modules.regime_session_interaction import validate_combination_support
        df = _make_synthetic_df(n=2000, seed=0, embed_interaction=False)
        result = validate_combination_support(df, min_support=10)
        self.assertEqual(result["n_total"], 2000)
        self.assertGreater(result["n_combinations"], 5)
        # min_support=10 + 2000 rows → معظم الـ combos فوق 10
        self.assertEqual(len(result["rare_combinations"]), 0)

    def test_rare_detected(self):
        from modules.regime_session_interaction import validate_combination_support
        df = pd.DataFrame({
            "regime_label": ["a"] * 100 + ["b"] * 5,
            "zone_full": ["x"] * 100 + ["y"] * 5,
        })
        result = validate_combination_support(df, min_support=50)
        self.assertIn("b_x_y", result["rare_combinations"])

    def test_coverage_pct(self):
        from modules.regime_session_interaction import validate_combination_support
        df = pd.DataFrame({
            "regime_label": ["a"] * 90 + ["b"] * 10,
            "zone_full": ["x"] * 90 + ["y"] * 10,
        })
        result = validate_combination_support(df, min_support=50)
        # only a_x_x (90 rows) passes
        self.assertAlmostEqual(result["coverage_pct"], 90.0, places=1)

    def test_auto_min_support(self):
        from modules.regime_session_interaction import validate_combination_support
        df = _make_synthetic_df(n=10000, seed=1)
        result = validate_combination_support(df, min_support=None)
        # auto = max(200, 0.5% × 10000) = max(200, 50) = 200
        self.assertEqual(result["min_support"], 200)


class TestStatistics(unittest.TestCase):
    """chi-square + Cramér's V."""

    def test_detects_strong_interaction(self):
        from modules.regime_session_interaction import compute_interaction_statistics
        df = _make_synthetic_df(n=5000, seed=42, embed_interaction=True)
        stats = compute_interaction_statistics(df)

        # Strong embedded signal → V should be moderate-to-strong
        self.assertGreater(stats.cramers_v, 0.10,
            f"Embedded interaction must produce Cramér's V > 0.10, got {stats.cramers_v}")
        self.assertLess(stats.p_value, 0.05,
            f"Embedded interaction must be statistically significant, p={stats.p_value}")
        self.assertIn(stats.effect_size_class, ["weak", "moderate", "strong"])

    def test_negligible_for_random_data(self):
        from modules.regime_session_interaction import compute_interaction_statistics
        # totally random — no embedded interaction
        rng = np.random.RandomState(7)
        n = 5000
        df = pd.DataFrame({
            "regime_label": rng.choice(["a", "b", "c"], n),
            "zone_full": rng.choice(["x", "y", "z"], n),
            "bias_label": rng.choice([1, -1, 0], n),
        })
        stats = compute_interaction_statistics(df)
        # Random data → V should be small
        self.assertLess(stats.cramers_v, 0.15,
            f"Random data should give weak Cramér's V, got {stats.cramers_v}")

    def test_contingency_shape(self):
        from modules.regime_session_interaction import compute_interaction_statistics
        df = _make_synthetic_df(n=1000, seed=0)
        stats = compute_interaction_statistics(df)
        # rows = unique regime_x_session combos, cols = unique bias values
        n_unique_combos = len(df.groupby(["regime_label", "zone_full"]))
        self.assertEqual(stats.contingency_shape[0], n_unique_combos)
        self.assertEqual(stats.contingency_shape[1], df["bias_label"].nunique())

    def test_degrees_of_freedom(self):
        from modules.regime_session_interaction import compute_interaction_statistics
        df = _make_synthetic_df(n=1000, seed=0)
        stats = compute_interaction_statistics(df)
        # dof = (rows-1) × (cols-1)
        expected_dof = (stats.contingency_shape[0] - 1) * (stats.contingency_shape[1] - 1)
        self.assertEqual(stats.degrees_of_freedom, expected_dof)


class TestAddFeatures(unittest.TestCase):
    """add_regime_session_features."""

    def test_adds_categorical_column(self):
        from modules.regime_session_interaction import add_regime_session_features
        df = _make_synthetic_df(n=500, seed=0)
        result = add_regime_session_features(df)
        self.assertIn("regime_session", result.columns)
        # Categorical for CatBoost
        self.assertTrue(isinstance(result["regime_session"].dtype,
                                   pd.CategoricalDtype))

    def test_buckets_rare_to_other(self):
        from modules.regime_session_interaction import (
            add_regime_session_features, RARE_LABEL,
        )
        df = pd.DataFrame({
            "regime_label": ["common"] * 200 + ["rare"] * 5,
            "zone_full":    ["x"] * 200 + ["y"] * 5,
        })
        result = add_regime_session_features(
            df, min_support=50, bucket_rare=True,
        )
        # rare_x_y → "other"
        self.assertIn(RARE_LABEL, result["regime_session"].cat.categories)

    def test_no_bucketing_when_disabled(self):
        from modules.regime_session_interaction import add_regime_session_features
        df = pd.DataFrame({
            "regime_label": ["a"] * 100 + ["b"] * 1,
            "zone_full": ["x"] * 100 + ["y"] * 1,
        })
        result = add_regime_session_features(
            df, min_support=10, bucket_rare=False,
        )
        # rare combo preserved
        self.assertIn("b_x_y", result["regime_session"].cat.categories)

    def test_preserves_original_when_not_inplace(self):
        from modules.regime_session_interaction import add_regime_session_features
        df = _make_synthetic_df(n=100, seed=0)
        orig_cols = list(df.columns)
        _ = add_regime_session_features(df, inplace=False)
        self.assertEqual(list(df.columns), orig_cols)


class TestMonteCarloStability(unittest.TestCase):
    def test_stable_for_strong_signal(self):
        from modules.regime_session_interaction import monte_carlo_interaction_stability
        df = _make_synthetic_df(n=3000, seed=42, embed_interaction=True)
        result = monte_carlo_interaction_stability(
            df, n_iterations=30, bootstrap_frac=0.7, seed=0,
        )
        # Strong signal → std < 0.15
        self.assertLess(result["std"], 0.15,
            f"Strong signal should be stable, got std={result['std']}")
        self.assertGreater(result["n_successful"], 25)

    def test_ci_bounds_valid(self):
        from modules.regime_session_interaction import monte_carlo_interaction_stability
        df = _make_synthetic_df(n=1500, seed=0)
        result = monte_carlo_interaction_stability(
            df, n_iterations=30, seed=1,
        )
        self.assertLessEqual(result["ci_95_lo"], result["median"])
        self.assertLessEqual(result["median"], result["ci_95_hi"])
        self.assertGreaterEqual(result["mean"], 0.0)
        self.assertLessEqual(result["mean"], 1.0)


class TestEnrichWithRegimeSession(unittest.TestCase):
    """High-level integration helper."""

    def test_returns_diagnostics(self):
        from modules.regime_session_interaction import enrich_with_regime_session
        df = _make_synthetic_df(n=500, seed=0)
        enriched, diag = enrich_with_regime_session(df)
        self.assertIn("regime_session", enriched.columns)
        self.assertIn("support", diag)
        self.assertIn("statistics", diag)
        self.assertGreaterEqual(diag["statistics"]["cramers_v"], 0.0)

    def test_skips_when_missing_cols(self):
        from modules.regime_session_interaction import enrich_with_regime_session
        df = pd.DataFrame({"some_col": [1, 2, 3]})
        enriched, diag = enrich_with_regime_session(df)
        self.assertTrue(diag.get("skipped"))
        # original df unchanged
        self.assertEqual(list(enriched.columns), ["some_col"])


class TestFeatureEnrichmentIntegration(unittest.TestCase):
    """التأكد إن feature_enrichment.py يضيف regime_session تلقائياً."""

    def test_enrichment_config_has_flag(self):
        from modules.feature_enrichment import EnrichmentConfig
        cfg = EnrichmentConfig()
        # default = True
        self.assertTrue(cfg.add_regime_session)

    def test_enrichment_adds_column_when_present(self):
        from modules.feature_enrichment import enrich_features, EnrichmentConfig

        # synthetic frame مع regime + zone (لا نشغّل V19.2 simulators)
        df = _make_synthetic_df(n=300, seed=0)
        cfg = EnrichmentConfig(
            add_simulators=False, add_wall_depth=False, add_iceberg=False,
            add_session_mapping=False, add_bell_pairs=False,
            add_regime_session=True,
            verbose=False,
        )
        result = enrich_features(df, config=cfg)
        self.assertIn("regime_session", result.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
