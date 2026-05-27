"""Tests for the feature-redundancy audit tool.

Each test constructs a synthetic parquet with KNOWN redundancy structure
and verifies the audit detects it. This is the empirical test bed for
Issue #1 from docs/PIPELINE_ISSUES_AUDIT.md (Feature Noise / Redundancy).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.audit_feature_redundancy import (
    _is_leakage,
    _select_audit_columns,
    audit,
    compute_correlation_matrix,
    find_redundancy_clusters,
    pick_cluster_representative,
)


# ════════════════════════════════════════════════════════════════════
# Unit tests
# ════════════════════════════════════════════════════════════════════
class TestLeakageDetection:
    def test_known_leakage_patterns_flagged(self):
        assert _is_leakage("event_flag")
        assert _is_leakage("event_direction")
        assert _is_leakage("path_outcome")
        assert _is_leakage("target_ret_24")
        assert _is_leakage("dataset_slice")

    def test_normal_features_not_flagged(self):
        assert not _is_leakage("close")
        assert not _is_leakage("atr_14")
        assert not _is_leakage("obi")
        assert not _is_leakage("cvd_slope_1h")


class TestSelectAuditColumns:
    def test_drops_string_and_leakage(self):
        df = pd.DataFrame({
            "ts_event": pd.to_datetime(["2025-01-01"]),
            "close": [1.30],
            "atr_14": [0.005],
            "event_flag": [1],          # leakage
            "regime_label": ["trending"],  # string, dropped
        })
        cols = _select_audit_columns(df)
        assert "close" in cols
        assert "atr_14" in cols
        assert "event_flag" not in cols
        assert "regime_label" not in cols
        assert "ts_event" not in cols


# ════════════════════════════════════════════════════════════════════
# Correlation matrix
# ════════════════════════════════════════════════════════════════════
class TestCorrelationMatrix:
    def test_identity_on_diagonal(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({
            "a": rng.randn(200), "b": rng.randn(200), "c": rng.randn(200),
        })
        m = compute_correlation_matrix(df, ["a", "b", "c"])
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-9)

    def test_perfect_correlation_detected(self):
        rng = np.random.RandomState(0)
        a = rng.randn(200)
        df = pd.DataFrame({
            "a": a, "b": a, "c": rng.randn(200),
        })
        m = compute_correlation_matrix(df, ["a", "b", "c"])
        assert m.loc["a", "b"] > 0.99
        assert m.loc["a", "c"] < 0.5

    def test_constant_column_dropped(self):
        df = pd.DataFrame({
            "a": np.linspace(0, 1, 100),
            "constant": np.zeros(100),    # constant — should be dropped
            "b": np.linspace(1, 0, 100),
        })
        m = compute_correlation_matrix(df, ["a", "constant", "b"])
        assert "constant" not in m.columns


# ════════════════════════════════════════════════════════════════════
# Cluster detection
# ════════════════════════════════════════════════════════════════════
class TestRedundancyClusters:
    def test_two_independent_clusters_detected(self):
        """Build a 6-feature df with two groups of 3 features each;
        within each group: r ≈ 0.99. Between groups: r ≈ 0."""
        rng = np.random.RandomState(0)
        n = 500
        signal_A = rng.randn(n)
        signal_B = rng.randn(n)
        df = pd.DataFrame({
            "a1": signal_A + 0.01 * rng.randn(n),
            "a2": signal_A + 0.01 * rng.randn(n),
            "a3": signal_A + 0.01 * rng.randn(n),
            "b1": signal_B + 0.01 * rng.randn(n),
            "b2": signal_B + 0.01 * rng.randn(n),
            "b3": signal_B + 0.01 * rng.randn(n),
        })
        corr = compute_correlation_matrix(df, list(df.columns))
        clusters = find_redundancy_clusters(corr, threshold=0.95)
        assert len(clusters) == 2
        cluster_sets = [set(c) for c in clusters]
        assert {"a1", "a2", "a3"} in cluster_sets
        assert {"b1", "b2", "b3"} in cluster_sets

    def test_no_redundancy_returns_no_clusters(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({c: rng.randn(500) for c in "abcde"})
        corr = compute_correlation_matrix(df, list(df.columns))
        clusters = find_redundancy_clusters(corr, threshold=0.95)
        assert clusters == []

    def test_threshold_controls_sensitivity(self):
        rng = np.random.RandomState(0)
        n = 500
        base = rng.randn(n)
        df = pd.DataFrame({
            "tight1": base + 0.005 * rng.randn(n),  # r ≈ 0.999
            "tight2": base + 0.005 * rng.randn(n),
            "loose1": base + 0.5 * rng.randn(n),    # r ≈ 0.9
            "loose2": base + 0.5 * rng.randn(n),
        })
        corr = compute_correlation_matrix(df, list(df.columns))
        # At threshold 0.99 only tight1/tight2 cluster
        tight_clusters = find_redundancy_clusters(corr, threshold=0.99)
        assert any(set(c) == {"tight1", "tight2"} for c in tight_clusters)
        # At threshold 0.85 all four cluster together
        loose_clusters = find_redundancy_clusters(corr, threshold=0.85)
        assert any(set(c) == {"tight1", "tight2", "loose1", "loose2"}
                   for c in loose_clusters)


# ════════════════════════════════════════════════════════════════════
# Representative selection
# ════════════════════════════════════════════════════════════════════
class TestRepresentativeSelection:
    def test_no_target_returns_first_member(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        cluster = ["b", "a"]  # alphabetical sort would put 'a' first
        rep = pick_cluster_representative(sorted(cluster), df, target_col=None)
        assert rep == "a"

    def test_target_picks_most_correlated(self):
        rng = np.random.RandomState(0)
        n = 500
        target = rng.randn(n)
        df = pd.DataFrame({
            "weak_signal": target + 2.0 * rng.randn(n),   # r ≈ 0.45
            "strong_signal": target + 0.1 * rng.randn(n),  # r ≈ 0.995
            "target_ret_24": target,
        })
        rep = pick_cluster_representative(
            ["strong_signal", "weak_signal"], df, target_col="target_ret_24",
        )
        assert rep == "strong_signal"


# ════════════════════════════════════════════════════════════════════
# Full pipeline + CLI
# ════════════════════════════════════════════════════════════════════
class TestFullAudit:
    def _make_synthetic_redundant_df(self) -> pd.DataFrame:
        """Mimic the day_trade situation: many CVD variants of the same
        underlying signal, plus a few independent features."""
        rng = np.random.RandomState(42)
        n = 1000
        cvd_base = rng.randn(n)
        atr_base = rng.randn(n)
        return pd.DataFrame({
            "ts_event": pd.date_range("2025-01-01", periods=n, freq="15min"),
            # CVD family (5 correlated variants)
            "cvd": cvd_base,
            "cvd_momentum": cvd_base + 0.02 * rng.randn(n),
            "cvd_slope_1h": cvd_base + 0.02 * rng.randn(n),
            "cvd_slope_4h": cvd_base + 0.02 * rng.randn(n),
            "session_cvd": cvd_base + 0.02 * rng.randn(n),
            # ATR family (4 correlated variants)
            "atr_14": atr_base,
            "atr_intrabar": atr_base + 0.03 * rng.randn(n),
            "atr_z_score": atr_base + 0.03 * rng.randn(n),
            "micro_atr": atr_base + 0.03 * rng.randn(n),
            # Independent features
            "close": rng.randn(n) * 0.005 + 1.30,
            "session_phase": rng.randn(n),
            # Leakage column (should be ignored)
            "event_flag": rng.choice([0, 1], n),
        })

    def test_audit_finds_two_known_clusters(self, tmp_path):
        df = self._make_synthetic_redundant_df()
        parquet_path = tmp_path / "synthetic.parquet"
        df.to_parquet(parquet_path, index=False)

        result = audit(
            features_path=parquet_path,
            output_dir=tmp_path / "audit_out",
            threshold=0.90,
        )
        # Expect 2 clusters (CVD family + ATR family) and a non-zero drop list
        assert result["n_redundancy_clusters"] == 2
        # CVD family has 5 members, ATR has 4 → total 9 redundant
        assert result["n_redundant_features"] == 9
        # Drop 7 (keep 1 per cluster, so drop n - 2)
        assert result["n_to_drop"] == 7
        # Leakage column never appears
        assert "event_flag" not in result["drop_list"]
        for cluster in result["clusters"]:
            assert "event_flag" not in cluster["members"]

    def test_audit_writes_all_three_outputs(self, tmp_path):
        df = self._make_synthetic_redundant_df()
        parquet_path = tmp_path / "x.parquet"
        df.to_parquet(parquet_path, index=False)
        out = tmp_path / "audit"
        audit(features_path=parquet_path, output_dir=out, threshold=0.9)
        assert (out / "correlation_matrix.csv").exists()
        assert (out / "redundancy_summary.json").exists()
        assert (out / "redundancy_report.txt").exists()

    def test_cli_subprocess(self, tmp_path):
        df = self._make_synthetic_redundant_df()
        parquet_path = tmp_path / "x.parquet"
        df.to_parquet(parquet_path, index=False)
        out = tmp_path / "audit"
        result = subprocess.run(
            [sys.executable,
             str(REPO_ROOT / "tools" / "audit_feature_redundancy.py"),
             "--features", str(parquet_path),
             "--output", str(out),
             "--threshold", "0.9"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        summary = json.loads((out / "redundancy_summary.json").read_text())
        assert summary["n_redundancy_clusters"] == 2
