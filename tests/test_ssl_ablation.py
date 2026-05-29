"""Tests for the SSL ablation comparison tool.

The tool has three testable layers:
  1. Per-fold metric extraction (handles nested/flat layouts)
  2. Pair discovery (fold by fold across two directories)
  3. Verdict cascade (paired t-stat + sign consistency)
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

from tools.ssl_ablation_comparison import (
    MIN_PAIRS,
    NEUTRAL_TOLERANCE,
    SIGN_CONSISTENCY_THRESHOLD,
    T_STAT_THRESHOLD,
    _extract_metric,
    build_per_fold_table,
    compute_verdict,
    discover_pairs,
    extract_per_fold_metrics,
    paired_t_statistic,
    run_comparison,
    sign_consistency,
)


# ════════════════════════════════════════════════════════════════════
# Metric extraction
# ════════════════════════════════════════════════════════════════════
class TestExtractMetric:
    def test_flat_key(self):
        assert _extract_metric({"hit_rate": 0.6}, "hit_rate") == 0.6

    def test_nested_in_hybrid(self):
        d = {"hybrid_filtered": {"hit_rate": 0.55}}
        assert _extract_metric(d, "hit_rate") == 0.55

    def test_missing_returns_default(self):
        assert _extract_metric({}, "hit_rate", default=-1.0) == -1.0

    def test_non_numeric_returns_default(self):
        assert _extract_metric({"hit_rate": "n/a"}, "hit_rate") == 0.0


class TestExtractPerFoldMetrics:
    def test_pulls_all_four(self):
        m = {
            "hit_rate": 0.6,
            "annualized_sharpe": 1.2,
            "lift_hit_rate": 0.05,
            "max_drawdown": -0.10,
        }
        out = extract_per_fold_metrics(m)
        assert out["hit_rate"] == 0.6
        assert out["sharpe"] == 1.2
        assert out["lift"] == 0.05
        assert out["max_drawdown"] == -0.10

    def test_sharpe_falls_back_to_alt_key(self):
        m = {"sharpe": 0.9}
        assert extract_per_fold_metrics(m)["sharpe"] == 0.9


# ════════════════════════════════════════════════════════════════════
# Pair discovery
# ════════════════════════════════════════════════════════════════════
class TestDiscoverPairs:
    def _write_fold(self, base: Path, name: str, metrics: dict):
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps(metrics))

    def test_matching_folds_paired(self, tmp_path):
        a = tmp_path / "with_ssl"
        b = tmp_path / "no_ssl"
        for name in ("fold_01", "fold_02", "fold_03"):
            self._write_fold(a, name, {"hit_rate": 0.6, "annualized_sharpe": 1.0})
            self._write_fold(b, name, {"hit_rate": 0.55, "annualized_sharpe": 0.8})
        pairs = discover_pairs(a, b)
        assert len(pairs) == 3
        assert [p[0] for p in pairs] == ["fold_01", "fold_02", "fold_03"]

    def test_only_intersection(self, tmp_path):
        a = tmp_path / "with"
        b = tmp_path / "without"
        self._write_fold(a, "fold_01", {"hit_rate": 0.6})
        self._write_fold(a, "fold_02", {"hit_rate": 0.6})
        self._write_fold(b, "fold_02", {"hit_rate": 0.5})
        self._write_fold(b, "fold_03", {"hit_rate": 0.5})
        pairs = discover_pairs(a, b)
        assert len(pairs) == 1
        assert pairs[0][0] == "fold_02"

    def test_missing_metrics_skipped(self, tmp_path):
        a = tmp_path / "with"
        b = tmp_path / "without"
        # fold_01 only has metrics in 'with' — skipped
        self._write_fold(a, "fold_01", {"hit_rate": 0.6})
        (b / "fold_01").mkdir(parents=True)   # exists but no metrics.json
        pairs = discover_pairs(a, b)
        assert len(pairs) == 0

    def test_missing_dirs_returns_empty(self, tmp_path):
        a = tmp_path / "does_not_exist_a"
        b = tmp_path / "does_not_exist_b"
        pairs = discover_pairs(a, b)
        assert pairs == []


# ════════════════════════════════════════════════════════════════════
# Statistics primitives
# ════════════════════════════════════════════════════════════════════
class TestPairedT:
    def test_known_t_stat(self):
        # Mean=2, std=√(4/3)≈1.155 (ddof=1), n=4 → t = 2 / (1.155/2) ≈ 3.464
        t = paired_t_statistic([1.0, 1.0, 3.0, 3.0])
        assert t == pytest.approx(3.464, abs=0.01)

    def test_too_few_returns_zero(self):
        assert paired_t_statistic([1.0]) == 0.0
        assert paired_t_statistic([]) == 0.0

    def test_zero_variance_returns_zero(self):
        assert paired_t_statistic([2.0, 2.0, 2.0, 2.0]) == 0.0


class TestSignConsistency:
    def test_all_same_sign_returns_one(self):
        assert sign_consistency([1.0, 2.0, 3.0]) == 1.0

    def test_mixed_signs(self):
        # mean positive (1.0), 3 positive / 1 negative → 75 %
        c = sign_consistency([2.0, 1.0, 3.0, -1.0])
        assert c == pytest.approx(0.75)

    def test_empty_returns_zero(self):
        assert sign_consistency([]) == 0.0


# ════════════════════════════════════════════════════════════════════
# Verdict cascade
# ════════════════════════════════════════════════════════════════════
class TestVerdict:
    def test_insufficient_pairs(self):
        v, _ = compute_verdict([0.5])
        assert v == "INSUFFICIENT_PAIRS"

    def test_ssl_useful(self):
        # 5 folds, all positive, mean well above tolerance
        deltas = [0.20, 0.25, 0.30, 0.22, 0.28]
        v, diag = compute_verdict(deltas)
        assert v == "SSL_USEFUL"
        assert diag["mean_delta_sharpe"] > 0.20
        assert diag["sign_consistency"] == 1.0

    def test_ssl_neutral_small_mean(self):
        deltas = [0.01, -0.02, 0.01, -0.01, 0.02]
        v, _ = compute_verdict(deltas)
        assert v == "SSL_NEUTRAL"

    def test_ssl_neutral_high_variance(self):
        # Large mean but inconsistent across folds → low t-stat
        deltas = [1.0, -0.8, 1.1, -0.9, 0.9]
        v, diag = compute_verdict(deltas)
        # |t| likely small relative to threshold
        assert v == "SSL_NEUTRAL"

    def test_ssl_harmful(self):
        deltas = [-0.20, -0.25, -0.30, -0.22, -0.28]
        v, _ = compute_verdict(deltas)
        assert v == "SSL_HARMFUL"


# ════════════════════════════════════════════════════════════════════
# Per-fold table
# ════════════════════════════════════════════════════════════════════
class TestPerFoldTable:
    def test_columns_and_deltas(self):
        pairs = [
            ("fold_01",
             {"hit_rate": 0.60, "annualized_sharpe": 1.50, "lift_hit_rate": 0.05},
             {"hit_rate": 0.55, "annualized_sharpe": 1.20, "lift_hit_rate": 0.02}),
        ]
        df = build_per_fold_table(pairs)
        assert df.shape == (1, 13)
        row = df.iloc[0]
        assert row["delta_sharpe"] == pytest.approx(0.30, abs=1e-6)
        assert row["delta_hit"] == pytest.approx(0.05, abs=1e-6)
        assert row["delta_lift"] == pytest.approx(0.03, abs=1e-6)


# ════════════════════════════════════════════════════════════════════
# End-to-end orchestration
# ════════════════════════════════════════════════════════════════════
class TestRunComparison:
    def _seed_pair(
        self, tmp_path: Path, with_deltas: list[float],
    ) -> tuple[Path, Path]:
        """Make synthetic with/without fold dirs where:
              delta_sharpe = with - without
              so we control the verdict by varying `with_deltas`."""
        a = tmp_path / "with_ssl"
        b = tmp_path / "no_ssl"
        baseline = 1.0
        for i, delta in enumerate(with_deltas):
            for name, val in [(a, baseline + delta), (b, baseline)]:
                d = name / f"fold_{i+1:02d}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "metrics.json").write_text(json.dumps({
                    "hit_rate": 0.5,
                    "annualized_sharpe": val,
                    "lift_hit_rate": 0.0,
                    "max_drawdown": -0.1,
                }))
        return a, b

    def test_useful_verdict_end_to_end(self, tmp_path):
        a, b = self._seed_pair(tmp_path, [0.30, 0.25, 0.35, 0.28, 0.32])
        out = tmp_path / "ablation"
        summary = run_comparison(a, b, out)
        assert summary["verdict"] == "SSL_USEFUL"
        assert (out / "ssl_ablation_summary.json").exists()
        assert (out / "ssl_ablation_report.txt").exists()
        assert (out / "per_fold_deltas.csv").exists()

    def test_harmful_verdict_end_to_end(self, tmp_path):
        a, b = self._seed_pair(tmp_path, [-0.30, -0.25, -0.35, -0.28, -0.32])
        out = tmp_path / "ablation"
        summary = run_comparison(a, b, out)
        assert summary["verdict"] == "SSL_HARMFUL"

    def test_insufficient_pairs(self, tmp_path):
        a, b = self._seed_pair(tmp_path, [0.30])  # only 1 fold
        out = tmp_path / "ablation"
        summary = run_comparison(a, b, out)
        assert summary["verdict"] == "INSUFFICIENT_PAIRS"


# ════════════════════════════════════════════════════════════════════
# End-to-end CLI subprocess
# ════════════════════════════════════════════════════════════════════
class TestCLI:
    def test_cli_runs(self, tmp_path):
        # Seed 4 folds with positive deltas
        for i in range(1, 5):
            for sub, sharpe in [("with_ssl", 1.3), ("no_ssl", 1.0)]:
                d = tmp_path / sub / f"fold_{i:02d}"
                d.mkdir(parents=True)
                (d / "metrics.json").write_text(json.dumps({
                    "annualized_sharpe": sharpe, "hit_rate": 0.5,
                    "lift_hit_rate": 0.0, "max_drawdown": -0.1,
                }))
        out = tmp_path / "ablation"
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "ssl_ablation_comparison.py"),
             "--with-ssl-dir", str(tmp_path / "with_ssl"),
             "--no-ssl-dir", str(tmp_path / "no_ssl"),
             "--output", str(out)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"failed:\n{result.stdout}\n{result.stderr}"
        )
        assert "verdict" in result.stdout
        summary = json.loads((out / "ssl_ablation_summary.json").read_text())
        assert summary["diag"]["n_pairs"] == 4
