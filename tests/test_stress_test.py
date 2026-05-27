"""Tests for the stress test tool.

Verifies:
  • Scenario library is well-formed (correct multipliers, descriptions)
  • extract_key_metrics handles both shapes (by_slice + all_events)
  • compute_verdict classifies all 4 strategy types correctly
  • Full subprocess invocation runs successfully on synthetic data
    and produces the expected output files
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

from tools.diagnostics.stress_test_backtest import (
    DEFAULT_SCENARIOS,
    StressScenario,
    compute_verdict,
    extract_key_metrics,
)


# ════════════════════════════════════════════════════════════════════
# Scenario library
# ════════════════════════════════════════════════════════════════════
class TestScenarioLibrary:
    def test_five_default_scenarios(self):
        assert set(DEFAULT_SCENARIOS.keys()) == {
            "baseline", "mild", "moderate", "severe", "extreme",
        }

    def test_slippage_monotonically_increases(self):
        """Each scenario should be at least as stressful as the previous."""
        order = ["baseline", "mild", "moderate", "severe", "extreme"]
        prev = -1.0
        for name in order:
            s = DEFAULT_SCENARIOS[name]
            assert s.slippage_multiplier >= prev
            prev = s.slippage_multiplier

    def test_latency_monotonically_increases(self):
        order = ["baseline", "mild", "moderate", "severe", "extreme"]
        prev = -1
        for name in order:
            s = DEFAULT_SCENARIOS[name]
            assert s.extra_latency_bars >= prev
            prev = s.extra_latency_bars

    def test_baseline_is_truly_baseline(self):
        b = DEFAULT_SCENARIOS["baseline"]
        assert b.slippage_multiplier == 1.0
        assert b.extra_latency_bars == 0
        assert b.stop_slippage_pips == 0.0


# ════════════════════════════════════════════════════════════════════
# extract_key_metrics
# ════════════════════════════════════════════════════════════════════
class TestExtractKeyMetrics:
    def test_prefers_holdout_slice_when_present(self):
        report = {
            "all_events": {"total_pnl_dollars": 1000, "n_trades": 100},
            "by_slice": {
                "holdout": {"total_pnl_dollars": 200, "n_trades": 25},
            },
        }
        m = extract_key_metrics(report)
        assert m["total_pnl_dollars"] == 200
        assert m["n_trades"] == 25

    def test_falls_back_to_all_events(self):
        report = {
            "all_events": {"total_pnl_dollars": 1000, "n_trades": 100},
        }
        m = extract_key_metrics(report)
        assert m["total_pnl_dollars"] == 1000

    def test_missing_keys_default_to_zero(self):
        report = {"all_events": {"total_pnl_dollars": 500}}
        m = extract_key_metrics(report)
        assert m["n_trades"] == 0
        assert m["hit_rate"] == 0.0
        assert m["annualized_sharpe"] == 0.0


# ════════════════════════════════════════════════════════════════════
# compute_verdict
# ════════════════════════════════════════════════════════════════════
class TestVerdict:
    def test_robust_strategy(self):
        per_scen = {
            "baseline": {"total_pnl_dollars": 5000, "annualized_sharpe": 3.0},
            "moderate": {"total_pnl_dollars": 3000, "annualized_sharpe": 2.0},
            "severe":   {"total_pnl_dollars": 1500, "annualized_sharpe": 1.0},
        }
        verdict = compute_verdict(per_scen)
        assert verdict.startswith("ROBUST")

    def test_fragile_strategy(self):
        per_scen = {
            "baseline": {"total_pnl_dollars": 5000, "annualized_sharpe": 3.0},
            "moderate": {"total_pnl_dollars": -500, "annualized_sharpe": -0.5},
            "severe":   {"total_pnl_dollars": -3000, "annualized_sharpe": -2.0},
        }
        verdict = compute_verdict(per_scen)
        assert verdict.startswith("FRAGILE")

    def test_broken_strategy(self):
        per_scen = {
            "baseline": {"total_pnl_dollars": -200, "annualized_sharpe": -0.3},
            "moderate": {"total_pnl_dollars": -800, "annualized_sharpe": -1.5},
        }
        verdict = compute_verdict(per_scen)
        assert verdict.startswith("BROKEN")

    def test_acceptable_strategy(self):
        per_scen = {
            "baseline": {"total_pnl_dollars": 5000, "annualized_sharpe": 3.0},
            "moderate": {"total_pnl_dollars": 1500, "annualized_sharpe": 1.5},
            "severe":   {"total_pnl_dollars": -200, "annualized_sharpe": -0.1},
        }
        verdict = compute_verdict(per_scen)
        assert verdict.startswith("ACCEPTABLE")


# ════════════════════════════════════════════════════════════════════
# End-to-end subprocess (uses synthetic features parquet)
# ════════════════════════════════════════════════════════════════════
class TestEndToEnd:
    def _make_features_with_events(self, tmp_path):
        """Build a synthetic features parquet matching the schema the
        backtest expects (event_flag, event_direction, atr_14, OHLC,
        dataset_slice for holdout-only mode)."""
        rng = np.random.RandomState(7)
        n = 2000
        ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz=None)
        # Generate a trending price series + some noise so events have a
        # plausible chance of hitting TP
        drift = np.cumsum(rng.randn(n) * 0.0005)
        close = 1.30 + drift
        df = pd.DataFrame({
            "ts_event": ts,
            "open":  close - 0.0001 * rng.rand(n),
            "high":  close + 0.0005 * (1 + rng.rand(n)),
            "low":   close - 0.0005 * (1 + rng.rand(n)),
            "close": close,
            "atr_14": np.full(n, 0.001, dtype=np.float32),
            "event_flag": rng.choice([0, 1], n, p=[0.92, 0.08]).astype("int8"),
            "event_direction": rng.choice([-1, 1], n, p=[0.5, 0.5]).astype("int8"),
            "regime_label": np.array(["trending"] * n),
            "signal_quality": np.full(n, 1, dtype="int8"),
            "label_horizon_steps": np.full(n, 6, dtype=np.int64),
            "dataset_slice": np.array(
                ["train"] * (n * 3 // 4) + ["holdout"] * (n - n * 3 // 4)
            ),
        })
        p = tmp_path / "features.parquet"
        df.to_parquet(p, index=False)
        return p

    def test_subset_run_produces_summary(self, tmp_path):
        feat = self._make_features_with_events(tmp_path)
        out = tmp_path / "stress_out"
        result = subprocess.run(
            [sys.executable,
             str(REPO_ROOT / "tools" / "diagnostics" / "stress_test_backtest.py"),
             "--features", str(feat),
             "--output", str(out),
             "--scenarios", "baseline", "moderate"],  # subset for speed
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"stress test failed:\n{result.stdout}\n{result.stderr}"
        )
        summary_path = out / "stress_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert len(summary["scenarios"]) == 2
        scenario_names = [s["name"] for s in summary["scenarios"]]
        assert scenario_names == ["baseline", "moderate"]
        # Verdict is one of the 5 categories
        assert any(
            summary["verdict"].startswith(prefix)
            for prefix in (
                "ROBUST", "ACCEPTABLE", "FRAGILE", "MARGINAL", "BROKEN", "INCOMPLETE",
            )
        )
        # Human-readable report exists
        assert (out / "stress_report.txt").exists()

    def test_unknown_scenario_raises(self, tmp_path):
        feat = self._make_features_with_events(tmp_path)
        out = tmp_path / "stress_out"
        result = subprocess.run(
            [sys.executable,
             str(REPO_ROOT / "tools" / "diagnostics" / "stress_test_backtest.py"),
             "--features", str(feat),
             "--output", str(out),
             "--scenarios", "ghost_scenario"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "Unknown scenarios" in result.stderr
