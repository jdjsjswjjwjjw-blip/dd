"""Tests for the regime parity test tool.

Each test crafts a synthetic trades.csv with KNOWN per-regime
characteristics and verifies the tool's metrics + verdict logic
classifies it correctly.
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

from tools.diagnostics.regime_parity_test import (
    DEFAULT_MIN_TRADES,
    _compute_max_drawdown,
    _safe_sharpe,
    add_volatility_quartile,
    compute_cv,
    compute_verdict,
    per_regime_metrics,
)


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════
class TestSafeSharpe:
    def test_zero_trades_returns_zero(self):
        assert _safe_sharpe(np.array([]), 0, None, None) == 0.0

    def test_constant_pnl_returns_zero(self):
        pnl = np.full(100, 10.0)
        assert _safe_sharpe(pnl, 100, None, None) == 0.0

    def test_positive_pnl_with_variance_positive_sharpe(self):
        rng = np.random.RandomState(0)
        pnl = rng.randn(100) + 0.5   # positive mean
        sh = _safe_sharpe(pnl, 100, None, None)
        assert sh > 0

    def test_uses_timestamps_when_available(self):
        rng = np.random.RandomState(0)
        pnl = rng.randn(100) + 0.5
        ts0 = pd.Timestamp("2024-01-01")
        ts1 = pd.Timestamp("2024-12-31")
        sh = _safe_sharpe(pnl, 100, ts0, ts1)
        assert sh > 0


class TestMaxDrawdown:
    def test_monotone_increasing_returns_zero(self):
        cum = np.array([0.0, 100, 200, 300])
        dd, dd_pct = _compute_max_drawdown(cum)
        assert dd == 0.0

    def test_peak_to_trough(self):
        # Goes up to 1000 then down to 500
        cum = np.array([0.0, 500, 1000, 700, 500])
        dd, dd_pct = _compute_max_drawdown(cum)
        assert dd == -500
        assert dd_pct == pytest.approx(-0.5, abs=0.01)  # -500 / 100k * 100


class TestComputeCV:
    def test_constant_values_cv_zero(self):
        assert compute_cv([5.0, 5.0, 5.0]) == 0.0

    def test_with_nan_skipped(self):
        # Should compute on the non-NaN subset
        cv1 = compute_cv([1.0, 2.0, 3.0])
        cv2 = compute_cv([1.0, 2.0, 3.0, float("nan")])
        assert cv1 == cv2

    def test_zero_mean_returns_inf(self):
        # Symmetric around zero
        assert compute_cv([-1.0, 1.0]) == float("inf")


# ════════════════════════════════════════════════════════════════════
# per_regime_metrics
# ════════════════════════════════════════════════════════════════════
class TestPerRegimeMetrics:
    def _trades(self, regime: str, n: int, win_pnl: float, loss_pnl: float,
                p_win: float, seed: int = 0) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        wins = rng.rand(n) < p_win
        pnl = np.where(wins, win_pnl, loss_pnl)
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        return pd.DataFrame({
            "ts_entry": ts,
            "ts_exit": ts + pd.Timedelta("15min"),
            "regime": [regime] * n,
            "net_pnl": pnl,
            "pips": pnl / 6.25,    # 6B tick value
            "atr_at_entry": np.full(n, 0.005),
            "entry_price": np.full(n, 1.30),
        })

    def test_two_regimes_aggregated_correctly(self):
        df = pd.concat([
            self._trades("trending", 100, 50, -30, 0.7, seed=1),
            self._trades("ranging",  50, 30, -40, 0.5, seed=2),
        ], ignore_index=True)
        m = per_regime_metrics(df)
        assert set(m["regime"]) == {"trending", "ranging"}
        t_row = m[m["regime"] == "trending"].iloc[0]
        assert t_row["n_trades"] == 100
        assert 0.6 <= t_row["hit_rate"] <= 0.8   # roughly 0.7
        assert t_row["total_pnl"] > 0
        r_row = m[m["regime"] == "ranging"].iloc[0]
        assert r_row["n_trades"] == 50
        assert 0.4 <= r_row["hit_rate"] <= 0.6   # roughly 0.5


class TestVolatilityQuartile:
    def test_adds_quartile_column(self):
        df = pd.DataFrame({
            "atr_at_entry": np.linspace(0.001, 0.01, 100),
            "entry_price": np.full(100, 1.30),
            "regime": ["trending"] * 100,
            "net_pnl": np.zeros(100),
            "pips": np.zeros(100),
        })
        out = add_volatility_quartile(df)
        assert "vol_quartile" in out.columns
        assert set(out["vol_quartile"].unique()) == {"Q1", "Q2", "Q3", "Q4"}

    def test_no_atr_column_returns_unchanged(self):
        df = pd.DataFrame({"regime": ["x"]})
        out = add_volatility_quartile(df)
        assert "vol_quartile" not in out.columns


# ════════════════════════════════════════════════════════════════════
# Verdict logic
# ════════════════════════════════════════════════════════════════════
class TestVerdict:
    def _per_regime(self, rows: list[dict]) -> pd.DataFrame:
        # Make sure n_trades >= DEFAULT_MIN_TRADES for verdict to engage
        return pd.DataFrame(rows)

    def test_robust_strategy(self):
        per = self._per_regime([
            {"regime": "trending", "n_trades": 100, "hit_rate": 0.75,
             "total_pnl": 500, "annualized_sharpe": 2.0,
             "max_drawdown_dollars": -50, "max_drawdown_pct": -0.05},
            {"regime": "ranging", "n_trades": 80, "hit_rate": 0.72,
             "total_pnl": 300, "annualized_sharpe": 1.5,
             "max_drawdown_dollars": -40, "max_drawdown_pct": -0.04},
            {"regime": "volatile", "n_trades": 60, "hit_rate": 0.70,
             "total_pnl": 200, "annualized_sharpe": 1.2,
             "max_drawdown_dollars": -30, "max_drawdown_pct": -0.03},
        ])
        verdict, diag = compute_verdict(per, min_trades=30)
        assert verdict.startswith("ROBUST")
        assert diag["n_profitable_regimes"] == 3

    def test_unstable_one_regime_negative(self):
        # Need ≥2 profitable regimes so REGIME_BIAS doesn't trigger first;
        # the negative regime triggers UNSTABLE.
        per = self._per_regime([
            {"regime": "trending", "n_trades": 100, "hit_rate": 0.75,
             "total_pnl": 500, "annualized_sharpe": 2.0,
             "max_drawdown_dollars": -50, "max_drawdown_pct": -0.05},
            {"regime": "ranging", "n_trades": 80, "hit_rate": 0.65,
             "total_pnl": 200, "annualized_sharpe": 1.0,
             "max_drawdown_dollars": -60, "max_drawdown_pct": -0.06},
            {"regime": "volatile", "n_trades": 60, "hit_rate": 0.45,
             "total_pnl": -100, "annualized_sharpe": -0.5,
             "max_drawdown_dollars": -200, "max_drawdown_pct": -0.2},
        ])
        verdict, _ = compute_verdict(per, min_trades=30)
        assert verdict.startswith("UNSTABLE")

    def test_regime_bias_single_profitable(self):
        per = self._per_regime([
            {"regime": "trending", "n_trades": 100, "hit_rate": 0.80,
             "total_pnl": 800, "annualized_sharpe": 3.0,
             "max_drawdown_dollars": 0, "max_drawdown_pct": 0.0},
            {"regime": "ranging", "n_trades": 50, "hit_rate": 0.40,
             "total_pnl": -200, "annualized_sharpe": -1.0,
             "max_drawdown_dollars": -300, "max_drawdown_pct": -0.3},
            {"regime": "volatile", "n_trades": 40, "hit_rate": 0.45,
             "total_pnl": -100, "annualized_sharpe": -0.5,
             "max_drawdown_dollars": -150, "max_drawdown_pct": -0.15},
        ])
        verdict, _ = compute_verdict(per, min_trades=30)
        # 1 out of 3 profitable — should be REGIME_BIAS (not UNSTABLE)
        # because the REGIME_BIAS branch comes before UNSTABLE
        assert verdict.startswith("REGIME_BIAS")

    def test_insufficient_data(self):
        per = self._per_regime([
            {"regime": "x", "n_trades": 5, "hit_rate": 0.6,
             "total_pnl": 100, "annualized_sharpe": 1.0,
             "max_drawdown_dollars": 0, "max_drawdown_pct": 0.0},
        ])
        verdict, _ = compute_verdict(per, min_trades=30)
        assert verdict.startswith("INSUFFICIENT_DATA")

    def test_empty_returns_incomplete(self):
        per = pd.DataFrame()
        verdict, _ = compute_verdict(per)
        assert verdict.startswith("INCOMPLETE")


# ════════════════════════════════════════════════════════════════════
# End-to-end (trades.csv → summary)
# ════════════════════════════════════════════════════════════════════
class TestEndToEndFromTradesCSV:
    def _make_trades_csv(self, tmp_path: Path) -> Path:
        """Mock trades.csv with 3 regimes — all profitable, low CV → ROBUST."""
        rng = np.random.RandomState(42)
        rows = []
        for regime, n, win_rate in [
            ("trending", 100, 0.78),
            ("ranging",  80,  0.75),
            ("volatile", 60,  0.72),
        ]:
            wins = rng.rand(n) < win_rate
            pnl = np.where(wins, 60.0, -45.0)
            ts = pd.date_range("2024-01-01", periods=n, freq="1h")
            for i in range(n):
                rows.append({
                    "ts_entry": ts[i],
                    "ts_exit": ts[i] + pd.Timedelta("15min"),
                    "regime": regime,
                    "net_pnl": pnl[i],
                    "pips": pnl[i] / 6.25,
                    "atr_at_entry": 0.005 + rng.rand() * 0.001,
                    "entry_price": 1.30,
                })
        df = pd.DataFrame(rows)
        p = tmp_path / "trades.csv"
        df.to_csv(p, index=False)
        return p

    def test_subprocess_run_produces_outputs(self, tmp_path):
        trades = self._make_trades_csv(tmp_path)
        out = tmp_path / "parity_out"
        result = subprocess.run(
            [sys.executable,
             str(REPO_ROOT / "tools" / "diagnostics" / "regime_parity_test.py"),
             "--trades-csv", str(trades),
             "--output", str(out),
             "--min-trades", "30"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"failed:\n{result.stdout}\n{result.stderr}"
        )
        for fname in ["per_regime_metrics.csv",
                       "per_vol_quartile_metrics.csv",
                       "regime_parity_summary.json",
                       "regime_parity_report.txt"]:
            assert (out / fname).exists(), f"missing {fname}"
        summary = json.loads((out / "regime_parity_summary.json").read_text())
        assert summary["n_trades"] > 200
        assert summary["regime_categorical"]["verdict"].startswith(
            ("ROBUST", "ACCEPTABLE")
        ), f"unexpected verdict: {summary['regime_categorical']['verdict']}"

    def test_missing_regime_column_raises(self, tmp_path):
        # trades.csv without regime → should fail loudly
        df = pd.DataFrame({"net_pnl": [10.0, -20.0]})
        p = tmp_path / "bad_trades.csv"
        df.to_csv(p, index=False)
        result = subprocess.run(
            [sys.executable,
             str(REPO_ROOT / "tools" / "diagnostics" / "regime_parity_test.py"),
             "--trades-csv", str(p),
             "--output", str(tmp_path / "out")],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "regime" in result.stderr.lower()
