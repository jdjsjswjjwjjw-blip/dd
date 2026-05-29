"""Tests for the SSL data health verifier.

Each test crafts a synthetic features parquet with KNOWN failure modes
and verifies the verdict + per-check breakdown matches the contract.
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

from self_supervised.verify_data_health import (
    HealthReport,
    NAN_TOLERANCE,
    check_causality,
    check_nan_fractions,
    check_required_columns,
    check_sample_count,
    estimate_effective_samples,
    verify,
    write_report,
)
from self_supervised.data_loader import (
    SSL_MIN_SAMPLES_ERROR,
    SSL_MIN_SAMPLES_WARN,
)


def _healthy_df(n: int = 50_000) -> pd.DataFrame:
    """Synthetic features that pass every check."""
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        'ts_event': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'open':  100 + rng.randn(n).cumsum() * 0.001,
        'high':  100 + rng.randn(n).cumsum() * 0.001,
        'low':   100 + rng.randn(n).cumsum() * 0.001,
        'close': 100 + rng.randn(n).cumsum() * 0.001,
        'volume': rng.randint(1, 100, n),
        'atr_14': np.abs(rng.randn(n)) * 0.001,
        'obi_net': rng.uniform(-1, 1, n),
        'is_event': (rng.rand(n) > 0.8).astype(np.int8),
        'regime_label': rng.choice(['trending', 'ranging', 'volatile'], n),
        'is_session_break': np.zeros(n, dtype=np.int8),
    })


# ════════════════════════════════════════════════════════════════════
# check_causality
# ════════════════════════════════════════════════════════════════════
class TestCausalityCheck:
    def test_clean_dataset_passes(self):
        df = _healthy_df()
        results = check_causality(df)
        for r in results:
            assert r.passed, f"{r.name}: {r.message}"

    def test_leaked_target_caught(self):
        df = _healthy_df()
        # Pre-bake a target the data_loader should compute itself
        df['next_price'] = 0.0
        results = check_causality(df)
        leaked = [r for r in results if "leaked_targets" in r.name]
        assert any(not r.passed and r.severity == "fail" for r in leaked)

    def test_forward_prefix_must_be_blacklisted(self):
        df = _healthy_df()
        # A column with the `forward_` prefix that ISN'T in the
        # blacklist — would slip into context if not flagged
        df['forward_unrecognized_thing'] = 0.0
        # The prefix matches → _is_leakage_column returns True →
        # data_loader will exclude it. So this check should PASS.
        results = check_causality(df)
        prefix_checks = [
            r for r in results if "prefix_blacklist" in r.name
        ]
        assert all(r.passed for r in prefix_checks), (
            f"forward_-prefixed columns should be auto-flagged: "
            f"{[r.message for r in prefix_checks]}"
        )


# ════════════════════════════════════════════════════════════════════
# check_required_columns
# ════════════════════════════════════════════════════════════════════
class TestRequiredColumns:
    def test_healthy_passes_all(self):
        df = _healthy_df()
        results = check_required_columns(df)
        for r in results:
            assert r.passed, f"{r.name}: {r.message}"

    def test_missing_close_is_fail(self):
        df = _healthy_df()
        df = df.drop(columns=['close'])
        results = check_required_columns(df)
        close_check = [r for r in results if r.name == "required.close"]
        assert close_check[0].severity == "fail"

    def test_missing_obi_alternatives_is_fail(self):
        df = _healthy_df()
        df = df.drop(columns=['obi_net'])
        results = check_required_columns(df)
        obi_check = [r for r in results if r.name == "required.obi"]
        assert obi_check[0].severity == "fail"

    def test_order_flow_imbalance_alternative_passes(self):
        df = _healthy_df()
        df = df.rename(columns={'obi_net': 'order_flow_imbalance'})
        results = check_required_columns(df)
        obi_check = [r for r in results if r.name == "required.obi"]
        assert obi_check[0].passed

    def test_missing_atr_alternatives_is_fail(self):
        df = _healthy_df()
        df = df.drop(columns=['atr_14'])
        results = check_required_columns(df)
        atr_check = [r for r in results if r.name == "required.atr"]
        assert atr_check[0].severity == "fail"

    def test_missing_open_is_warn_only(self):
        df = _healthy_df()
        df = df.drop(columns=['open'])
        results = check_required_columns(df)
        open_check = [r for r in results if r.name == "required.open"]
        assert open_check[0].severity == "warn"


# ════════════════════════════════════════════════════════════════════
# check_nan_fractions
# ════════════════════════════════════════════════════════════════════
class TestNanFractions:
    def test_clean_passes(self):
        df = _healthy_df()
        results = check_nan_fractions(df)
        for r in results:
            assert r.passed, f"{r.name}: {r.message}"

    def test_high_nan_fails(self):
        df = _healthy_df()
        # 10 % NaN on close → above NAN_TOLERANCE (5%)
        idx = np.arange(0, len(df), 10)
        df.loc[idx, 'close'] = np.nan
        results = check_nan_fractions(df)
        close_check = [r for r in results if r.name == "nan.close"]
        assert close_check[0].severity == "fail"

    def test_low_nan_warns(self):
        df = _healthy_df()
        # 2 % NaN — above warn (1%) but below fail (5%)
        idx = np.arange(0, len(df), 50)
        df.loc[idx, 'close'] = np.nan
        results = check_nan_fractions(df)
        close_check = [r for r in results if r.name == "nan.close"]
        assert close_check[0].severity == "warn"


# ════════════════════════════════════════════════════════════════════
# Sample-count estimation
# ════════════════════════════════════════════════════════════════════
class TestSampleCount:
    def test_above_warn_passes(self):
        df = _healthy_df(n=SSL_MIN_SAMPLES_WARN + 5000)
        results, est = check_sample_count(
            df, lookback_bars=50, embargo_bars=24,
        )
        assert results[0].passed
        assert est > SSL_MIN_SAMPLES_WARN

    def test_below_warn_warns(self):
        df = _healthy_df(n=SSL_MIN_SAMPLES_WARN - 1000)
        results, _ = check_sample_count(
            df, lookback_bars=50, embargo_bars=24,
        )
        assert results[0].severity == "warn"

    def test_below_error_fails(self):
        df = _healthy_df(n=SSL_MIN_SAMPLES_ERROR - 100)
        results, _ = check_sample_count(
            df, lookback_bars=50, embargo_bars=24,
        )
        assert results[0].severity == "fail"

    def test_session_breaks_reduce_estimate(self):
        df = _healthy_df(n=SSL_MIN_SAMPLES_WARN + 2000)
        baseline = estimate_effective_samples(df, 50, 24)
        # Introduce 10 session breaks → drops samples whose window crosses
        df.loc[np.arange(1000, len(df), 1000), 'is_session_break'] = 1
        with_breaks = estimate_effective_samples(df, 50, 24)
        assert with_breaks < baseline


# ════════════════════════════════════════════════════════════════════
# End-to-end verify()
# ════════════════════════════════════════════════════════════════════
class TestVerify:
    def test_healthy_dataset_verdict_healthy(self, tmp_path):
        df = _healthy_df()
        p = tmp_path / "features.parquet"
        df.to_parquet(p)
        report = verify(p)
        assert report.verdict == "HEALTHY"
        assert report.n_failures == 0

    def test_missing_critical_yields_fail(self, tmp_path):
        df = _healthy_df()
        df = df.drop(columns=['close'])
        p = tmp_path / "features.parquet"
        df.to_parquet(p)
        report = verify(p)
        assert report.verdict == "FAIL"

    def test_high_nan_yields_fail(self, tmp_path):
        df = _healthy_df()
        df.loc[np.arange(0, len(df), 10), 'atr_14'] = np.nan
        p = tmp_path / "features.parquet"
        df.to_parquet(p)
        report = verify(p)
        assert report.verdict == "FAIL"


# ════════════════════════════════════════════════════════════════════
# CLI subprocess
# ════════════════════════════════════════════════════════════════════
class TestCLI:
    def test_cli_pass(self, tmp_path):
        df = _healthy_df()
        p = tmp_path / "features.parquet"
        df.to_parquet(p)
        out = tmp_path / "health"
        result = subprocess.run(
            [sys.executable,
             str(REPO_ROOT / "self_supervised" / "verify_data_health.py"),
             "--features", str(p),
             "--output", str(out)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"verdict=FAIL on healthy data:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert (out / "data_health_summary.json").exists()
        assert (out / "data_health_report.txt").exists()
        summary = json.loads((out / "data_health_summary.json").read_text())
        assert summary["verdict"] == "HEALTHY"

    def test_cli_fail_returns_nonzero(self, tmp_path):
        df = _healthy_df().drop(columns=['close'])
        p = tmp_path / "features.parquet"
        df.to_parquet(p)
        out = tmp_path / "health"
        result = subprocess.run(
            [sys.executable,
             str(REPO_ROOT / "self_supervised" / "verify_data_health.py"),
             "--features", str(p),
             "--output", str(out)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        summary = json.loads((out / "data_health_summary.json").read_text())
        assert summary["verdict"] == "FAIL"
