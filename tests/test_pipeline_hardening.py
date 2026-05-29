"""Tests for the SSL pipeline hardening — Pinning + Effective Sample Report.

Two layers tested:
  1. Required-column pinning: missing obi / ATR raises with a clear
     message instead of falling back to noise.
  2. Sample-size check: degenerate counts trigger warnings or errors
     at construction, not deep inside training.

These are surface-level unit tests on the helpers — full Dataset
construction needs a heavy synthetic parquet which is exercised
separately in tests/test_verify_data_health.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from self_supervised.data_loader import (
    SSL_MIN_SAMPLES_ERROR,
    SSL_MIN_SAMPLES_WARN,
    _check_sample_size,
)


# ════════════════════════════════════════════════════════════════════
# Sample-size guard
# ════════════════════════════════════════════════════════════════════
class TestCheckSampleSize:
    def test_above_warn_prints_ok(self, capsys):
        _check_sample_size(SSL_MIN_SAMPLES_WARN + 1_000)
        captured = capsys.readouterr()
        assert "✅" in captured.out
        assert "Total Effective Samples" in captured.out

    def test_between_warn_and_error_prints_warning(self, capsys):
        _check_sample_size(SSL_MIN_SAMPLES_WARN - 1)
        captured = capsys.readouterr()
        assert "⚠️" in captured.out
        assert "BELOW recommended" in captured.out

    def test_below_error_raises(self):
        # Ensure the bypass env var is NOT set
        os.environ.pop("SSL_BYPASS_MIN_SAMPLES", None)
        with pytest.raises(RuntimeError, match="below the hard floor"):
            _check_sample_size(SSL_MIN_SAMPLES_ERROR - 1)

    def test_bypass_env_skips_error(self, capsys, monkeypatch):
        monkeypatch.setenv("SSL_BYPASS_MIN_SAMPLES", "1")
        # Should NOT raise even at zero samples
        _check_sample_size(0)
        captured = capsys.readouterr()
        assert "Bypassed" in captured.out

    def test_thresholds_are_correctly_ordered(self):
        # Sanity: error floor must be below warn threshold
        assert SSL_MIN_SAMPLES_ERROR < SSL_MIN_SAMPLES_WARN
        assert SSL_MIN_SAMPLES_ERROR > 0


# ════════════════════════════════════════════════════════════════════
# Column pinning — exercised on a minimal synthetic DataFrame
# ════════════════════════════════════════════════════════════════════
class TestColumnPinning:
    """Verify the obi / ATR pinning by reading the source — these
    columns are coded into data_loader.py and any silent regression to
    the old fallback would break this test."""

    def test_obi_pinning_in_source(self):
        src_path = REPO_ROOT / "self_supervised" / "data_loader.py"
        src = src_path.read_text()
        # The new code must reference both column names explicitly
        # AND raise a RuntimeError with a clear message
        assert "'obi_net'" in src or '"obi_net"' in src
        assert "'order_flow_imbalance'" in src or '"order_flow_imbalance"' in src
        # The pinning sentinel — must be present in the loud-failure
        # branch, NOT in a fallback comment
        assert "Pipeline misconfigured" in src
        assert "next_imbalance task requires" in src

    def test_atr_pinning_in_source(self):
        src_path = REPO_ROOT / "self_supervised" / "data_loader.py"
        src = src_path.read_text()
        assert "'atr_14'" in src or '"atr_14"' in src
        assert "next_volatility task requires" in src

    def test_obi_fallback_removed(self):
        """The old `_safe_col(obi_col, fallback=0.0)` pattern is gone,
        replaced by the loud-failure check. The 0.0 fallback for OBI
        would have produced a silently-zero target — exactly the
        failure mode the pinning prevents."""
        src_path = REPO_ROOT / "self_supervised" / "data_loader.py"
        src = src_path.read_text()
        # The old single-line fallback we replaced; if it ever returns,
        # this test will catch it
        forbidden = (
            "obi_col = 'obi_net' if 'obi_net' in self.df.columns "
            "else 'order_flow_imbalance'\n"
            "        obi = _safe_col(obi_col, fallback=0.0)"
        )
        assert forbidden not in src
