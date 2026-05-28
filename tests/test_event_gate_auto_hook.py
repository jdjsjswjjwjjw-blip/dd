"""Tests for the event-gate auto-hook in prepare_day_trading.py.

The hook is a safe wrapper around `tools.diagnostics.audit_event_gate.run_audit`
that runs as a post-write step in the day-trade pipeline. The contract:

  - Healthy gate              → prints "✅ Event gate: HEALTHY"
  - Starved gate              → prints "🚨 Event gate: STARVED" + diag
  - Missing parquet           → prints "⚠️ event-gate audit failed" + continues
  - Missing audit module      → prints "⚠️ event-gate audit unavailable"
                                 + continues
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import the helper directly — no need to run the 4700-line main flow
from prepare_day_trading import _run_event_gate_audit_safe


def _make_features(
    tmp_path: Path,
    rate: float,
    n: int = 2000,
    cvd_all_nan: bool = False,
) -> Path:
    """Synthetic features parquet with the columns the audit needs.

    Sessions are arranged in contiguous blocks (asian → london → ny),
    matching real day-bar data. Random session assignment would make
    every transition look like a warm-up start and dominate the verdict.
    """
    rng = np.random.RandomState(0)
    is_event = (rng.rand(n) < rate).astype(np.int8)
    cvd = (
        np.full(n, np.nan)
        if cvd_all_nan
        else rng.rand(n).astype(np.float64)
    )
    # Contiguous session blocks: 1/3 asian, 1/3 london, 1/3 ny
    third = n // 3
    session = (
        ["asian"] * third
        + ["london"] * third
        + ["ny"] * (n - 2 * third)
    )
    df = pd.DataFrame(
        {
            "ts_event": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "is_event": is_event,
            "event_score": rng.rand(n).astype(np.float32),
            "regime_label": rng.choice(
                ["trending", "ranging", "volatile"], size=n
            ),
            "session": session,
            "hawkes_intensity": rng.randn(n),
            "absorption_intensity": rng.randn(n),
            "kyle_lambda": rng.randn(n),
            "cvd_direction_pct": cvd,
        }
    )
    p = tmp_path / "features.parquet"
    df.to_parquet(p)
    return p


# ════════════════════════════════════════════════════════════════════
# Happy path — verdict + output files
# ════════════════════════════════════════════════════════════════════
class TestHealthyGate:
    def test_returns_summary_and_writes_files(self, tmp_path, capsys):
        feats = _make_features(tmp_path, rate=0.25)
        summary = _run_event_gate_audit_safe(str(feats), str(tmp_path))

        assert summary is not None
        assert "verdict" in summary
        assert summary["n_rows"] == 2000

        audit_dir = tmp_path / "_audit_event_gate"
        assert (audit_dir / "event_gate_summary.json").exists()
        assert (audit_dir / "event_gate_report.txt").exists()
        assert (audit_dir / "component_failures.csv").exists()

    def test_prints_verdict_line(self, tmp_path, capsys):
        feats = _make_features(tmp_path, rate=0.25)
        _run_event_gate_audit_safe(str(feats), str(tmp_path))
        captured = capsys.readouterr()
        assert "Event gate:" in captured.out
        assert "_audit_event_gate" in captured.out


# ════════════════════════════════════════════════════════════════════
# Starved gate — loud diagnostic block
# ════════════════════════════════════════════════════════════════════
class TestStarvedGate:
    def test_starved_prints_root_cause(self, tmp_path, capsys):
        # rate ~5 % → STARVED
        feats = _make_features(tmp_path, rate=0.05)
        summary = _run_event_gate_audit_safe(str(feats), str(tmp_path))
        captured = capsys.readouterr()

        assert summary["verdict"] == "STARVED"
        assert "STARVED" in captured.out
        assert "gate وليس market behavior" in captured.out

    def test_component_dominated_prints_root_cause(self, tmp_path, capsys):
        # cvd all NaN → fillna(0.5) < 0.6 → 100 % failure on that component
        feats = _make_features(tmp_path, rate=0.25, cvd_all_nan=True)
        summary = _run_event_gate_audit_safe(str(feats), str(tmp_path))
        captured = capsys.readouterr()

        assert summary["verdict"] == "COMPONENT_DOMINATED"
        assert "COMPONENT_DOMINATED" in captured.out
        assert "gate وليس market behavior" in captured.out
        # Diagnostic block lists the failing component
        assert "cvd_align_above_06" in captured.out


# ════════════════════════════════════════════════════════════════════
# Graceful degradation — never breaks the pipeline
# ════════════════════════════════════════════════════════════════════
class TestGracefulFailure:
    def test_missing_parquet_returns_none(self, tmp_path, capsys):
        bogus = tmp_path / "does_not_exist.parquet"
        summary = _run_event_gate_audit_safe(str(bogus), str(tmp_path))
        captured = capsys.readouterr()

        assert summary is None
        assert "event-gate audit failed" in captured.out

    def test_parquet_without_is_event_returns_none(self, tmp_path, capsys):
        # Parquet exists but missing the required `is_event` column
        df = pd.DataFrame({"x": np.zeros(2000)})
        p = tmp_path / "bad.parquet"
        df.to_parquet(p)

        summary = _run_event_gate_audit_safe(str(p), str(tmp_path))
        captured = capsys.readouterr()

        assert summary is None
        assert "failed" in captured.out
        # Pipeline keeps running — no exception propagates out
