"""Tests for the event-gate audit tool.

Each test crafts a synthetic features.parquet with KNOWN gate
characteristics and verifies the audit's metrics + verdict.
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

from tools.diagnostics.audit_event_gate import (
    COMPONENT_DOMINATION_PCT,
    DESIGN_RATE_MIN,
    MIN_ROWS_FOR_AUDIT,
    STARVED_RATE,
    WARMUP_HEAVY_PCT,
    compute_component_failures,
    compute_event_rate,
    compute_per_group_rate,
    compute_verdict,
    detect_cvd_fillna_contamination,
    detect_warmup_zero_bias,
    run_audit,
)


# ════════════════════════════════════════════════════════════════════
# Metric helpers
# ════════════════════════════════════════════════════════════════════
class TestEventRate:
    def test_empty_returns_nan(self):
        df = pd.DataFrame()
        assert np.isnan(compute_event_rate(df))

    def test_missing_column_returns_nan(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert np.isnan(compute_event_rate(df))

    def test_known_rate(self):
        df = pd.DataFrame({"is_event": [1, 1, 0, 0, 1]})
        assert compute_event_rate(df) == pytest.approx(0.6)


class TestPerGroupRate:
    def test_two_groups(self):
        df = pd.DataFrame(
            {
                "regime_label": ["trending"] * 10 + ["volatile"] * 10,
                "is_event": [1] * 6 + [0] * 4 + [1] * 2 + [0] * 8,
            }
        )
        out = compute_per_group_rate(df, "regime_label")
        assert out["trending"]["rate"] == pytest.approx(0.6)
        assert out["trending"]["n_events"] == 6
        assert out["volatile"]["rate"] == pytest.approx(0.2)

    def test_missing_group_col_returns_empty(self):
        df = pd.DataFrame({"is_event": [1, 0]})
        assert compute_per_group_rate(df, "nonexistent") == {}


# ════════════════════════════════════════════════════════════════════
# Component failure analysis
# ════════════════════════════════════════════════════════════════════
class TestComponentFailures:
    def test_missing_sources_marked_unavailable(self):
        df = pd.DataFrame({"is_event": [0] * 100})
        out = compute_component_failures(df)
        for comp in [
            "hawkes_z_above_1",
            "absorb_z_above_1",
            "kyle_z_above_05",
            "cvd_align_above_06",
        ]:
            assert out[comp]["available"] is False

    def test_cvd_fillna_05_dominated(self):
        # cvd_direction_pct all-NaN → fillna(0.5) → all fail > 0.6
        n = 500
        df = pd.DataFrame({"cvd_direction_pct": [np.nan] * n})
        out = compute_component_failures(df)
        comp = out["cvd_align_above_06"]
        assert comp["available"]
        assert comp["fail_rate"] == 1.0
        assert comp["nan_rate"] == 1.0

    def test_strong_cvd_signal_passes(self):
        n = 500
        df = pd.DataFrame({"cvd_direction_pct": [0.8] * n})
        out = compute_component_failures(df)
        assert out["cvd_align_above_06"]["pass_rate"] == 1.0
        assert out["cvd_align_above_06"]["fail_rate"] == 0.0


class TestCvdFillnaContamination:
    def test_no_cvd_col_unavailable(self):
        df = pd.DataFrame({"x": [1]})
        assert detect_cvd_fillna_contamination(df)["available"] is False

    def test_half_nan(self):
        df = pd.DataFrame(
            {"cvd_direction_pct": [0.7] * 500 + [np.nan] * 500}
        )
        out = detect_cvd_fillna_contamination(df)
        assert out["available"]
        assert out["nan_rate"] == pytest.approx(0.5)
        assert out["auto_fail_rate"] == pytest.approx(0.5)


# ════════════════════════════════════════════════════════════════════
# Warm-up zero bias
# ════════════════════════════════════════════════════════════════════
class TestWarmupBias:
    def test_global_warmup(self):
        df = pd.DataFrame({"is_event": [0] * 1000})
        out = detect_warmup_zero_bias(df, window=100, min_periods=20)
        assert out["scope"] == "global"
        assert out["warmup_rows"] == 20
        assert out["warmup_rate"] == pytest.approx(0.02)

    def test_per_session_warmup(self):
        # 3 sessions × 200 bars each, min_periods=20 → 60 warm-up rows total
        df = pd.DataFrame(
            {
                "is_event": [0] * 600,
                "session": ["asian"] * 200 + ["london"] * 200 + ["ny"] * 200,
            }
        )
        out = detect_warmup_zero_bias(df, window=100, min_periods=20)
        assert out["scope"] == "per_session"
        assert out["warmup_rows"] == 60
        assert out["warmup_rate"] == pytest.approx(0.1)


# ════════════════════════════════════════════════════════════════════
# Verdict cascade
# ════════════════════════════════════════════════════════════════════
class TestVerdict:
    def _base(self):
        return dict(
            per_regime={},
            per_session={},
            components={},
            warmup={"available": False},
            n_rows=10_000,
        )

    def test_incomplete_on_nan_rate(self):
        v, _ = compute_verdict(overall_rate=float("nan"), **self._base())
        assert v == "INCOMPLETE"

    def test_insufficient_data(self):
        base = self._base()
        base["n_rows"] = 500
        v, _ = compute_verdict(overall_rate=0.25, **base)
        assert v == "INSUFFICIENT_DATA"

    def test_starved(self):
        v, _ = compute_verdict(overall_rate=0.05, **self._base())
        assert v == "STARVED"

    def test_warmup_heavy_triggers(self):
        base = self._base()
        base["warmup"] = {"available": True, "warmup_rate": 0.10}
        v, _ = compute_verdict(overall_rate=0.25, **base)
        assert v == "WARMUP_HEAVY"

    def test_component_dominated(self):
        base = self._base()
        base["components"] = {
            "cvd_align_above_06": {"available": True, "fail_rate": 0.95}
        }
        v, diag = compute_verdict(overall_rate=0.25, **base)
        assert v == "COMPONENT_DOMINATED"
        assert "cvd_align_above_06" in diag["dominated_components"]

    def test_regime_starved(self):
        base = self._base()
        base["per_regime"] = {
            "trending": {"rate": 0.10, "n_rows": 5000, "n_events": 500},
            "ranging": {"rate": 0.25, "n_rows": 5000, "n_events": 1250},
        }
        v, _ = compute_verdict(overall_rate=0.22, **base)
        assert v == "REGIME_STARVED"

    def test_regime_starvation_ignores_volatile(self):
        # volatile is SUPPOSED to be strict — low rate there is not a flag
        base = self._base()
        base["per_regime"] = {
            "trending": {"rate": 0.25, "n_rows": 5000, "n_events": 1250},
            "volatile": {"rate": 0.05, "n_rows": 5000, "n_events": 250},
        }
        v, _ = compute_verdict(overall_rate=0.22, **base)
        assert v == "HEALTHY"

    def test_low_rate(self):
        v, _ = compute_verdict(overall_rate=0.15, **self._base())
        assert v == "LOW_RATE"

    def test_healthy(self):
        v, _ = compute_verdict(overall_rate=0.25, **self._base())
        assert v == "HEALTHY"


# ════════════════════════════════════════════════════════════════════
# End-to-end via subprocess
# ════════════════════════════════════════════════════════════════════
class TestEndToEnd:
    def _make_features(self, tmp_path: Path, rate: float = 0.25) -> Path:
        rng = np.random.RandomState(0)
        n = 2000
        is_event = (rng.rand(n) < rate).astype(np.int8)
        df = pd.DataFrame(
            {
                "ts_event": pd.date_range("2024-01-01", periods=n, freq="5min"),
                "is_event": is_event,
                "event_score": rng.rand(n).astype(np.float32),
                "regime_label": rng.choice(
                    ["trending", "ranging", "volatile"], size=n
                ),
                "session": rng.choice(["asian", "london", "ny"], size=n),
                "hawkes_intensity": rng.randn(n),
                "absorption_intensity": rng.randn(n),
                "kyle_lambda": rng.randn(n),
                "cvd_direction_pct": rng.rand(n),
            }
        )
        p = tmp_path / "features.parquet"
        df.to_parquet(p)
        return p

    def test_subprocess_produces_outputs(self, tmp_path):
        feats = self._make_features(tmp_path, rate=0.25)
        out = tmp_path / "audit"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "diagnostics" / "audit_event_gate.py"),
                "--features",
                str(feats),
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"failed:\n{result.stdout}\n{result.stderr}"
        )
        for fname in [
            "event_gate_summary.json",
            "event_gate_report.txt",
            "component_failures.csv",
        ]:
            assert (out / fname).exists(), f"missing {fname}"
        summary = json.loads((out / "event_gate_summary.json").read_text())
        # 25% synthetic rate, no warm-up issue, all components present →
        # should be HEALTHY (or REGIME_STARVED on very unlucky seed)
        assert summary["verdict"] in {
            "HEALTHY",
            "REGIME_STARVED",
            "WARMUP_HEAVY",
        }
        assert summary["overall_event_rate"] == pytest.approx(0.25, abs=0.05)

    def test_missing_is_event_raises(self, tmp_path):
        df = pd.DataFrame({"x": np.zeros(2000)})
        p = tmp_path / "bad.parquet"
        df.to_parquet(p)
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "diagnostics" / "audit_event_gate.py"),
                "--features",
                str(p),
                "--output",
                str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "is_event" in result.stderr.lower()
