"""Tests for the slippage calibration tool.

The tool is decomposed into pure pieces: each region's fit, the verdict
classifier, and the orchestrator. Each is tested in isolation so a
regression points at one suspect.
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

from tools.calibrate_slippage import (
    DEPTH_FLOOR,
    MIN_EVENTS_PER_REGION,
    MIN_TOTAL_EVENTS,
    CalibratedConstants,
    apply_safety_clamps,
    calibrate,
    classify_events,
    compute_verdict,
    fit_base_ticks,
    fit_extra_per_level,
    fit_mid_ticks,
    load_log,
    run_calibration,
    summarise_regions,
    _enough_nonzero_realised,
    _ols_no_intercept,
    _ratio_and_region,
)


# ════════════════════════════════════════════════════════════════════
# Region classification
# ════════════════════════════════════════════════════════════════════
class TestRatioAndRegion:
    def test_sub_l1_region(self):
        r, reg = _ratio_and_region(5.0, 100.0)
        assert r == pytest.approx(0.05)
        assert reg == "A_sub_L1"

    def test_linear_region(self):
        _, reg = _ratio_and_region(30.0, 100.0)
        assert reg == "B_linear"

    def test_walking_region(self):
        _, reg = _ratio_and_region(150.0, 100.0)
        assert reg == "C_walking"

    def test_zero_l1_depth(self):
        r, reg = _ratio_and_region(5.0, 0.0)
        assert r == float("inf")
        assert reg == "D_zero_L1"

    def test_boundary_at_depth_floor(self):
        # Exactly at floor → still region A (closed boundary)
        _, reg = _ratio_and_region(10.0, 100.0)
        assert reg == "A_sub_L1"

    def test_boundary_at_one(self):
        _, reg = _ratio_and_region(100.0, 100.0)
        assert reg == "B_linear"


# ════════════════════════════════════════════════════════════════════
# OLS-through-origin
# ════════════════════════════════════════════════════════════════════
class TestOLSNoIntercept:
    def test_recovers_known_slope(self):
        x = np.linspace(0.1, 1.0, 100)
        y = 2.5 * x
        slope = _ols_no_intercept(x, y)
        assert slope == pytest.approx(2.5)

    def test_with_noise_close_to_true(self):
        rng = np.random.RandomState(0)
        x = np.linspace(0.1, 1.0, 200)
        y = 3.0 * x + rng.randn(200) * 0.05
        slope = _ols_no_intercept(x, y)
        assert slope == pytest.approx(3.0, abs=0.1)

    def test_empty_returns_none(self):
        assert _ols_no_intercept(np.array([]), np.array([])) is None

    def test_zero_variance_returns_none(self):
        x = np.zeros(10)
        y = np.ones(10)
        assert _ols_no_intercept(x, y) is None


# ════════════════════════════════════════════════════════════════════
# Helpers — synthetic execution log
# ════════════════════════════════════════════════════════════════════
def _synth_event(
    intended_size: float, l1_depth: float, realised: float,
    model: float = 1.5, levels: int = 1, signal_id: int = 0,
) -> dict:
    """One row of an execution log — only the fields the calibrator reads."""
    return {
        "signal_id": signal_id,
        "side": "BUY",
        "intended_size": float(intended_size),
        "decision_lob_l1_depth": float(l1_depth),
        "realised_slippage_ticks": float(realised),
        "model_slippage_ticks": float(model),
        "levels_consumed": int(levels),
    }


def _df(events: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(events)


# ════════════════════════════════════════════════════════════════════
# Region-by-region fits
# ════════════════════════════════════════════════════════════════════
class TestFitBaseTicks:
    def test_recovers_mean(self):
        events = [
            _synth_event(5, 100, realised=0.3) for _ in range(8)
        ] + [
            _synth_event(5, 100, realised=0.5) for _ in range(8)
        ]
        df = classify_events(_df(events))
        base = fit_base_ticks(df[df["region"] == "A_sub_L1"])
        assert base == pytest.approx(0.4)

    def test_too_few_returns_none(self):
        events = [_synth_event(5, 100, realised=0.3) for _ in range(2)]
        df = classify_events(_df(events))
        base = fit_base_ticks(df[df["region"] == "A_sub_L1"])
        assert base is None


class TestFitMidTicks:
    def test_recovers_linear_ramp(self):
        # Generate region-B events from a known (base, mid) pair
        # realised = base + t · (mid − base)
        true_base, true_mid = 0.5, 2.0
        events = []
        for i, ratio in enumerate(np.linspace(0.15, 1.0, 20)):
            t = (ratio - DEPTH_FLOOR) / (1 - DEPTH_FLOOR)
            realised = true_base + t * (true_mid - true_base)
            events.append(
                _synth_event(
                    intended_size=ratio * 100, l1_depth=100,
                    realised=realised, signal_id=i,
                )
            )
        df = classify_events(_df(events))
        b = df[df["region"] == "B_linear"]
        mid = fit_mid_ticks(b, base_ticks=true_base)
        assert mid == pytest.approx(true_mid, abs=1e-6)

    def test_too_few_returns_none(self):
        events = [
            _synth_event(30, 100, realised=1.0, signal_id=i)
            for i in range(MIN_EVENTS_PER_REGION - 1)
        ]
        df = classify_events(_df(events))
        b = df[df["region"] == "B_linear"]
        assert fit_mid_ticks(b, base_ticks=0.5) is None


class TestFitExtraPerLevel:
    def test_recovers_slope(self):
        # realised = mid + extra · (levels − 1)
        true_mid, true_extra = 2.0, 0.8
        events = []
        for i in range(20):
            levels = 2 + (i % 5)   # 2..6
            realised = true_mid + true_extra * (levels - 1)
            events.append(
                _synth_event(
                    intended_size=200, l1_depth=20,   # ratio = 10 → walking
                    realised=realised, levels=levels, signal_id=i,
                )
            )
        df = classify_events(_df(events))
        c = df[df["region"] == "C_walking"]
        extra = fit_extra_per_level(c, mid_ticks=true_mid)
        assert extra == pytest.approx(true_extra, abs=1e-6)

    def test_too_few_returns_none(self):
        events = [
            _synth_event(200, 20, realised=3.0, levels=3, signal_id=i)
            for i in range(MIN_EVENTS_PER_REGION - 1)
        ]
        df = classify_events(_df(events))
        c = df[df["region"] == "C_walking"]
        assert fit_extra_per_level(c, mid_ticks=2.0) is None


# ════════════════════════════════════════════════════════════════════
# End-to-end calibrate()
# ════════════════════════════════════════════════════════════════════
class TestCalibrateEndToEnd:
    def test_recovers_full_param_set(self):
        true = (0.4, 1.8, 0.7)   # base, mid, extra
        events = []
        sid = 0

        # 10 region-A events
        for _ in range(10):
            events.append(_synth_event(5, 100, realised=true[0], signal_id=sid))
            sid += 1
        # 20 region-B events
        for ratio in np.linspace(0.15, 1.0, 20):
            t = (ratio - DEPTH_FLOOR) / (1 - DEPTH_FLOOR)
            realised = true[0] + t * (true[1] - true[0])
            events.append(
                _synth_event(
                    intended_size=ratio * 100, l1_depth=100,
                    realised=realised, signal_id=sid,
                )
            )
            sid += 1
        # 15 region-C events
        for i in range(15):
            levels = 2 + (i % 5)
            realised = true[1] + true[2] * (levels - 1)
            events.append(
                _synth_event(
                    intended_size=200, l1_depth=20,
                    realised=realised, levels=levels, signal_id=sid,
                )
            )
            sid += 1

        result = calibrate(_df(events))
        assert result.base_ticks == pytest.approx(true[0])
        assert result.mid_ticks == pytest.approx(true[1], abs=1e-5)
        assert result.extra_per_level == pytest.approx(true[2], abs=1e-5)

    def test_empty_log_returns_all_none(self):
        result = calibrate(pd.DataFrame())
        assert result.base_ticks is None
        assert result.mid_ticks is None
        assert result.extra_per_level is None


# ════════════════════════════════════════════════════════════════════
# Verdict
# ════════════════════════════════════════════════════════════════════
class TestVerdict:
    def _events(self, model_mean: float, realised_mean: float, n: int = 20):
        return _df([
            _synth_event(
                intended_size=10, l1_depth=100,
                realised=realised_mean, model=model_mean, signal_id=i,
            )
            for i in range(n)
        ])

    def test_insufficient_data(self):
        df = self._events(1.5, 1.0, n=MIN_TOTAL_EVENTS - 1)
        v, _ = compute_verdict(df)
        assert v == "INSUFFICIENT_DATA"

    def test_well_calibrated(self):
        df = self._events(1.0, 1.0, n=20)
        v, diag = compute_verdict(df, tolerance=0.25)
        assert v == "WELL_CALIBRATED"
        assert diag["ratio_model_over_realised"] == pytest.approx(1.0)

    def test_over_calibrated(self):
        df = self._events(3.0, 1.0, n=20)
        v, diag = compute_verdict(df, tolerance=0.25)
        assert v == "OVER_CALIBRATED"
        assert diag["ratio_model_over_realised"] == pytest.approx(3.0)

    def test_under_calibrated(self):
        df = self._events(0.4, 1.0, n=20)
        v, _ = compute_verdict(df, tolerance=0.25)
        assert v == "UNDER_CALIBRATED"

    def test_zero_realised_returns_insufficient(self):
        df = self._events(1.5, 0.0, n=20)
        v, diag = compute_verdict(df)
        assert v == "INSUFFICIENT_DATA"
        assert "note" in diag


# ════════════════════════════════════════════════════════════════════
# Region summary
# ════════════════════════════════════════════════════════════════════
class TestSummariseRegions:
    def test_per_region_counts_match(self):
        events = (
            [_synth_event(5, 100, 0.3, model=0.5, signal_id=i) for i in range(8)]
            + [_synth_event(50, 100, 1.0, model=1.5, signal_id=10 + i) for i in range(8)]
            + [_synth_event(200, 100, 2.0, model=2.5, signal_id=20 + i, levels=3) for i in range(8)]
        )
        df = classify_events(_df(events))
        out = summarise_regions(df)
        assert out["A_sub_L1"]["n"] == 8
        assert out["B_linear"]["n"] == 8
        assert out["C_walking"]["n"] == 8
        assert out["A_sub_L1"]["mean_realised"] == pytest.approx(0.3)


# ════════════════════════════════════════════════════════════════════
# Safety clamps — guard against unsafe OLS suggestions
# ════════════════════════════════════════════════════════════════════
class TestEnoughNonzero:
    def test_all_zero_returns_false(self):
        df = _df([_synth_event(50, 100, realised=0.0) for _ in range(10)])
        df = classify_events(df)
        assert _enough_nonzero_realised(df) is False

    def test_all_nonzero_returns_true(self):
        df = _df([_synth_event(50, 100, realised=1.5) for _ in range(10)])
        df = classify_events(df)
        assert _enough_nonzero_realised(df) is True

    def test_empty_returns_false(self):
        assert _enough_nonzero_realised(pd.DataFrame()) is False


class TestFitMidWithL0FitsRefusesUnsafe:
    def test_all_zero_realised_returns_none(self):
        # 10 region-B events all with realised=0 → OLS would give mid<base
        events = []
        for i, ratio in enumerate(np.linspace(0.15, 1.0, 10)):
            events.append(
                _synth_event(
                    intended_size=ratio * 100, l1_depth=100,
                    realised=0.0, signal_id=i,
                )
            )
        df = classify_events(_df(events))
        b = df[df["region"] == "B_linear"]
        # Refuses to fit because realised is dominated by L0-fits
        assert fit_mid_ticks(b, base_ticks=0.5) is None


class TestApplyClamps:
    def test_negative_mid_falls_back(self):
        raw = CalibratedConstants(
            base_ticks=0.5, mid_ticks=-1.0, extra_per_level=0.8,
        )
        clamped, notes = apply_safety_clamps(
            raw, current={"base_ticks": 0.5, "mid_ticks": 1.5, "extra_per_level": 1.0},
        )
        assert clamped.mid_ticks is None       # falls back to current
        assert clamped.base_ticks == 0.5
        assert clamped.extra_per_level == 0.8
        assert any("mid_ticks" in n for n in notes)

    def test_mid_below_base_falls_back(self):
        raw = CalibratedConstants(
            base_ticks=1.0, mid_ticks=0.5, extra_per_level=0.5,
        )
        clamped, notes = apply_safety_clamps(
            raw, current={"base_ticks": 1.0, "mid_ticks": 1.5, "extra_per_level": 1.0},
        )
        assert clamped.mid_ticks is None
        assert any("< base_ticks" in n for n in notes)

    def test_negative_extra_falls_back(self):
        raw = CalibratedConstants(
            base_ticks=0.5, mid_ticks=1.5, extra_per_level=-0.5,
        )
        clamped, notes = apply_safety_clamps(
            raw, current={"base_ticks": 0.5, "mid_ticks": 1.5, "extra_per_level": 1.0},
        )
        assert clamped.extra_per_level is None
        assert any("extra_per_level" in n for n in notes)

    def test_all_safe_no_notes(self):
        raw = CalibratedConstants(
            base_ticks=0.4, mid_ticks=1.8, extra_per_level=0.7,
        )
        clamped, notes = apply_safety_clamps(
            raw, current={"base_ticks": 0.5, "mid_ticks": 1.5, "extra_per_level": 1.0},
        )
        assert clamped.base_ticks == 0.4
        assert clamped.mid_ticks == 1.8
        assert clamped.extra_per_level == 0.7
        assert notes == []


# ════════════════════════════════════════════════════════════════════
# CalibratedConstants kwargs
# ════════════════════════════════════════════════════════════════════
class TestCalibratedConstants:
    def test_to_kwargs_with_all_fields(self):
        c = CalibratedConstants(
            base_ticks=0.4, mid_ticks=1.8, extra_per_level=0.7,
        )
        kw = c.to_kwargs(tick_size=0.0001)
        assert kw["base_ticks"] == 0.4
        assert kw["mid_ticks"] == 1.8
        assert kw["extra_per_level"] == 0.7
        assert kw["tick_size"] == 0.0001

    def test_to_kwargs_falls_back_for_none(self):
        c = CalibratedConstants(
            base_ticks=None, mid_ticks=1.8, extra_per_level=None,
        )
        fallback = {"base_ticks": 0.5, "extra_per_level": 1.0}
        kw = c.to_kwargs(tick_size=0.0001, fallback=fallback)
        assert kw["base_ticks"] == 0.5
        assert kw["mid_ticks"] == 1.8
        assert kw["extra_per_level"] == 1.0


# ════════════════════════════════════════════════════════════════════
# I/O — load_log + run_calibration
# ════════════════════════════════════════════════════════════════════
class TestIO:
    def test_load_log_missing_columns_raises(self, tmp_path):
        log = tmp_path / "bad.jsonl"
        log.write_text(json.dumps({"foo": 1}) + "\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_log(log)

    def test_load_log_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_log(tmp_path / "absent.jsonl")

    def test_run_calibration_writes_all_outputs(self, tmp_path):
        events = (
            [_synth_event(5, 100, 0.4, model=0.5, signal_id=i) for i in range(8)]
            + [_synth_event(50, 100, 1.2, model=1.5, signal_id=10 + i) for i in range(8)]
        )
        log = tmp_path / "log.jsonl"
        log.write_text("\n".join(json.dumps(e) for e in events))
        out = tmp_path / "calib"
        summary = run_calibration(log, out, tick_size=0.01)
        assert (out / "calibration_summary.json").exists()
        assert (out / "slippage_config_suggested.json").exists()
        assert (out / "calibration_report.txt").exists()
        assert summary["verdict"] in {
            "WELL_CALIBRATED", "OVER_CALIBRATED",
            "UNDER_CALIBRATED", "INSUFFICIENT_DATA",
        }


# ════════════════════════════════════════════════════════════════════
# End-to-end CLI subprocess
# ════════════════════════════════════════════════════════════════════
class TestCLI:
    def test_cli_produces_outputs(self, tmp_path):
        events = (
            [_synth_event(5, 100, 0.4, model=0.5, signal_id=i) for i in range(8)]
            + [_synth_event(50, 100, 1.2, model=1.5, signal_id=10 + i) for i in range(8)]
        )
        log = tmp_path / "log.jsonl"
        log.write_text("\n".join(json.dumps(e) for e in events))
        out = tmp_path / "calib"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "calibrate_slippage.py"),
                "--log", str(log),
                "--output", str(out),
                "--tick-size", "0.01",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"failed:\n{result.stdout}\n{result.stderr}"
        )
        assert (out / "calibration_summary.json").exists()
        summary = json.loads((out / "calibration_summary.json").read_text())
        assert "verdict" in summary
        assert "suggested_constants" in summary
