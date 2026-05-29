"""Tests for the SSL early-loss watchdog.

The watchdog is a pure stateful monitor over per-step losses. Each test
feeds a synthetic loss trajectory and asserts the verdict + diagnosis
match the documented contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from self_supervised.loss_watchdog import (
    DIVERGING,
    HEALTHY,
    SATURATED,
    WARMING_UP,
    LossWatchdog,
    WatchdogConfig,
)


# ════════════════════════════════════════════════════════════════════
# Config validation
# ════════════════════════════════════════════════════════════════════
class TestConfig:
    def test_defaults_valid(self):
        cfg = WatchdogConfig()
        assert cfg.warmup_steps == 100
        assert cfg.window == 20

    def test_warmup_too_small_raises(self):
        with pytest.raises(ValueError, match="warmup_steps"):
            WatchdogConfig(warmup_steps=1)

    def test_window_overlap_raises(self):
        # 2*window must be ≤ warmup_steps
        with pytest.raises(ValueError, match="don't overlap"):
            WatchdogConfig(warmup_steps=30, window=20)

    def test_negative_min_improvement_raises(self):
        with pytest.raises(ValueError, match="min_improvement"):
            WatchdogConfig(min_improvement=-0.1)

    def test_negative_divergence_tolerance_raises(self):
        with pytest.raises(ValueError, match="divergence_tolerance"):
            WatchdogConfig(divergence_tolerance=-0.1)


# ════════════════════════════════════════════════════════════════════
# Warm-up behaviour
# ════════════════════════════════════════════════════════════════════
class TestWarmup:
    def test_warming_up_until_threshold(self):
        wd = LossWatchdog(warmup_steps=10, window=4)
        for i in range(9):
            v = wd.observe(5.0 - i * 0.1)
            assert v == WARMING_UP
        # 10th observation crosses the warmup boundary → real verdict
        v = wd.observe(4.0)
        assert v != WARMING_UP

    def test_n_observed_tracks(self):
        wd = LossWatchdog(warmup_steps=10, window=4)
        for i in range(5):
            wd.observe(5.0)
        assert wd.n_observed == 5


# ════════════════════════════════════════════════════════════════════
# HEALTHY — loss falling
# ════════════════════════════════════════════════════════════════════
class TestHealthy:
    def test_steady_decline_is_healthy(self):
        wd = LossWatchdog(warmup_steps=40, window=10, min_improvement=0.05)
        verdict = WARMING_UP
        # Decline from 5.0 to ~1.0 over 50 steps
        for i in range(50):
            loss = 5.0 - (i / 50) * 4.0
            verdict = wd.observe(loss)
        assert verdict == HEALTHY
        assert "converging" in wd.diagnosis()

    def test_small_but_sufficient_improvement(self):
        wd = LossWatchdog(warmup_steps=40, window=10, min_improvement=0.05)
        verdict = WARMING_UP
        # Early mean ≈ 5.0, recent mean ≈ 4.7 → 6 % improvement > 5 %
        for i in range(50):
            loss = 5.0 - (i / 50) * 0.35
            verdict = wd.observe(loss)
        assert verdict == HEALTHY


# ════════════════════════════════════════════════════════════════════
# SATURATED — loss flat
# ════════════════════════════════════════════════════════════════════
class TestSaturated:
    def test_flat_loss_is_saturated(self):
        wd = LossWatchdog(warmup_steps=40, window=10, min_improvement=0.05)
        verdict = WARMING_UP
        for _ in range(50):
            verdict = wd.observe(4.2)   # perfectly flat
        assert verdict == SATURATED
        assert "flat" in wd.diagnosis()
        assert "learning rate" in wd.diagnosis()

    def test_tiny_improvement_below_threshold_is_saturated(self):
        wd = LossWatchdog(warmup_steps=40, window=10, min_improvement=0.05)
        verdict = WARMING_UP
        # Only 1 % improvement — below the 5 % floor
        for i in range(50):
            loss = 5.0 - (i / 50) * 0.05
            verdict = wd.observe(loss)
        assert verdict == SATURATED


# ════════════════════════════════════════════════════════════════════
# DIVERGING — loss rising / NaN
# ════════════════════════════════════════════════════════════════════
class TestDiverging:
    def test_nan_loss_immediately_diverging(self):
        wd = LossWatchdog(warmup_steps=40, window=10)
        wd.observe(5.0)
        v = wd.observe(float("nan"))
        assert v == DIVERGING
        assert "non-finite" in wd.diagnosis()

    def test_inf_loss_diverging(self):
        wd = LossWatchdog(warmup_steps=40, window=10)
        v = wd.observe(float("inf"))
        assert v == DIVERGING

    def test_rising_loss_is_diverging(self):
        wd = LossWatchdog(
            warmup_steps=40, window=10, divergence_tolerance=0.10,
        )
        verdict = WARMING_UP
        # Loss rises from 2.0 to 4.0 → +100 % > 10 % tolerance
        for i in range(50):
            loss = 2.0 + (i / 50) * 2.0
            verdict = wd.observe(loss)
        assert verdict == DIVERGING
        assert "rising" in wd.diagnosis()

    def test_nan_sticks_even_if_later_finite(self):
        # Once a non-finite loss is seen, the run is considered tainted
        wd = LossWatchdog(warmup_steps=40, window=10)
        wd.observe(float("nan"))
        for _ in range(50):
            v = wd.observe(3.0)
        assert v == DIVERGING


# ════════════════════════════════════════════════════════════════════
# Snapshot / diagnosis
# ════════════════════════════════════════════════════════════════════
class TestReporting:
    def test_snapshot_structure(self):
        wd = LossWatchdog(warmup_steps=10, window=4)
        for i in range(15):
            wd.observe(5.0 - i * 0.2)
        snap = wd.snapshot()
        assert "verdict" in snap
        assert "n_observed" in snap
        assert snap["n_observed"] == 15
        assert "early_mean" in snap
        assert "recent_mean" in snap

    def test_warmup_diagnosis_message(self):
        wd = LossWatchdog(warmup_steps=100, window=10)
        wd.observe(5.0)
        assert "WARMING_UP" in wd.diagnosis()
        assert "1/100" in wd.diagnosis()

    def test_last_verdict_property(self):
        wd = LossWatchdog(warmup_steps=10, window=4)
        wd.observe(5.0)
        assert wd.last_verdict == WARMING_UP


# ════════════════════════════════════════════════════════════════════
# Integration contract — matches the training loop usage
# ════════════════════════════════════════════════════════════════════
class TestTrainingLoopContract:
    def test_observe_returns_string_verdict_every_call(self):
        wd = LossWatchdog(warmup_steps=10, window=4)
        for i in range(20):
            v = wd.observe(5.0 - i * 0.05)
            assert v in {WARMING_UP, HEALTHY, SATURATED, DIVERGING}

    def test_realistic_healthy_then_warns_never(self):
        # A realistic decaying-loss trajectory should never warn
        wd = LossWatchdog(warmup_steps=100, window=20, min_improvement=0.05)
        import math
        verdict = WARMING_UP
        for step in range(150):
            loss = 0.5 + 4.5 * math.exp(-step / 30.0)   # exp decay
            verdict = wd.observe(loss)
        assert verdict == HEALTHY
