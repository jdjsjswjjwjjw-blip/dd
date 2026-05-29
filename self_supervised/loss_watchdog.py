"""
self_supervised/loss_watchdog.py
═══════════════════════════════════════════════════════════════════════════════
Early-loss watchdog for SSL pretraining. Watches the first N optimisation
steps and decides — fast — whether the run is converging, saturating, or
diverging, so the operator doesn't waste a GPU evening on a dead run.

Why a watchdog and not "just look at the loss"
──────────────────────────────────────────────
Saturation is silent. A run with a bad learning rate or bad weight
init produces a loss that is *finite* and *stable* — it just never
goes down. The training loop's existing NaN-skip guard doesn't catch
this because the loss isn't NaN; it's flat. By the time a human
notices "the loss has been 4.2 for twenty minutes", twenty minutes of
GPU are gone. This watchdog formalises the "is it actually learning?"
check into a per-step contract.

What it measures
────────────────
Over the first `warmup_steps`, it records each step's total loss.
After warmup, it compares a window of recent losses against the
window of earliest losses and computes the relative improvement:

    improvement = (early_mean - recent_mean) / |early_mean|

Verdict
───────
DIVERGING    any NaN/Inf loss seen, OR recent_mean > early_mean by
             more than `divergence_tolerance` (loss going UP).
SATURATED    improvement < min_improvement (loss flat — bad LR / init).
HEALTHY      improvement ≥ min_improvement (loss falling as expected).
WARMING_UP   fewer than warmup_steps observed so far — no verdict yet.

Usage
─────
    wd = LossWatchdog(warmup_steps=100, min_improvement=0.05)
    for step, batch in enumerate(loader):
        loss = train_step(...)
        verdict = wd.observe(float(loss))
        if verdict == "DIVERGING":
            raise RuntimeError(wd.diagnosis())
        if step == wd.warmup_steps and verdict == "SATURATED":
            print(f"⚠️  {wd.diagnosis()}")
            # operator decides whether to abort

The watchdog is pure-ish: it holds a rolling buffer but has no IO and
no torch dependency, so it's trivially unit-testable with plain floats.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


# Verdicts
WARMING_UP = "WARMING_UP"
HEALTHY = "HEALTHY"
SATURATED = "SATURATED"
DIVERGING = "DIVERGING"


@dataclass
class WatchdogConfig:
    """Knobs for the early-loss watchdog."""

    warmup_steps: int = 100          # steps to observe before verdict
    window: int = 20                 # size of the early/recent comparison windows
    min_improvement: float = 0.05    # relative drop required to call HEALTHY
    divergence_tolerance: float = 0.10  # relative rise that triggers DIVERGING

    def __post_init__(self) -> None:
        if self.warmup_steps < 2:
            raise ValueError(
                f"warmup_steps must be ≥ 2, got {self.warmup_steps}"
            )
        if self.window < 1:
            raise ValueError(f"window must be ≥ 1, got {self.window}")
        if 2 * self.window > self.warmup_steps:
            raise ValueError(
                f"need warmup_steps ≥ 2*window so the early and recent "
                f"windows don't overlap: warmup_steps={self.warmup_steps}, "
                f"window={self.window}"
            )
        if self.min_improvement < 0:
            raise ValueError(
                f"min_improvement must be ≥ 0, got {self.min_improvement}"
            )
        if self.divergence_tolerance < 0:
            raise ValueError(
                f"divergence_tolerance must be ≥ 0, got "
                f"{self.divergence_tolerance}"
            )


class LossWatchdog:
    """Stateful early-loss monitor. Feed it per-step losses via
    `observe()`; it returns the current verdict each call."""

    def __init__(
        self,
        warmup_steps: int = 100,
        window: int = 20,
        min_improvement: float = 0.05,
        divergence_tolerance: float = 0.10,
        config: WatchdogConfig | None = None,
    ) -> None:
        self.config = config or WatchdogConfig(
            warmup_steps=warmup_steps,
            window=window,
            min_improvement=min_improvement,
            divergence_tolerance=divergence_tolerance,
        )
        # Early window: the FIRST `window` losses (frozen once full).
        self._early: list[float] = []
        # Recent window: a rolling buffer of the most recent `window` losses.
        self._recent: deque[float] = deque(maxlen=self.config.window)
        self._n_observed = 0
        self._saw_nonfinite = False
        self._last_verdict = WARMING_UP

    # ── convenience accessors ───────────────────────────────────────
    @property
    def warmup_steps(self) -> int:
        return self.config.warmup_steps

    @property
    def n_observed(self) -> int:
        return self._n_observed

    @property
    def last_verdict(self) -> str:
        return self._last_verdict

    # ── core ────────────────────────────────────────────────────────
    def observe(self, loss: float) -> str:
        """Record one step's loss and return the current verdict."""
        self._n_observed += 1

        if not math.isfinite(loss):
            self._saw_nonfinite = True
            self._last_verdict = DIVERGING
            return DIVERGING

        # Fill the early window first (frozen once full)
        if len(self._early) < self.config.window:
            self._early.append(loss)
        self._recent.append(loss)

        if self._n_observed < self.config.warmup_steps:
            self._last_verdict = WARMING_UP
            return WARMING_UP

        self._last_verdict = self._classify()
        return self._last_verdict

    def _classify(self) -> str:
        if self._saw_nonfinite:
            return DIVERGING
        early_mean = sum(self._early) / len(self._early)
        recent_mean = sum(self._recent) / len(self._recent)
        denom = abs(early_mean) if abs(early_mean) > 1e-12 else 1e-12

        # Loss going UP relative to start → diverging
        rise = (recent_mean - early_mean) / denom
        if rise > self.config.divergence_tolerance:
            return DIVERGING

        # Loss flat → saturated
        improvement = (early_mean - recent_mean) / denom
        if improvement < self.config.min_improvement:
            return SATURATED

        return HEALTHY

    # ── reporting ───────────────────────────────────────────────────
    def diagnosis(self) -> str:
        """Human-readable explanation of the current state + a hint."""
        if self._saw_nonfinite:
            return (
                f"DIVERGING: non-finite loss observed at step "
                f"{self._n_observed}. Lower the learning rate or check "
                f"for exploding gradients / bad targets."
            )
        if self._n_observed < self.config.warmup_steps:
            return (
                f"WARMING_UP: {self._n_observed}/{self.config.warmup_steps} "
                f"steps observed — no verdict yet."
            )
        early_mean = sum(self._early) / len(self._early)
        recent_mean = sum(self._recent) / len(self._recent)
        denom = abs(early_mean) if abs(early_mean) > 1e-12 else 1e-12
        improvement = (early_mean - recent_mean) / denom
        verdict = self._classify()
        base = (
            f"{verdict}: early_mean={early_mean:.4f} → "
            f"recent_mean={recent_mean:.4f} "
            f"(improvement={improvement:+.1%} over {self.config.warmup_steps} steps)"
        )
        if verdict == SATURATED:
            return base + (
                " — loss is flat. Likely causes: learning rate too low, "
                "weight init poor, or the task is degenerate (check "
                "target variance). Try LR×3 or re-init."
            )
        if verdict == DIVERGING:
            return base + " — loss rising. Lower the learning rate."
        return base + " — converging as expected."

    def snapshot(self) -> dict:
        """Structured state for logging / JSON."""
        early_mean = (
            sum(self._early) / len(self._early) if self._early else float("nan")
        )
        recent_mean = (
            sum(self._recent) / len(self._recent)
            if self._recent else float("nan")
        )
        return {
            "verdict": self._last_verdict,
            "n_observed": self._n_observed,
            "warmup_steps": self.config.warmup_steps,
            "early_mean": float(early_mean),
            "recent_mean": float(recent_mean),
            "saw_nonfinite": self._saw_nonfinite,
        }
