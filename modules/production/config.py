"""
modules.production.config
═══════════════════════════════════════════════════════════════════════════════
ProductionConfig — every knob the production runner exposes, in one
validated dataclass.

Design philosophy
─────────────────
The confidence threshold is NOT a magic number. Production decisions
that involve real money must come from a calibration on historical
data — a precision-recall curve, an expected-value calculation, or a
Bayesian decision-theoretic optimum. The config has two modes:

  calibrated:    `confidence_threshold` was set from an explicit
                 calibration run. The operator marked it so.
  uncalibrated:  `confidence_threshold` is at its conservative default
                 of 0.85 — but the runner will REFUSE to emit any
                 non-SILENT action until the operator either:
                   (a) flips `allow_uncalibrated=True` (knowingly
                       accepting an arbitrary threshold), OR
                   (b) provides a calibration artifact and re-runs
                       with `confidence_threshold_source="calibrated"`.

This guards against the "I just picked 85% because it sounds high"
failure mode — the system fails LOUDLY (every prediction is SILENT)
rather than silently making bad decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


THRESHOLD_SOURCE_CALIBRATED = "calibrated"
THRESHOLD_SOURCE_UNCALIBRATED = "uncalibrated"
THRESHOLD_SOURCE_OVERRIDDEN = "overridden"

VALID_THRESHOLD_SOURCES = {
    THRESHOLD_SOURCE_CALIBRATED,
    THRESHOLD_SOURCE_UNCALIBRATED,
    THRESHOLD_SOURCE_OVERRIDDEN,
}


@dataclass
class ConfidenceGateConfig:
    """Thresholds that decide whether a model's output is strong enough
    to act on. Both event_prob AND confidence must clear their floors."""

    event_prob_threshold: float = 0.50
    confidence_threshold: float = 0.85
    confidence_threshold_source: str = THRESHOLD_SOURCE_UNCALIBRATED
    allow_uncalibrated: bool = False
    calibration_artifact_path: Path | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.event_prob_threshold <= 1.0):
            raise ValueError(
                f"event_prob_threshold must be in [0, 1], "
                f"got {self.event_prob_threshold}"
            )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], "
                f"got {self.confidence_threshold}"
            )
        if self.confidence_threshold_source not in VALID_THRESHOLD_SOURCES:
            raise ValueError(
                f"confidence_threshold_source must be one of "
                f"{sorted(VALID_THRESHOLD_SOURCES)}, "
                f"got {self.confidence_threshold_source!r}"
            )

    @property
    def is_calibrated(self) -> bool:
        return self.confidence_threshold_source == THRESHOLD_SOURCE_CALIBRATED

    @property
    def is_actionable(self) -> bool:
        """True iff the gate will ever emit non-SILENT actions."""
        return self.is_calibrated or self.allow_uncalibrated or (
            self.confidence_threshold_source == THRESHOLD_SOURCE_OVERRIDDEN
        )


@dataclass
class DriftGuardConfig:
    """Anomaly detection on the live embedding vs the training
    distribution. Drift score above the threshold → SILENT."""

    enabled: bool = False
    method: str = "mahalanobis"           # "mahalanobis" | "knn" | "multitask"
    threshold: float = 3.0                # z-score-like cutoff
    artifact_path: Path | None = None     # training stats file

    def __post_init__(self) -> None:
        if self.method not in ("mahalanobis", "knn", "multitask"):
            raise ValueError(
                f"method must be one of mahalanobis/knn/multitask, "
                f"got {self.method!r}"
            )
        if self.threshold < 0.0:
            raise ValueError(
                f"threshold must be ≥ 0, got {self.threshold}"
            )


@dataclass
class SizingConfig:
    """Position-size scaling. Conservative defaults; off-line calibration
    expected to update."""

    base_size: float = 1.0                # contracts at confidence=1.0
    min_size: float = 0.0                 # don't trade below this
    confidence_scaling: bool = True       # size *= max(0, conf - thr)

    def __post_init__(self) -> None:
        if self.base_size < 0:
            raise ValueError(f"base_size must be ≥ 0, got {self.base_size}")
        if self.min_size < 0:
            raise ValueError(f"min_size must be ≥ 0, got {self.min_size}")


@dataclass
class ProductionConfig:
    """Top-level production config — composes the four sub-configs."""

    checkpoint_path: Path | None = None        # required for the runner to load
    device: str = "cpu"                        # "cpu" | "cuda" | "cuda:0" etc.
    confidence_gate: ConfidenceGateConfig = field(
        default_factory=ConfidenceGateConfig
    )
    drift_guard: DriftGuardConfig = field(default_factory=DriftGuardConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)

    def __post_init__(self) -> None:
        if self.checkpoint_path is not None and not isinstance(
            self.checkpoint_path, Path
        ):
            self.checkpoint_path = Path(self.checkpoint_path)
        if self.drift_guard.artifact_path is not None and not isinstance(
            self.drift_guard.artifact_path, Path
        ):
            self.drift_guard.artifact_path = Path(self.drift_guard.artifact_path)

    @property
    def is_ready_to_trade(self) -> bool:
        """The runner is ready to emit non-SILENT actions only when all
        of: a checkpoint is provided AND the confidence gate is
        actionable (calibrated or explicitly override-allowed)."""
        return (
            self.checkpoint_path is not None
            and self.confidence_gate.is_actionable
        )

    def readiness_reason(self) -> str | None:
        """Returns a human-readable reason if NOT ready, else None."""
        if self.checkpoint_path is None:
            return "checkpoint_path is None"
        if not self.confidence_gate.is_actionable:
            return (
                f"confidence threshold is "
                f"{self.confidence_gate.confidence_threshold_source!r} and "
                f"allow_uncalibrated is False — refuse to emit non-SILENT"
            )
        return None
