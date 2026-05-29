"""
modules.production.guards
═══════════════════════════════════════════════════════════════════════════════
Three guards that sit between the model's raw output and the action
the system emits. Each is a pure stateless function — easy to test,
easy to swap.

  ConfidenceGate   Two-stage threshold (event_prob then confidence).
                   Refuses if either fails — surfaces WHICH one failed.
  DriftGuard       Anomaly-distance check on the live embedding vs the
                   training distribution. Wraps the existing detectors
                   in modules/deep_lob/anomaly_detection.py.
  StartupGuard     Refuses every action for the first N predictions
                   after process start (lets caches and live LOB state
                   warm up).

The guards return either None (pass — let the action through) or a
PredictionResponse with action=SILENT and a refusal reason.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from modules.production.config import (
    ConfidenceGateConfig,
    DriftGuardConfig,
)
from modules.production.response import (
    PredictionResponse,
    REFUSED_DRIFT_DETECTED,
    REFUSED_LOW_CONFIDENCE,
    REFUSED_LOW_EVENT_PROB,
    REFUSED_STARTUP,
)


# ══════════════════════════════════════════════════════════════════════════════
# ConfidenceGate
# ══════════════════════════════════════════════════════════════════════════════
class ConfidenceGate:
    """Two-stage gate: event_prob, then class confidence."""

    def __init__(self, config: ConfidenceGateConfig) -> None:
        self.config = config

    def check(
        self, event_prob: float, confidence: float,
    ) -> Optional[PredictionResponse]:
        """Returns None on pass, or a SILENT PredictionResponse on fail.

        The fail responses populate event_prob/confidence so the audit
        log shows exactly what the model emitted vs what was required.
        """
        if event_prob < self.config.event_prob_threshold:
            return PredictionResponse(
                action="SILENT",
                refused_reason=REFUSED_LOW_EVENT_PROB,
                event_prob=event_prob,
                confidence=confidence,
            )
        if confidence < self.config.confidence_threshold:
            return PredictionResponse(
                action="SILENT",
                refused_reason=REFUSED_LOW_CONFIDENCE,
                event_prob=event_prob,
                confidence=confidence,
            )
        return None


# ══════════════════════════════════════════════════════════════════════════════
# DriftGuard
# ══════════════════════════════════════════════════════════════════════════════
class DriftGuard:
    """Wraps an embedding-anomaly detector. Lazy-imports the detector
    so a missing artifact only fails when the guard is actually used."""

    def __init__(
        self, config: DriftGuardConfig, detector: object | None = None,
    ) -> None:
        """`detector` is an optional pre-built anomaly detector. When None
        and `config.enabled`, the runner is expected to call
        `attach_detector()` before the first prediction."""
        self.config = config
        self._detector = detector

    def attach_detector(self, detector: object) -> None:
        self._detector = detector

    def check(
        self, embedding: np.ndarray,
    ) -> Optional[PredictionResponse]:
        if not self.config.enabled:
            return None
        if self._detector is None:
            # Misconfiguration: drift guard is on but no detector wired.
            # Fail closed — refuse to act rather than skip the check.
            return PredictionResponse.silent(
                reason=REFUSED_DRIFT_DETECTED, drift_score=float("inf"),
                drift_is_anomaly=True,
            )
        result = self._detector.detect(embedding)
        score = float(getattr(result, "score", float("nan")))
        is_anomaly = bool(getattr(result, "is_anomaly", False))
        # Either the detector's own is_anomaly flag, OR our threshold
        # exceeded — be conservative.
        score_too_high = (
            score > self.config.threshold if np.isfinite(score) else True
        )
        if is_anomaly or score_too_high:
            return PredictionResponse.silent(
                reason=REFUSED_DRIFT_DETECTED,
                drift_score=score if np.isfinite(score) else 0.0,
                drift_is_anomaly=True,
            )
        return None


# ══════════════════════════════════════════════════════════════════════════════
# StartupGuard
# ══════════════════════════════════════════════════════════════════════════════
class StartupGuard:
    """Refuses the first `warmup_predictions` calls after construction.

    Useful so the first few live predictions — where caches are cold
    and the live LOB state may be partial — never trigger trades. Pure
    counter, no IO."""

    def __init__(self, warmup_predictions: int = 0) -> None:
        if warmup_predictions < 0:
            raise ValueError(
                f"warmup_predictions must be ≥ 0, "
                f"got {warmup_predictions}"
            )
        self.warmup_predictions = int(warmup_predictions)
        self._count = 0

    @property
    def predictions_seen(self) -> int:
        return self._count

    def check(self) -> Optional[PredictionResponse]:
        self._count += 1
        if self._count <= self.warmup_predictions:
            return PredictionResponse.silent(reason=REFUSED_STARTUP)
        return None
