"""
modules.production.response
═══════════════════════════════════════════════════════════════════════════════
PredictionResponse — the structured output the Production Runner emits
to whoever consumes its predictions (execution adapter, paper-trade
recorder, downstream logger).

The dataclass is intentionally flat so it serialises cleanly to JSON
for log streams and post-mortems. Every refusal path is named so the
operator can audit *why* the system stayed out of a trade — not just
*that* it did.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

# Action taxonomy
ACTION_BUY = "BUY"
ACTION_SELL = "SELL"
ACTION_HOLD = "HOLD"          # active prediction but no edge → flat position
ACTION_SILENT = "SILENT"      # refused by a guard — DO NOT trade

ALL_ACTIONS = {ACTION_BUY, ACTION_SELL, ACTION_HOLD, ACTION_SILENT}

# Refusal reason taxonomy. Every SILENT response carries one of these.
REFUSED_NONE = None
REFUSED_LOW_CONFIDENCE = "low_confidence"
REFUSED_LOW_EVENT_PROB = "low_event_prob"
REFUSED_DRIFT_DETECTED = "drift_detected"
REFUSED_NO_CHECKPOINT = "no_checkpoint"
REFUSED_INVALID_INPUT = "invalid_input"
REFUSED_STARTUP = "startup_warmup"

ALL_REFUSAL_REASONS = {
    REFUSED_LOW_CONFIDENCE,
    REFUSED_LOW_EVENT_PROB,
    REFUSED_DRIFT_DETECTED,
    REFUSED_NO_CHECKPOINT,
    REFUSED_INVALID_INPUT,
    REFUSED_STARTUP,
}


@dataclass
class PredictionResponse:
    """One row of production output — structured, JSON-serialisable."""

    # Decision
    action: str                          # one of ACTION_*
    refused_reason: str | None = None    # one of REFUSED_* (None unless SILENT)

    # Raw model output (always populated when the model was actually run)
    event_prob: float = 0.0              # P(event) ∈ [0, 1]
    p_long: float = 0.0                  # P(long | event) ∈ [0, 1]
    p_short: float = 0.0                 # P(short | event) ∈ [0, 1]
    p_neutral: float = 0.0               # P(neutral | event) ∈ [0, 1]
    confidence: float = 0.0              # max-class margin, ∈ [0, 1]

    # Guards diagnostics
    drift_score: float = 0.0             # Mahalanobis or KNN distance
    drift_is_anomaly: bool = False

    # Sizing — opt-in, the runner can leave it at zero
    suggested_size: float = 0.0          # contracts / units, ≥ 0

    # Audit
    ts_emitted: str = ""                 # ISO-8601 UTC

    def __post_init__(self) -> None:
        if self.action not in ALL_ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(ALL_ACTIONS)}, "
                f"got {self.action!r}"
            )
        if self.refused_reason is not None:
            if self.refused_reason not in ALL_REFUSAL_REASONS:
                raise ValueError(
                    f"refused_reason must be None or one of "
                    f"{sorted(ALL_REFUSAL_REASONS)}, "
                    f"got {self.refused_reason!r}"
                )
            if self.action != ACTION_SILENT:
                raise ValueError(
                    f"refused_reason={self.refused_reason!r} requires "
                    f"action=SILENT, got action={self.action!r}"
                )
        else:
            if self.action == ACTION_SILENT:
                raise ValueError(
                    "action=SILENT requires a refused_reason"
                )
        for name in ("event_prob", "p_long", "p_short", "p_neutral",
                     "confidence", "drift_score", "suggested_size"):
            v = getattr(self, name)
            if v < 0.0:
                raise ValueError(f"{name} must be ≥ 0, got {v}")

        if not self.ts_emitted:
            self.ts_emitted = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def silent(
        cls, reason: str, drift_score: float = 0.0,
        drift_is_anomaly: bool = False,
    ) -> "PredictionResponse":
        """Factory for the common case: refuse with a reason."""
        return cls(
            action=ACTION_SILENT,
            refused_reason=reason,
            drift_score=drift_score,
            drift_is_anomaly=drift_is_anomaly,
        )
