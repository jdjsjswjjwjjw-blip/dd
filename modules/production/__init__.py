"""
modules.production
═══════════════════════════════════════════════════════════════════════════════
The production runtime layer — a thin, testable shell around the
existing trained hybrid model.

What it does
────────────
- Loads a hybrid checkpoint via the existing FrozenScaler machinery.
- Runs single-tick inference through the existing `predict_one`.
- Applies three guards that decide whether the prediction is tradable:
    StartupGuard    refuses the first N predictions after process start
    ConfidenceGate  refuses when event_prob OR confidence is below
                    its calibrated threshold; refuses *every*
                    prediction when the threshold is uncalibrated and
                    `allow_uncalibrated=False` — fail-loud by design
    DriftGuard      refuses when the live backbone embedding's
                    distance from the training distribution exceeds
                    a configurable threshold
- Emits a structured PredictionResponse (JSON-serialisable) per call.

What it does NOT do
───────────────────
- Talk to a broker / execution venue (next layer's responsibility).
- Touch the live tick stream (the caller passes raw features in).
- Train or fine-tune (read-only).

Why it's small
──────────────
The Production Runner is the LAST mile before real money. Keeping it
under 500 lines makes the audit surface tractable. Anything bigger
should live behind a proper interface (broker adapter, tick buffer,
etc.) — not inside the runner itself.
"""
from modules.production.config import (
    ConfidenceGateConfig,
    DriftGuardConfig,
    ProductionConfig,
    SizingConfig,
    THRESHOLD_SOURCE_CALIBRATED,
    THRESHOLD_SOURCE_OVERRIDDEN,
    THRESHOLD_SOURCE_UNCALIBRATED,
)
from modules.production.guards import (
    ConfidenceGate,
    DriftGuard,
    StartupGuard,
)
from modules.production.response import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    ACTION_SILENT,
    PredictionResponse,
    REFUSED_DRIFT_DETECTED,
    REFUSED_INVALID_INPUT,
    REFUSED_LOW_CONFIDENCE,
    REFUSED_LOW_EVENT_PROB,
    REFUSED_NO_CHECKPOINT,
    REFUSED_STARTUP,
)
from modules.production.runner import ProductionRunner

__all__ = [
    # response
    "PredictionResponse",
    "ACTION_BUY", "ACTION_SELL", "ACTION_HOLD", "ACTION_SILENT",
    "REFUSED_LOW_CONFIDENCE", "REFUSED_LOW_EVENT_PROB",
    "REFUSED_DRIFT_DETECTED", "REFUSED_NO_CHECKPOINT",
    "REFUSED_INVALID_INPUT", "REFUSED_STARTUP",
    # config
    "ProductionConfig", "ConfidenceGateConfig",
    "DriftGuardConfig", "SizingConfig",
    "THRESHOLD_SOURCE_CALIBRATED", "THRESHOLD_SOURCE_UNCALIBRATED",
    "THRESHOLD_SOURCE_OVERRIDDEN",
    # guards
    "ConfidenceGate", "DriftGuard", "StartupGuard",
    # runner
    "ProductionRunner",
]
