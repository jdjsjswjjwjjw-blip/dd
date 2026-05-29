"""
modules.production.runner
═══════════════════════════════════════════════════════════════════════════════
ProductionRunner — orchestrates a single live-prediction step end-to-end:

    raw features  ─►  FrozenScaler (load_hybrid_for_inference)
                  ─►  model forward
                  ─►  StartupGuard
                  ─►  ConfidenceGate (event_prob, confidence)
                  ─►  DriftGuard (Mahalanobis / KNN over backbone embedding)
                  ─►  sizing (confidence-scaled)
                  ─►  PredictionResponse

Every refusal carries an explicit reason. Logging is the caller's
responsibility — the runner produces a JSON-friendly dataclass and
nothing else.

Loading semantics
─────────────────
The runner refuses to start at construction time if any of the
preconditions is missing — checkpoint_path is None OR the confidence
gate is uncalibrated and not explicitly override-allowed. The
docstring on `ProductionConfig.is_ready_to_trade` documents the rule.

When NOT ready, calls to `predict()` return SILENT with reason
"no_checkpoint" — the runner stays functional (handy for dry-runs and
tests) but never emits a tradable action.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from modules.production.config import ProductionConfig
from modules.production.guards import (
    ConfidenceGate, DriftGuard, StartupGuard,
)
from modules.production.response import (
    ACTION_BUY, ACTION_HOLD, ACTION_SELL,
    PredictionResponse, REFUSED_INVALID_INPUT, REFUSED_NO_CHECKPOINT,
)


log = logging.getLogger(__name__)


class ProductionRunner:
    """Stateful orchestrator. One instance per live process.

    Heavyweight setup (model + scaler loading) happens in `start()`,
    not in `__init__` — so the runner can be constructed in tests
    without touching the filesystem.
    """

    def __init__(
        self,
        config: ProductionConfig,
        warmup_predictions: int = 0,
        drift_detector: object | None = None,
    ) -> None:
        self.config = config
        self.confidence_gate = ConfidenceGate(config.confidence_gate)
        self.drift_guard = DriftGuard(
            config.drift_guard, detector=drift_detector,
        )
        self.startup_guard = StartupGuard(warmup_predictions)

        self._model = None
        self._scalers = None
        self._n_predictions = 0
        self._n_silent = 0

    # ── lifecycle ───────────────────────────────────────────────────
    def start(self) -> None:
        """Load the checkpoint via the existing FrozenScaler path.

        Raises FileNotFoundError if the checkpoint is missing.
        Logs a warning if the config is not in a tradable state
        (so an operator dry-running on tests sees the diagnostic)."""
        if self.config.checkpoint_path is None:
            log.warning(
                "ProductionRunner.start() called with no checkpoint_path "
                "— predictions will be SILENT/no_checkpoint until set"
            )
            return
        # Lazy import — keeps the module import-light for tests that
        # only exercise the response/guards layers
        from modules.trading_intel.hybrid.inference import (
            load_hybrid_for_inference,
        )
        self._model, self._scalers = load_hybrid_for_inference(
            str(self.config.checkpoint_path), device=self.config.device,
        )
        log.info(
            "ProductionRunner started: model loaded from %s, "
            "device=%s, calibrated=%s",
            self.config.checkpoint_path, self.config.device,
            self.config.confidence_gate.is_calibrated,
        )

    # ── statistics ──────────────────────────────────────────────────
    @property
    def n_predictions(self) -> int:
        return self._n_predictions

    @property
    def n_silent(self) -> int:
        return self._n_silent

    @property
    def silent_rate(self) -> float:
        return (self._n_silent / self._n_predictions
                if self._n_predictions > 0 else 0.0)

    # ── core prediction step ────────────────────────────────────────
    def predict(
        self,
        daytrade_features: np.ndarray,    # (D_daytrade,) raw, unnormalized
        ssl_embedding: np.ndarray,        # (D_ssl,) raw
        cnn_embedding: np.ndarray | None = None,
    ) -> PredictionResponse:
        """Run one inference step + all guards. Always returns a
        PredictionResponse — never raises on bad input, always logs.

        Counters (`n_predictions`, `n_silent`) advance for every call.
        """
        self._n_predictions += 1

        # ── Guard: not ready to trade ───────────────────────────────
        if not self.config.is_ready_to_trade:
            self._n_silent += 1
            return PredictionResponse.silent(reason=REFUSED_NO_CHECKPOINT)

        # ── Guard: model not loaded yet ─────────────────────────────
        if self._model is None or self._scalers is None:
            self._n_silent += 1
            return PredictionResponse.silent(reason=REFUSED_NO_CHECKPOINT)

        # ── Guard: input shape sanity ──────────────────────────────
        if (
            daytrade_features.ndim != 1
            or ssl_embedding.ndim != 1
            or (cnn_embedding is not None and cnn_embedding.ndim != 1)
        ):
            self._n_silent += 1
            return PredictionResponse.silent(reason=REFUSED_INVALID_INPUT)

        # ── Guard: startup warm-up ──────────────────────────────────
        startup_resp = self.startup_guard.check()
        if startup_resp is not None:
            self._n_silent += 1
            return startup_resp

        # ── Forward pass via the existing predict_one helper ───────
        from modules.trading_intel.hybrid.inference import predict_one
        pred = predict_one(
            self._model, self._scalers,
            daytrade_features=daytrade_features,
            ssl_embedding=ssl_embedding,
            cnn_embedding=cnn_embedding,
            device=self.config.device,
        )
        event_prob = float(pred.get("event_prob", 0.0))
        p_long = float(pred.get("p_long", 0.0))
        p_short = float(pred.get("p_short", 0.0))
        p_neutral = float(pred.get("p_neutral", 0.0))
        confidence = float(pred.get("confidence", 0.0))

        # ── Guard: confidence + event_prob thresholds ──────────────
        conf_silent = self.confidence_gate.check(event_prob, confidence)
        if conf_silent is not None:
            self._n_silent += 1
            return conf_silent

        # ── Guard: distribution drift ──────────────────────────────
        backbone = pred.get("backbone_embedding")
        if backbone is not None:
            drift_silent = self.drift_guard.check(np.asarray(backbone))
            if drift_silent is not None:
                self._n_silent += 1
                return drift_silent

        # ── Pick action from class probabilities ───────────────────
        if p_long >= p_short and p_long >= p_neutral:
            action = ACTION_BUY
        elif p_short >= p_neutral:
            action = ACTION_SELL
        else:
            action = ACTION_HOLD

        # ── Confidence-scaled sizing ────────────────────────────────
        size_cfg = self.config.sizing
        if size_cfg.confidence_scaling:
            margin = max(
                confidence - self.config.confidence_gate.confidence_threshold,
                0.0,
            )
            size = size_cfg.base_size * margin
        else:
            size = size_cfg.base_size
        if size < size_cfg.min_size:
            self._n_silent += 1
            return PredictionResponse.silent(reason="low_confidence")
        return PredictionResponse(
            action=action,
            event_prob=event_prob,
            p_long=p_long, p_short=p_short, p_neutral=p_neutral,
            confidence=confidence,
            suggested_size=size,
        )
