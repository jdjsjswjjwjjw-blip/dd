"""Tests for the production runtime layer.

Four layers tested in isolation:
  1. PredictionResponse (dataclass + factory + validation)
  2. Configs (ProductionConfig, ConfidenceGateConfig, etc.)
  3. Guards (ConfidenceGate, DriftGuard, StartupGuard)
  4. ProductionRunner (with a stub model — no checkpoint needed)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.production import (
    ACTION_BUY, ACTION_HOLD, ACTION_SELL, ACTION_SILENT,
    ConfidenceGate, ConfidenceGateConfig,
    DriftGuard, DriftGuardConfig,
    PredictionResponse, ProductionConfig, ProductionRunner,
    REFUSED_DRIFT_DETECTED, REFUSED_INVALID_INPUT,
    REFUSED_LOW_CONFIDENCE, REFUSED_LOW_EVENT_PROB,
    REFUSED_NO_CHECKPOINT, REFUSED_STARTUP,
    SizingConfig, StartupGuard,
    THRESHOLD_SOURCE_CALIBRATED, THRESHOLD_SOURCE_OVERRIDDEN,
    THRESHOLD_SOURCE_UNCALIBRATED,
)


# ════════════════════════════════════════════════════════════════════
# PredictionResponse
# ════════════════════════════════════════════════════════════════════
class TestPredictionResponse:
    def test_valid_buy(self):
        r = PredictionResponse(action=ACTION_BUY)
        assert r.action == "BUY"
        assert r.refused_reason is None
        assert r.ts_emitted   # auto-populated

    def test_silent_factory(self):
        r = PredictionResponse.silent(reason=REFUSED_LOW_CONFIDENCE)
        assert r.action == ACTION_SILENT
        assert r.refused_reason == REFUSED_LOW_CONFIDENCE

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="action"):
            PredictionResponse(action="WHATEVER")

    def test_silent_without_reason_raises(self):
        with pytest.raises(ValueError, match="requires a refused_reason"):
            PredictionResponse(action=ACTION_SILENT)

    def test_reason_without_silent_raises(self):
        with pytest.raises(ValueError, match="requires action=SILENT"):
            PredictionResponse(
                action=ACTION_BUY, refused_reason=REFUSED_LOW_CONFIDENCE,
            )

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError, match="event_prob"):
            PredictionResponse(action=ACTION_BUY, event_prob=-0.1)

    def test_to_dict_is_json_friendly(self):
        import json
        r = PredictionResponse(
            action=ACTION_BUY, event_prob=0.7, confidence=0.9,
        )
        d = r.to_dict()
        # Round-trip via JSON
        s = json.dumps(d)
        assert "BUY" in s and "event_prob" in s


# ════════════════════════════════════════════════════════════════════
# ConfidenceGateConfig
# ════════════════════════════════════════════════════════════════════
class TestConfidenceGateConfig:
    def test_defaults_uncalibrated(self):
        cfg = ConfidenceGateConfig()
        assert cfg.confidence_threshold_source == THRESHOLD_SOURCE_UNCALIBRATED
        assert cfg.is_actionable is False
        assert cfg.is_calibrated is False

    def test_calibrated_is_actionable(self):
        cfg = ConfidenceGateConfig(
            confidence_threshold_source=THRESHOLD_SOURCE_CALIBRATED,
        )
        assert cfg.is_calibrated is True
        assert cfg.is_actionable is True

    def test_allow_uncalibrated_makes_actionable(self):
        cfg = ConfidenceGateConfig(allow_uncalibrated=True)
        assert cfg.is_calibrated is False
        assert cfg.is_actionable is True

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            ConfidenceGateConfig(confidence_threshold=1.5)
        with pytest.raises(ValueError, match="event_prob_threshold"):
            ConfidenceGateConfig(event_prob_threshold=-0.1)

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="confidence_threshold_source"):
            ConfidenceGateConfig(
                confidence_threshold_source="totally_calibrated_yes",
            )


# ════════════════════════════════════════════════════════════════════
# ProductionConfig
# ════════════════════════════════════════════════════════════════════
class TestProductionConfig:
    def test_default_not_ready_to_trade(self):
        cfg = ProductionConfig()
        assert cfg.is_ready_to_trade is False
        assert "checkpoint_path" in cfg.readiness_reason()

    def test_checkpoint_alone_not_ready(self):
        cfg = ProductionConfig(checkpoint_path=Path("/tmp/x.pt"))
        # checkpoint provided but gate uncalibrated → still not ready
        assert cfg.is_ready_to_trade is False
        assert "uncalibrated" in cfg.readiness_reason()

    def test_calibrated_and_checkpoint_ready(self):
        cfg = ProductionConfig(
            checkpoint_path=Path("/tmp/x.pt"),
            confidence_gate=ConfidenceGateConfig(
                confidence_threshold_source=THRESHOLD_SOURCE_CALIBRATED,
            ),
        )
        assert cfg.is_ready_to_trade is True
        assert cfg.readiness_reason() is None

    def test_override_allows_ready(self):
        cfg = ProductionConfig(
            checkpoint_path=Path("/tmp/x.pt"),
            confidence_gate=ConfidenceGateConfig(allow_uncalibrated=True),
        )
        assert cfg.is_ready_to_trade is True

    def test_string_path_coerced(self):
        cfg = ProductionConfig(checkpoint_path="/tmp/y.pt")
        assert isinstance(cfg.checkpoint_path, Path)


# ════════════════════════════════════════════════════════════════════
# ConfidenceGate
# ════════════════════════════════════════════════════════════════════
class TestConfidenceGate:
    def test_pass_through(self):
        cfg = ConfidenceGateConfig(
            event_prob_threshold=0.5, confidence_threshold=0.7,
        )
        g = ConfidenceGate(cfg)
        assert g.check(event_prob=0.8, confidence=0.9) is None

    def test_low_event_prob_blocks(self):
        cfg = ConfidenceGateConfig(
            event_prob_threshold=0.5, confidence_threshold=0.7,
        )
        g = ConfidenceGate(cfg)
        r = g.check(event_prob=0.3, confidence=0.9)
        assert r is not None
        assert r.action == ACTION_SILENT
        assert r.refused_reason == REFUSED_LOW_EVENT_PROB
        assert r.event_prob == 0.3

    def test_low_confidence_blocks(self):
        cfg = ConfidenceGateConfig(
            event_prob_threshold=0.5, confidence_threshold=0.7,
        )
        g = ConfidenceGate(cfg)
        r = g.check(event_prob=0.8, confidence=0.6)
        assert r is not None
        assert r.refused_reason == REFUSED_LOW_CONFIDENCE


# ════════════════════════════════════════════════════════════════════
# DriftGuard
# ════════════════════════════════════════════════════════════════════
@dataclass
class _StubAnomalyResult:
    score: float
    is_anomaly: bool


class _StubDetector:
    def __init__(self, score: float, is_anomaly: bool) -> None:
        self.score = score
        self.is_anomaly = is_anomaly

    def detect(self, _emb):
        return _StubAnomalyResult(self.score, self.is_anomaly)


class TestDriftGuard:
    def test_disabled_passes_through(self):
        g = DriftGuard(DriftGuardConfig(enabled=False))
        assert g.check(np.zeros(8)) is None

    def test_enabled_no_detector_fails_closed(self):
        # Misconfiguration: enabled but no detector → refuse
        g = DriftGuard(DriftGuardConfig(enabled=True))
        r = g.check(np.zeros(8))
        assert r is not None
        assert r.refused_reason == REFUSED_DRIFT_DETECTED

    def test_anomaly_flag_blocks(self):
        g = DriftGuard(
            DriftGuardConfig(enabled=True, threshold=10.0),
            detector=_StubDetector(score=0.5, is_anomaly=True),
        )
        r = g.check(np.zeros(8))
        assert r is not None
        assert r.refused_reason == REFUSED_DRIFT_DETECTED

    def test_score_over_threshold_blocks(self):
        g = DriftGuard(
            DriftGuardConfig(enabled=True, threshold=3.0),
            detector=_StubDetector(score=5.0, is_anomaly=False),
        )
        r = g.check(np.zeros(8))
        assert r is not None
        assert r.drift_score == pytest.approx(5.0)

    def test_low_score_passes(self):
        g = DriftGuard(
            DriftGuardConfig(enabled=True, threshold=3.0),
            detector=_StubDetector(score=1.0, is_anomaly=False),
        )
        assert g.check(np.zeros(8)) is None


# ════════════════════════════════════════════════════════════════════
# StartupGuard
# ════════════════════════════════════════════════════════════════════
class TestStartupGuard:
    def test_zero_warmup_passes_first(self):
        g = StartupGuard(warmup_predictions=0)
        assert g.check() is None

    def test_warmup_blocks_first_n(self):
        g = StartupGuard(warmup_predictions=3)
        for _ in range(3):
            r = g.check()
            assert r is not None
            assert r.refused_reason == REFUSED_STARTUP
        # 4th call should pass
        assert g.check() is None
        assert g.predictions_seen == 4

    def test_negative_warmup_raises(self):
        with pytest.raises(ValueError, match="warmup_predictions"):
            StartupGuard(warmup_predictions=-1)


# ════════════════════════════════════════════════════════════════════
# ProductionRunner (with stub model)
# ════════════════════════════════════════════════════════════════════
class _StubModel:
    """Minimal model — predict() will go through predict_one indirectly,
    but for these tests we monkey-patch predict_one to bypass the model."""
    pass


@dataclass
class _StubScaler:
    pass


class TestProductionRunner:
    def test_predict_without_checkpoint_returns_silent(self):
        runner = ProductionRunner(ProductionConfig())
        r = runner.predict(np.zeros(5), np.zeros(3))
        assert r.action == ACTION_SILENT
        assert r.refused_reason == REFUSED_NO_CHECKPOINT
        assert runner.n_predictions == 1
        assert runner.n_silent == 1

    def test_predict_with_uncalibrated_returns_silent(self):
        # checkpoint provided but gate uncalibrated
        cfg = ProductionConfig(checkpoint_path=Path("/nonexistent.pt"))
        runner = ProductionRunner(cfg)
        # Even if we forcibly attach a stub model, the gate stays uncalibrated
        runner._model = _StubModel()
        runner._scalers = {"daytrade": _StubScaler(), "ssl": _StubScaler()}
        r = runner.predict(np.zeros(5), np.zeros(3))
        assert r.action == ACTION_SILENT
        assert r.refused_reason == REFUSED_NO_CHECKPOINT

    def test_predict_with_override_runs_model(self, monkeypatch):
        # Configure: checkpoint + override-allowed
        cfg = ProductionConfig(
            checkpoint_path=Path("/nonexistent.pt"),
            confidence_gate=ConfidenceGateConfig(
                allow_uncalibrated=True,
                event_prob_threshold=0.3,
                confidence_threshold=0.5,
            ),
            sizing=SizingConfig(base_size=2.0, min_size=0.0),
        )
        runner = ProductionRunner(cfg)
        runner._model = _StubModel()
        runner._scalers = {"daytrade": _StubScaler(), "ssl": _StubScaler()}

        # Monkey-patch predict_one to return a strong BUY signal
        def _fake_predict_one(model, scalers, **kwargs):
            return {
                "event_prob": 0.9, "p_long": 0.8, "p_short": 0.1,
                "p_neutral": 0.1, "confidence": 0.9,
            }
        from modules.trading_intel.hybrid import inference as inf_module
        monkeypatch.setattr(inf_module, "predict_one", _fake_predict_one)
        r = runner.predict(np.zeros(5), np.zeros(3))
        assert r.action == ACTION_BUY
        assert r.refused_reason is None
        assert r.event_prob == pytest.approx(0.9)
        assert r.suggested_size > 0   # confidence-scaled

    def test_predict_low_confidence_silent(self, monkeypatch):
        cfg = ProductionConfig(
            checkpoint_path=Path("/x.pt"),
            confidence_gate=ConfidenceGateConfig(
                allow_uncalibrated=True,
                event_prob_threshold=0.5, confidence_threshold=0.8,
            ),
        )
        runner = ProductionRunner(cfg)
        runner._model = _StubModel()
        runner._scalers = {"daytrade": _StubScaler(), "ssl": _StubScaler()}

        def _fake(model, scalers, **kwargs):
            return {
                "event_prob": 0.9, "p_long": 0.4, "p_short": 0.3,
                "p_neutral": 0.3, "confidence": 0.5,
            }
        from modules.trading_intel.hybrid import inference as inf_module
        monkeypatch.setattr(inf_module, "predict_one", _fake)
        r = runner.predict(np.zeros(5), np.zeros(3))
        assert r.action == ACTION_SILENT
        assert r.refused_reason == REFUSED_LOW_CONFIDENCE

    def test_invalid_input_shape_returns_silent(self):
        cfg = ProductionConfig(
            checkpoint_path=Path("/x.pt"),
            confidence_gate=ConfidenceGateConfig(allow_uncalibrated=True),
        )
        runner = ProductionRunner(cfg)
        runner._model = _StubModel()
        runner._scalers = {"daytrade": _StubScaler()}
        # 2-D daytrade features — invalid
        r = runner.predict(np.zeros((2, 5)), np.zeros(3))
        assert r.action == ACTION_SILENT
        assert r.refused_reason == REFUSED_INVALID_INPUT

    def test_silent_rate_tracked(self, monkeypatch):
        runner = ProductionRunner(ProductionConfig())
        for _ in range(5):
            runner.predict(np.zeros(5), np.zeros(3))
        assert runner.n_predictions == 5
        assert runner.n_silent == 5
        assert runner.silent_rate == 1.0

    def test_warmup_blocks_initial_predictions(self, monkeypatch):
        cfg = ProductionConfig(
            checkpoint_path=Path("/x.pt"),
            confidence_gate=ConfidenceGateConfig(allow_uncalibrated=True),
        )
        runner = ProductionRunner(cfg, warmup_predictions=2)
        runner._model = _StubModel()
        runner._scalers = {"daytrade": _StubScaler(), "ssl": _StubScaler()}

        def _fake(model, scalers, **kwargs):
            return {
                "event_prob": 0.9, "p_long": 0.8, "p_short": 0.1,
                "p_neutral": 0.1, "confidence": 0.9,
            }
        from modules.trading_intel.hybrid import inference as inf_module
        monkeypatch.setattr(inf_module, "predict_one", _fake)
        # First two predictions should be SILENT/startup
        r1 = runner.predict(np.zeros(5), np.zeros(3))
        r2 = runner.predict(np.zeros(5), np.zeros(3))
        assert r1.refused_reason == REFUSED_STARTUP
        assert r2.refused_reason == REFUSED_STARTUP
        # Third should pass through
        r3 = runner.predict(np.zeros(5), np.zeros(3))
        assert r3.action == ACTION_BUY


# ════════════════════════════════════════════════════════════════════
# Boundary check — modules.production must not break existing rules
# ════════════════════════════════════════════════════════════════════
class TestBoundary:
    def test_module_imports_cleanly(self):
        # Mere import shouldn't fail — exercises all submodule loading
        import modules.production
        assert hasattr(modules.production, "ProductionRunner")
        assert hasattr(modules.production, "ProductionConfig")
        assert hasattr(modules.production, "PredictionResponse")
