"""
deployment/ — Sprint 8 (التقرير 3.2): Production deployment layer.

يحوي:
    - backtest_v19      : backtest engine
    - walkforward_v19   : walk-forward validation
    - predict_v19       : V19PredictionEngine
    - paper_v19         : paper trading runner
    - live_predictor    : live prediction pipeline
    - integration_bridge: Phase 6 — alphas + DL + walls

facade pattern: lazy imports (heavy dependencies كـ sklearn، tf، catboost).
"""

from __future__ import annotations


def get_backtest_v19():
    """Lazy load backtest_v19 (heavy: sklearn)."""
    import backtest_v19
    return backtest_v19


def get_walkforward_v19():
    """Lazy load walkforward_v19."""
    import walkforward_v19
    return walkforward_v19


def get_predict_v19():
    """Lazy load predict_v19 (heavy: catboost + tf)."""
    import predict_v19
    return predict_v19


def get_paper_v19():
    """Lazy load paper_v19."""
    import paper_v19
    return paper_v19


def get_live_predictor():
    """Lazy load live_predictor."""
    import live_predictor
    return live_predictor


def get_integration_bridge():
    """Lazy load Phase 6 IntegrationBridge."""
    from modules.integration_bridge import IntegrationBridge, BridgeConfig
    return IntegrationBridge, BridgeConfig


__all__ = [
    "get_backtest_v19",
    "get_walkforward_v19",
    "get_predict_v19",
    "get_paper_v19",
    "get_live_predictor",
    "get_integration_bridge",
]
