"""trading_intel.hybrid — Fusion model + trading decision policy + inference.

Modules:
  • model.py     — HybridModel: MLP that fuses day_trade features
                   + SSL embeddings + (optional) LOB CNN embedding
                   into event / direction / confidence predictions.
  • policy.py    — Rule-based decision layer that combines day_trade
                   rule outputs + HybridModel outputs + AdaptiveTargetHeads
                   outputs into a single auditable TradeDecision.
  • inference.py — Live-trading-safe loaders + frozen-scaler helpers.
                   Use these from any live serving path to avoid the
                   pipeline-issue failure modes (normalization drift,
                   pandas-in-the-loop, fit-at-inference-time).
"""

from modules.trading_intel.hybrid.model import (
    HybridConfig,
    HybridModel,
    HybridOutput,
    HybridTargets,
    compute_hybrid_loss,
)
from modules.trading_intel.hybrid.policy import (
    TradeDecision,
    TradeDecisionPolicy,
    decide,
)
from modules.trading_intel.hybrid.inference import (
    FrozenScaler,
    apply_saved_scalers,
    load_hybrid_for_inference,
    predict_one,
)

__all__ = [
    "HybridConfig", "HybridModel", "HybridOutput", "HybridTargets",
    "compute_hybrid_loss",
    "TradeDecision", "TradeDecisionPolicy", "decide",
    # Inference helpers (live-trading-safe)
    "FrozenScaler", "apply_saved_scalers", "load_hybrid_for_inference",
    "predict_one",
]
