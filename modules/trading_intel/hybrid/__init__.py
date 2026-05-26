"""trading_intel.hybrid — Fusion model + trading decision policy.

Modules:
  • model.py  — HybridModel: MLP that fuses day_trade features
                + SSL embeddings + (optional) LOB CNN embedding
                into event / direction / confidence predictions.
  • policy.py — Rule-based decision layer that combines day_trade
                rule outputs + HybridModel outputs + AdaptiveTargetHeads
                outputs into a single auditable TradeDecision.
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

__all__ = [
    "HybridConfig", "HybridModel", "HybridOutput", "HybridTargets",
    "compute_hybrid_loss",
    "TradeDecision", "TradeDecisionPolicy", "decide",
]
