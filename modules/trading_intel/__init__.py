"""
modules.trading_intel
─────────────────────
Trading intelligence layer — all the enhancements added on top of the
legacy day_trade rule system and SSL backbone.

Top-level public API (most callers only need these):

    from modules.trading_intel import (
        # LOB enhanced reading
        build_lob_tensor_v2_for_bar, N_LOB_CHANNELS_V2, HumanLOBCNN,
        # SSL heads
        ShortTermHeads, AdaptiveTargetHeads,
        # Hybrid + decision
        HybridModel, HybridConfig, decide, TradeDecisionPolicy,
    )

Subpackage layout:
  • trading_intel.lob          — Enhanced LOB feature builder + CNN
  • trading_intel.ssl_heads    — Short-term & adaptive SSL heads + labels
  • trading_intel.hybrid       — Fusion model + decision policy
  • trading_intel.training     — End-to-end training scripts

See `README.md` in this directory for architecture details.
"""

# Re-export the most-used public symbols so callers can write
# `from modules.trading_intel import HybridModel` directly.

from modules.trading_intel.lob.features import (
    N_LOB_CHANNELS_V2,
    CH,
    build_lob_tensor_v2_for_bar,
    stack_bars,
)
from modules.trading_intel.lob.cnn import (
    HumanLOBCNN,
    HumanLOBCNNConfig,
    make_human_lob_cnn,
)
from modules.trading_intel.ssl_heads.short_term import (
    ShortTermHeads,
    ShortTermHeadsConfig,
    ShortTermOutput,
    ShortTermTargets,
    compute_short_term_loss,
)
from modules.trading_intel.ssl_heads.adaptive import (
    AdaptiveTargetHeads,
    AdaptiveTargetConfig,
    AdaptiveTargetOutput,
    AdaptiveTargetTargets,
    compute_adaptive_target_loss,
)
from modules.trading_intel.hybrid.model import (
    HybridModel,
    HybridConfig,
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
    # LOB
    "N_LOB_CHANNELS_V2", "CH", "build_lob_tensor_v2_for_bar", "stack_bars",
    "HumanLOBCNN", "HumanLOBCNNConfig", "make_human_lob_cnn",
    # SSL heads
    "ShortTermHeads", "ShortTermHeadsConfig", "ShortTermOutput", "ShortTermTargets",
    "compute_short_term_loss",
    "AdaptiveTargetHeads", "AdaptiveTargetConfig", "AdaptiveTargetOutput",
    "AdaptiveTargetTargets", "compute_adaptive_target_loss",
    # Hybrid
    "HybridModel", "HybridConfig", "HybridOutput", "HybridTargets",
    "compute_hybrid_loss",
    "TradeDecision", "TradeDecisionPolicy", "decide",
]
