"""trading_intel.ssl_heads — Trading-aware SSL heads + causal label builders.

Two head families that attach to the existing SSL backbone embedding:

  • short_term     — 5 binary/3-way heads at 1-4 bar horizons:
                     wall_break, imbalance_shift, micro_target,
                     liquidity_sweep, gap_fill
  • adaptive       — 3 heads for adaptive TP / regime-aware sizing:
                     max_R_reached (regression), target_bucket (4-way),
                     regime_risk (3-way)

Each head module has a paired *_labels.py with pure-numpy, causal,
session-break-aware label builders.
"""

from modules.trading_intel.ssl_heads.short_term import (
    ShortTermHeads,
    ShortTermHeadsConfig,
    ShortTermOutput,
    ShortTermTargets,
    compute_short_term_loss,
)
from modules.trading_intel.ssl_heads.short_term_labels import (
    build_all_short_term_targets,
    build_next_gap_fill_targets,
    build_next_imbalance_shift_targets,
    build_next_liquidity_sweep_targets,
    build_next_micro_target_targets,
    build_next_wall_break_targets,
)
from modules.trading_intel.ssl_heads.adaptive import (
    AdaptiveTargetHeads,
    AdaptiveTargetConfig,
    AdaptiveTargetOutput,
    AdaptiveTargetTargets,
    compute_adaptive_target_loss,
)
from modules.trading_intel.ssl_heads.adaptive_labels import (
    build_all_adaptive_targets,
    build_max_R_reached_targets,
    build_regime_risk_targets,
    build_target_bucket_targets,
)

__all__ = [
    # Short-term
    "ShortTermHeads", "ShortTermHeadsConfig", "ShortTermOutput",
    "ShortTermTargets", "compute_short_term_loss",
    "build_all_short_term_targets",
    "build_next_gap_fill_targets", "build_next_imbalance_shift_targets",
    "build_next_liquidity_sweep_targets", "build_next_micro_target_targets",
    "build_next_wall_break_targets",
    # Adaptive
    "AdaptiveTargetHeads", "AdaptiveTargetConfig", "AdaptiveTargetOutput",
    "AdaptiveTargetTargets", "compute_adaptive_target_loss",
    "build_all_adaptive_targets",
    "build_max_R_reached_targets", "build_regime_risk_targets",
    "build_target_bucket_targets",
]
