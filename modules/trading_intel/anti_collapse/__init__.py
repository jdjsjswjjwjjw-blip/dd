"""
trading_intel.anti_collapse
───────────────────────────
Anti-collapse + anti-overlap + multi-task + utility modules for SSL
fine-tuning, based on the synthesis of Western / Chinese / Russian-school
research addressing the empirical failure modes observed in our 6-month
SSL run:

  1. Direction head collapsed to 100% UP        → SimplexETFClassifier
  2. SSL embeddings overlapped with day_trade   → OrthogonalRepresentationModule
     rules (52.4% on non-event bars = random)
  3. Multi-task tradeoff (v1: magnitude wins,   → DBMTLBalancer
     v2: direction wins, never both)
  4. Loss-landscape mismatch (surrogate ≠ goal) → SharpeRegularizer

All modules are drop-in additions to the HybridModel training loop. They
do not require any data we don't already have.

See docs/RESEARCH_SYNTHESIS.md for the full mapping of seven failure
mechanisms to fourteen proposed solutions across three schools.
"""

from modules.trading_intel.anti_collapse.simplex_etf import (
    SimplexETFConfig,
    SimplexETFClassifier,
    dot_regression_loss,
)
from modules.trading_intel.anti_collapse.orthogonal_rep import (
    OrthogonalityPenaltyConfig,
    OrthogonalRepresentationModule,
    gram_schmidt_project_out,
    orthogonality_penalty,
)
from modules.trading_intel.anti_collapse.dbmtl_balancer import (
    DBMTLConfig,
    DBMTLState,
    DBMTLBalancer,
    compute_dbmtl_loss,
)
from modules.trading_intel.anti_collapse.sharpe_loss import (
    SharpeLossConfig,
    SharpeRegularizer,
    differentiable_sharpe_loss,
)

__all__ = [
    # Anti class-collapse
    "SimplexETFConfig", "SimplexETFClassifier", "dot_regression_loss",
    # Anti rule-overlap
    "OrthogonalityPenaltyConfig", "OrthogonalRepresentationModule",
    "gram_schmidt_project_out", "orthogonality_penalty",
    # Multi-task balancing
    "DBMTLConfig", "DBMTLState", "DBMTLBalancer", "compute_dbmtl_loss",
    # Utility-aligned loss
    "SharpeLossConfig", "SharpeRegularizer", "differentiable_sharpe_loss",
]
