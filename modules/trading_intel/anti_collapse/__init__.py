"""
trading_intel.anti_collapse
───────────────────────────
Anti-collapse + anti-overlap modules for SSL fine-tuning, based on the
synthesis of Chinese-school research (Neural Collapse, OMoE) addressing
the empirical failure modes observed in our 6-month SSL run:

  1. Direction head collapsed to 100% UP        → SimplexETFClassifier
  2. SSL embeddings overlapped with day_trade   → OrthogonalRepresentationModule
     rules (52.4% on non-event bars = random)

Both modules are drop-in additions to the HybridModel training loop. They
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

__all__ = [
    "SimplexETFConfig", "SimplexETFClassifier", "dot_regression_loss",
    "OrthogonalityPenaltyConfig", "OrthogonalRepresentationModule",
    "gram_schmidt_project_out", "orthogonality_penalty",
]
