"""
discovery/ — Sprint 8 (التقرير 3.2): Statistical discovery layer من V19.2.

يحوي:
    - edge_scanner:    FDR + 3-way + permutation
    - edge_scanner_v2: vectorized (33-182× speedup)
    - cluster_engine:  alphas grouping
    - run_pipeline:    orchestrator
    - show_candidates: inspection tool

+ Phase C discovery من modules/quantum_discovery (Grover, QAOA, VQE, clustering).

facade pattern.
"""

from __future__ import annotations

import os
import sys

_V19_2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v19_2")
if _V19_2 not in sys.path:
    sys.path.insert(0, _V19_2)

try:
    import edge_scanner  # noqa: F401
    import edge_scanner_v2  # noqa: F401
    import cluster_engine  # noqa: F401
    import run_pipeline  # noqa: F401
    import show_candidates  # noqa: F401
except ImportError:
    pass

# Quantum discovery (Sprint 3)
try:
    from modules.quantum_discovery import (  # noqa: F401
        grover_alpha_search,
        qaoa_select_subset,
        vqe_find_alpha_mode,
        quantum_cluster,
    )
except ImportError:
    pass


__all__ = [
    "edge_scanner", "edge_scanner_v2", "cluster_engine",
    "run_pipeline", "show_candidates",
    "grover_alpha_search", "qaoa_select_subset",
    "vqe_find_alpha_mode", "quantum_cluster",
]
