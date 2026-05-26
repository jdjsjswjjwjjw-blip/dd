"""
modules/quantum_features — Quantum-inspired feature engineering.

من التقرير الفصل 6.2 و 6.4:
    - ghz_states:           3+ particle entangled features
    - tensor_network:       full feature tensor (MPS-like)
    - quantum_walk_features: random walk على market topology

الـ 9 Bell pairs الأصلية في v19_2/quantum/bell_pairs.py
(محفوظة هناك للـ backward compat مع 22 tests الموجودة).
"""

from .ghz_states import (
    STANDARD_GHZ_TRIPLETS,
    add_ghz_features,
    compute_ghz_features,
)
from .tensor_network import (
    build_feature_tensor,
    build_outer_product_tensor,
    flatten_tensor,
    mps_truncate,
    tensor_correlation,
)
from .quantum_walk_features import (
    add_quantum_walk_features,
    build_topology_adjacency,
    compute_quantum_walk_features,
    quantum_walk_step,
)

__all__ = [
    # ghz_states
    "STANDARD_GHZ_TRIPLETS", "add_ghz_features", "compute_ghz_features",
    # tensor_network
    "build_feature_tensor", "build_outer_product_tensor", "flatten_tensor",
    "mps_truncate", "tensor_correlation",
    # quantum_walk_features
    "add_quantum_walk_features", "build_topology_adjacency",
    "compute_quantum_walk_features", "quantum_walk_step",
]
