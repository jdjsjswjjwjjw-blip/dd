"""
modules/quantum_discovery — Phase C: Grover-based alpha discovery.

من التقرير الفصل 6.5:
    - grover_alpha_search: O(√N) discovery (بدل sequential stages)
    - qaoa_optimizer: subset/portfolio optimization
    - vqe_alpha_finder: eigenvalue-based alpha mode detection
    - quantum_clustering: spectral clustering (بدل K-means)
"""

from .grover_alpha_search import (
    GroverDiscoveryConfig,
    GroverSearchResult,
    grover_alpha_search,
    grover_from_edge_scanner,
)
from .qaoa_optimizer import (
    QAOAConfig,
    qaoa_portfolio_weights,
    qaoa_select_subset,
)
from .vqe_alpha_finder import (
    VQEConfig,
    build_hamiltonian,
    inverse_power_iteration,
    power_iteration,
    vqe_find_alpha_mode,
)
from .quantum_clustering import (
    QuantumClusteringConfig,
    cluster_alphas_by_returns,
    quantum_cluster,
)

__all__ = [
    # grover_alpha_search
    "GroverDiscoveryConfig", "GroverSearchResult",
    "grover_alpha_search", "grover_from_edge_scanner",
    # qaoa_optimizer
    "QAOAConfig", "qaoa_portfolio_weights", "qaoa_select_subset",
    # vqe_alpha_finder
    "VQEConfig", "build_hamiltonian",
    "inverse_power_iteration", "power_iteration", "vqe_find_alpha_mode",
    # quantum_clustering
    "QuantumClusteringConfig", "cluster_alphas_by_returns", "quantum_cluster",
]
