"""
simulators/ — Sprint 8 (التقرير 3.2): Microstructure simulators من V19.2.

يحوي:
    - feature_simulators:  18 محاكي عمق (Absorption، OrderFlow، Informed،
                          Wall Dynamics، Liquidity)
    - wall_depth_simulator: 8 outputs لجدران (يحل wall tracking ناقص)
    - iceberg_simulator:   5 outputs (يحل لا iceberg detection)

facade pattern.
"""

from __future__ import annotations

import os
import sys

_V19_2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v19_2")
if _V19_2 not in sys.path:
    sys.path.insert(0, _V19_2)

try:
    import feature_simulators  # noqa: F401
    import wall_depth_simulator  # noqa: F401
    import iceberg_simulator  # noqa: F401
except ImportError:
    pass


__all__ = [
    "feature_simulators",
    "wall_depth_simulator",
    "iceberg_simulator",
]
