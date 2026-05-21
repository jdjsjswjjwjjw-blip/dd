"""
context/ — Sprint 8 (التقرير 3.2): Session + Liquidity context من V19.2.

يحوي:
    - session_mapper:           16 zone × 12 level
    - liquidity_topology_engine: 20+ liquidity feature

facade pattern.
"""

from __future__ import annotations

import os
import sys

_V19_2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v19_2")
if _V19_2 not in sys.path:
    sys.path.insert(0, _V19_2)

try:
    import session_mapper  # noqa: F401
except ImportError:
    pass

try:
    import liquidity_topology_engine  # noqa: F401
except ImportError:
    # liquidity_topology قد يكون في modules/ بدلاً من v19_2/
    try:
        from modules import liquidity_topology_engine  # noqa: F401
    except ImportError:
        pass


__all__ = [
    "session_mapper",
    "liquidity_topology_engine",
]
