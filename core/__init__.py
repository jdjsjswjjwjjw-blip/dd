"""
core/ — Sprint 8 (التقرير 3.2): Core infrastructure من V19.2.

يحوي:
    - market_specs:    تكاليف موحدة + tick sizes + sessions
    - statistics_module: BH-FDR + permutation tests
    - fix_ohlc:        تنظيف OHLC outliers من MBO
    - combine_months:  دمج شهور (يحل manual concatenation errors)
    - combine_raw:     دمج MBO/MBP خام (duplicates، ترتيب)

facade pattern: re-export من v19_2/ بدون نسخ.
backward compat: الـ imports القديمة (from v19_2.market_specs ...) تبقى.
"""

from __future__ import annotations

import os
import sys

_V19_2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v19_2")
if _V19_2 not in sys.path:
    sys.path.insert(0, _V19_2)

# Re-export
try:
    import market_specs  # noqa: F401
    import statistics_module  # noqa: F401
    import fix_ohlc  # noqa: F401
    import combine_months  # noqa: F401
    import combine_raw  # noqa: F401
except ImportError:
    # graceful: module load بدون كسر لو v19_2 ناقص
    pass


__all__ = [
    "market_specs",
    "statistics_module",
    "fix_ohlc",
    "combine_months",
    "combine_raw",
]
