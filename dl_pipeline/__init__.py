"""
dl_pipeline/ — Sprint 8 (التقرير 3.2): DL preparation + tensor builders.

يحوي:
    - tensor_builder (V19.2):  tensors للـ CNN/LSTM
    - label_engine_v2 (Phase 2): Triple Barrier + MFE/MAE
    - prepare_day_trading (الأحدث): MBO → dataset
    - deeplob_v7ch (Phase 5): 7-channel tensor
    - feature_enrichment (Phase 3): V19.2 features

facade pattern.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V19_2 = os.path.join(_ROOT, "v19_2")
if _V19_2 not in sys.path:
    sys.path.insert(0, _V19_2)


def _safe_import(modname: str):
    try:
        return __import__(modname)
    except ImportError:
        return None


# V19.2
tensor_builder = None
try:
    import importlib.util
    _tb = os.path.join(_V19_2, "data_pipeline.py")
    if os.path.exists(_tb):
        # data_pipeline.py contains tensor builders
        spec = importlib.util.spec_from_file_location("v19_2_data_pipeline", _tb)
        if spec is not None and spec.loader is not None:
            tensor_builder = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(tensor_builder)
            except Exception:
                tensor_builder = None
except Exception:
    tensor_builder = None

# modules/ (Phases 2, 3, 5)
try:
    from modules import label_engine_v2  # noqa: F401
except ImportError:
    label_engine_v2 = None

try:
    from modules import deeplob_v7ch  # noqa: F401
except ImportError:
    deeplob_v7ch = None

try:
    from modules import feature_enrichment  # noqa: F401
except ImportError:
    feature_enrichment = None


__all__ = [
    "tensor_builder",
    "label_engine_v2",
    "deeplob_v7ch",
    "feature_enrichment",
]
