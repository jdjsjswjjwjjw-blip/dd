"""
training/data_loader.py
──────────────────────
Sprint 7: facade لـ data loading functions من train_v19.py.
"""

from __future__ import annotations


def load_training_csv(*args, **kwargs):
    """Load + sanitize training CSV/parquet.

    See train_v19.load_training_csv.
    """
    from train_v19 import load_training_csv as _impl
    return _impl(*args, **kwargs)


def build_event_training_view(*args, **kwargs):
    """Build event-filtered training view (Event Gate applied).

    See train_v19.build_event_training_view.
    """
    from train_v19 import build_event_training_view as _impl
    return _impl(*args, **kwargs)


def handle_rare_classes(*args, **kwargs):
    """Rare class handling (SMOTE/oversample).

    See train_v19.handle_rare_classes.
    """
    from train_v19 import handle_rare_classes as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "load_training_csv",
    "build_event_training_view",
    "handle_rare_classes",
]
