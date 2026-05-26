"""
training/checkpoints.py
──────────────────────
Sprint 7: facade لـ save/load helpers (artifacts، scalers، meta layouts).
"""

from __future__ import annotations


def load_stage1_meta_feature_names(*args, **kwargs):
    """Load Stage 1 meta feature names من artifacts.

    See train_v19._load_stage1_meta_feature_names.
    """
    from train_v19 import _load_stage1_meta_feature_names as _impl
    return _impl(*args, **kwargs)


def meta_layout_label(*args, **kwargs):
    """Describe meta-feature layout.

    See train_v19._meta_layout_label.
    """
    from train_v19 import _meta_layout_label as _impl
    return _impl(*args, **kwargs)


def fit_scaler_params_from_frame(*args, **kwargs):
    """Fit scaler params (mean/std) من DataFrame.

    See train_v19._fit_scaler_params_from_frame.
    """
    from train_v19 import _fit_scaler_params_from_frame as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "load_stage1_meta_feature_names",
    "meta_layout_label",
    "fit_scaler_params_from_frame",
]
