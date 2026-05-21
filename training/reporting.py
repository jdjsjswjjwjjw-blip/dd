"""
training/reporting.py
────────────────────
Sprint 7: facade لـ reporting + calibration helpers من train_v19.py.
"""

from __future__ import annotations


def calibration_metrics(*args, **kwargs):
    """Compute calibration metrics (ECE, Brier, etc).

    See train_v19._calibration_metrics.
    """
    from train_v19 import _calibration_metrics as _impl
    return _impl(*args, **kwargs)


def binary_ece(*args, **kwargs):
    """Expected Calibration Error.

    See train_v19._binary_ece.
    """
    from train_v19 import _binary_ece as _impl
    return _impl(*args, **kwargs)


def feature_coverage_drift_report(*args, **kwargs):
    """Feature coverage / drift report.

    See train_v19._write_feature_coverage_drift_report.
    """
    from train_v19 import _write_feature_coverage_drift_report as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "calibration_metrics",
    "binary_ece",
    "feature_coverage_drift_report",
]
