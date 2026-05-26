"""
training/orchestrator.py
───────────────────────
Sprint 7: facade لـ run_training_pipeline من train_v19.py.

يربط Stages 1 → 2 → 3 في pipeline واحد.

استخدام:
    from training import run_training_pipeline
    run_training_pipeline(csv_path='...', output_dir='outputs_v19', ...)
"""

from __future__ import annotations


def run_training_pipeline(*args, **kwargs):
    """Full 3-stage training pipeline.

    See train_v19.run_training_pipeline للـ signature الكامل.
    """
    from train_v19 import run_training_pipeline as _impl
    return _impl(*args, **kwargs)


__all__ = ["run_training_pipeline"]
