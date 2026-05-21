"""
training/stage3_runner.py
────────────────────────
Sprint 7 (التقرير 1.5): facade لـ Stage 3 من train_v19.py.

Stage 3 = MetaLearner LSTM ensemble.
يدمج Stage 1 (CatBoost) + Stage 2 (DeepLOB visual) عبر LSTM.

استخدام:
    from training import run_stage3
    run_stage3(...)
"""

from __future__ import annotations


def run_stage3(*args, **kwargs):
    """Stage 3 MetaLearner LSTM pipeline.

    See train_v19.stage3_meta_learner_v19 للـ signature الكامل.
    """
    from train_v19 import stage3_meta_learner_v19
    return stage3_meta_learner_v19(*args, **kwargs)


stage3_meta_learner_v19 = run_stage3


__all__ = ["run_stage3", "stage3_meta_learner_v19"]
