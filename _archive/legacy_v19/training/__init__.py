"""
training/ — Sprint 7: 3-stage training facade.

يحلّ المشكلة #5 من التقرير (train_v19.py Bloat) بـ facade pattern:
    - الـ logic يبقى في train_v19.py (4707 سطر) — لا نقل خطر
    - الـ public API يُكشف عبر training/*.py wrappers
    - المستخدم يستورد من training/ بنظافة

استخدام:
    from training import (
        load_training_csv,
        run_stage1, run_stage2, run_stage3,
        run_training_pipeline,
    )

التقرير 1.5 (Bloat في train_v19.py):
    stage1_refinery.py    182 سطر ← قصير جداً
    stage2_catboost.py    243 سطر
    stage3_train.py        70 سطر ← stub فقط!
    → التقسيم تم نظرياً لكن الجوهر بقي في train_v19.py
    → Sprint 7 حلّ هذا بـ wrappers (facade pattern)
"""

from .data_loader import (
    build_event_training_view,
    handle_rare_classes,
    load_training_csv,
)
from .stage1_runner import run_stage1, stage1_oof_meta
from .stage2_runner import run_stage2, stage2_oof_visual_embeddings
from .stage3_runner import run_stage3, stage3_meta_learner_v19
from .orchestrator import run_training_pipeline
from .reporting import (
    binary_ece,
    calibration_metrics,
    feature_coverage_drift_report,
)
from .checkpoints import (
    fit_scaler_params_from_frame,
    load_stage1_meta_feature_names,
    meta_layout_label,
)

__all__ = [
    # data_loader
    "load_training_csv", "build_event_training_view", "handle_rare_classes",
    # stages
    "run_stage1", "stage1_oof_meta",
    "run_stage2", "stage2_oof_visual_embeddings",
    "run_stage3", "stage3_meta_learner_v19",
    # orchestrator
    "run_training_pipeline",
    # reporting
    "calibration_metrics", "binary_ece", "feature_coverage_drift_report",
    # checkpoints
    "load_stage1_meta_feature_names", "meta_layout_label",
    "fit_scaler_params_from_frame",
]
