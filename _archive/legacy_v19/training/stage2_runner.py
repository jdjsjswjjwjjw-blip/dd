"""
training/stage2_runner.py
────────────────────────
Sprint 7 (التقرير 1.5): facade لـ Stage 2 من train_v19.py.

Stage 2 = OOF DeepLOB visual embeddings.
يبني visual embeddings من LOB tensors عبر CNN، تُمرَّر لـ Stage 3.

استخدام:
    from training import run_stage2
    run_stage2(...)
"""

from __future__ import annotations


def run_stage2(*args, **kwargs):
    """Stage 2 OOF visual embeddings pipeline.

    See train_v19.stage2_oof_visual_embeddings للـ signature الكامل.
    """
    from train_v19 import stage2_oof_visual_embeddings
    return stage2_oof_visual_embeddings(*args, **kwargs)


stage2_oof_visual_embeddings = run_stage2


__all__ = ["run_stage2", "stage2_oof_visual_embeddings"]
