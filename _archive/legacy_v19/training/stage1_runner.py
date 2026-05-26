"""
training/stage1_runner.py
────────────────────────
Sprint 7 (التقرير 1.5): facade لـ Stage 1 من train_v19.py.

Stage 1 = CatBoost + Regime meta-features.
يبني regime-specific OOF predictions تُستخدم كـ meta-features لـ Stage 2/3.

التقرير: "stage1_refinery.py 182 سطر ← قصير جداً، الجوهر بقي في train_v19.py".
هذا الـ wrapper يعرض الـ public API من train_v19 بدون نسخ الـ logic
(forward-compat: لو train_v19 تطوّر، الـ wrapper يبقى صحيحاً).

استخدام:
    from training import run_stage1
    run_stage1(...)
"""

from __future__ import annotations


def run_stage1(*args, **kwargs):
    """Stage 1 OOF meta-features pipeline.

    See train_v19.stage1_oof_meta للـ signature الكامل.
    """
    from train_v19 import stage1_oof_meta
    return stage1_oof_meta(*args, **kwargs)


# اسم alias متوافق مع التقرير
stage1_oof_meta = run_stage1


__all__ = ["run_stage1", "stage1_oof_meta"]
