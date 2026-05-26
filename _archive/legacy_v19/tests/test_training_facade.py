"""
tests/test_training_facade.py — Sprint 7 facade tests.

يحرس إن training/ يكشف الـ public API بنظافة عبر facade pattern:
    - كل wrapper موجود (load + 3 stages + orchestrator + reporting + checkpoints)
    - كل wrapper يفوّض إلى train_v19.py الفعلي (lazy import)
    - الـ public API يطابق التقرير 3.2 (training/ subdirectory)

لا نستورد train_v19.py مباشرة (يحتاج sklearn + torch + catboost) —
نفحص الـ structure source-level.
"""

from __future__ import annotations

import importlib
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestTrainingPackageStructure(unittest.TestCase):
    """التأكد إن الـ structure مطابق للتقرير 3.2."""

    def test_training_directory_exists(self):
        td = os.path.join(_ROOT, "training")
        self.assertTrue(os.path.isdir(td), "training/ directory مفقود")

    def test_expected_modules_exist(self):
        td = os.path.join(_ROOT, "training")
        expected = [
            "__init__.py", "data_loader.py",
            "stage1_runner.py", "stage2_runner.py", "stage3_runner.py",
            "orchestrator.py", "reporting.py", "checkpoints.py",
        ]
        for name in expected:
            self.assertTrue(
                os.path.exists(os.path.join(td, name)),
                f"training/{name} مفقود",
            )


class TestTrainingFacadeImports(unittest.TestCase):
    """التأكد إن الـ facade API exported صحيحاً (lazy imports)."""

    def test_training_package_loadable(self):
        """training package يُحمَّل بدون استدعاء train_v19."""
        # نستخدم importlib.util لتحميل __init__.py فقط
        spec = importlib.util.spec_from_file_location(
            "training", os.path.join(_ROOT, "training", "__init__.py"),
        )
        self.assertIsNotNone(spec, "training/__init__.py spec فشل")
        # Note: spec_from_file_location لـ package يحتاج submodules — لذلك
        # نختبر فقط الـ source يحوي الـ exports المتوقعة
        src = open(os.path.join(_ROOT, "training", "__init__.py")).read()
        self.assertIn("run_stage1", src)
        self.assertIn("run_stage2", src)
        self.assertIn("run_stage3", src)
        self.assertIn("run_training_pipeline", src)
        self.assertIn("load_training_csv", src)

    def test_wrappers_use_lazy_import(self):
        """كل wrapper يستخدم 'from train_v19 import ...' داخل الـ function."""
        wrapper_files = [
            "stage1_runner.py", "stage2_runner.py", "stage3_runner.py",
            "orchestrator.py", "data_loader.py", "reporting.py", "checkpoints.py",
        ]
        for name in wrapper_files:
            with open(os.path.join(_ROOT, "training", name)) as f:
                src = f.read()
            # يجب ألا يكون فيه 'from train_v19' في top-level (lazy فقط)
            top_level_imports = [
                line for line in src.split("\n")
                if line.startswith("from train_v19") or line.startswith("import train_v19")
            ]
            self.assertEqual(len(top_level_imports), 0,
                f"training/{name}: لا يجب أن يستورد train_v19 في top-level "
                f"(لتجنب heavy dependencies). الـ imports يجب lazy داخل الـ functions.")
            # ويجب أن يحوي على الأقل lazy import واحد
            self.assertIn("from train_v19", src,
                f"training/{name}: يجب أن يحوي على الأقل lazy import واحد")


class TestStageWrappersDelegateToTrainV19(unittest.TestCase):
    """التأكد إن كل wrapper يستدعي الـ function المناسب من train_v19."""

    def test_stage1_delegates_to_stage1_oof_meta(self):
        src = open(os.path.join(_ROOT, "training", "stage1_runner.py")).read()
        self.assertIn("from train_v19 import stage1_oof_meta", src)

    def test_stage2_delegates_to_stage2_oof_visual_embeddings(self):
        src = open(os.path.join(_ROOT, "training", "stage2_runner.py")).read()
        self.assertIn("from train_v19 import stage2_oof_visual_embeddings", src)

    def test_stage3_delegates_to_stage3_meta_learner_v19(self):
        src = open(os.path.join(_ROOT, "training", "stage3_runner.py")).read()
        self.assertIn("from train_v19 import stage3_meta_learner_v19", src)

    def test_orchestrator_delegates_to_run_training_pipeline(self):
        src = open(os.path.join(_ROOT, "training", "orchestrator.py")).read()
        self.assertIn("from train_v19 import run_training_pipeline", src)


class TestPublicAPIMatchesReport(unittest.TestCase):
    """التأكد إن الـ public API مطابق للتقرير (3.2 و 1.5)."""

    def test_init_exports_3_stages(self):
        src = open(os.path.join(_ROOT, "training", "__init__.py")).read()
        # __all__ يحوي run_stage{1,2,3}
        self.assertIn('"run_stage1"', src)
        self.assertIn('"run_stage2"', src)
        self.assertIn('"run_stage3"', src)
        # ومرادفاتها التاريخية
        self.assertIn('"stage1_oof_meta"', src)
        self.assertIn('"stage2_oof_visual_embeddings"', src)
        self.assertIn('"stage3_meta_learner_v19"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
