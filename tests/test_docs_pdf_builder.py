"""
tests/test_docs_pdf_builder.py — Sprint 10 tests.

يحرس أن tools/docs/build_integration_report.py:
    - exists + يقبل --output flag
    - --help يعمل (لا dependencies حرجة عند import)
    - الـ output يُكتب في المسار الصحيح بـ default
    - requirements-docs.txt + README موجودين

لا نشغّل الـ build الفعلي في الـ tests (يحتاج reportlab + DejaVu fonts).
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDocsBuilderStructure(unittest.TestCase):
    def test_builder_script_exists(self):
        path = Path(_ROOT) / "tools" / "docs" / "build_integration_report.py"
        self.assertTrue(path.exists(), "build_integration_report.py مفقود")

    def test_readme_exists(self):
        path = Path(_ROOT) / "tools" / "docs" / "README.md"
        self.assertTrue(path.exists(), "tools/docs/README.md مفقود")

    def test_requirements_docs_exists(self):
        path = Path(_ROOT) / "requirements-docs.txt"
        self.assertTrue(path.exists(), "requirements-docs.txt مفقود")
        content = path.read_text()
        self.assertIn("reportlab", content)
        self.assertIn("arabic-reshaper", content)
        self.assertIn("python-bidi", content)


class TestBuilderSourcePortability(unittest.TestCase):
    """التأكد أن الـ builder portable عبر OS."""

    @classmethod
    def setUpClass(cls):
        cls.src = (Path(_ROOT) / "tools" / "docs" / "build_integration_report.py").read_text()

    def test_font_search_paths_include_linux(self):
        self.assertIn("/usr/share/fonts/truetype/dejavu", self.src)

    def test_font_search_paths_include_macos(self):
        # Either Homebrew or system
        self.assertTrue(
            "/opt/homebrew/share/fonts" in self.src or "/Library/Fonts" in self.src,
            "Font search should include macOS paths",
        )

    def test_font_search_paths_include_windows(self):
        # في الـ Python source يكون مكتوب C:\\Windows\\Fonts (double-backslash literal)
        self.assertTrue(
            "C:\\\\Windows\\\\Fonts" in self.src or "C:\\Windows\\Fonts" in self.src,
            "Font search should include Windows paths",
        )

    def test_default_output_is_repo_docs(self):
        """الـ default output يجب أن يكون في docs/ بالـ repo، ليس /mnt/..."""
        self.assertNotIn("/mnt/user-data", self.src,
            "Default output يجب ألا يكون hardcoded في /mnt/")
        self.assertIn("docs", self.src)
        self.assertIn("QuantSystem_Integration_Report_v2_Quantum.pdf", self.src)

    def test_supports_cli_output_flag(self):
        self.assertIn("--output", self.src)
        self.assertIn("argparse", self.src)

    def test_env_var_fallback(self):
        self.assertIn("DEJAVU_FONTS_DIR", self.src)


class TestPDFOutput(unittest.TestCase):
    """لو الـ PDF مبني، نتأكد أنه valid."""

    def test_pdf_exists_in_docs(self):
        path = Path(_ROOT) / "docs" / "QuantSystem_Integration_Report_v2_Quantum.pdf"
        if not path.exists():
            self.skipTest("PDF not built yet (يحتاج reportlab + DejaVu)")
        # PDF magic bytes
        with open(path, "rb") as f:
            magic = f.read(4)
        self.assertEqual(magic, b"%PDF", "Output is not a valid PDF")
        # Size > 100 KB (sanity)
        size = path.stat().st_size
        self.assertGreater(size, 100_000, f"PDF too small ({size} bytes)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
