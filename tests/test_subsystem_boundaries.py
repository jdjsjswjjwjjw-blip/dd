"""Verify that subsystem boundaries (SUBSYSTEMS.md) are enforced."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LINT_SCRIPT = REPO_ROOT / "tools" / "check_subsystem_boundaries.py"


def test_lint_script_exists():
    assert LINT_SCRIPT.exists(), f"missing {LINT_SCRIPT}"


def test_subsystem_boundaries_clean():
    """The current codebase must pass the boundary lint."""
    result = subprocess.run(
        [sys.executable, str(LINT_SCRIPT)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"Boundary violations detected:\n{result.stdout}\n{result.stderr}"
    )


def test_lint_detects_synthetic_violation(tmp_path):
    """If we plant a forbidden import in a sandbox copy, the lint must
    catch it. This proves the rules actually fire."""
    # Build a minimal repo skeleton in tmp_path:
    #   tmp_path/prepare_day_trading.py with a forbidden import
    #   tmp_path/tools/check_subsystem_boundaries.py (copy of real)
    skeleton = tmp_path / "sandbox"
    skeleton.mkdir()
    (skeleton / "tools").mkdir()
    # Copy the lint script
    (skeleton / "tools" / "check_subsystem_boundaries.py").write_text(
        LINT_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    # Plant a forbidden import — this is a real SSL training module,
    # not in the shared-helper allowlist
    (skeleton / "prepare_day_trading.py").write_text(
        "from modules.deep_lob.hierarchical_model import HierarchicalLOBTransformer\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "tools/check_subsystem_boundaries.py"],
        capture_output=True, text=True, cwd=skeleton,
    )
    assert result.returncode == 1, "lint should reject the synthetic violation"
    assert "prepare_day_trading.py" in result.stdout
    assert "hierarchical_model" in result.stdout or "deep_lob" in result.stdout


def test_lint_allows_shared_helpers(tmp_path):
    """The shared-helper allowlist (BarSequence, build_cycle_features) must
    NOT trigger a violation."""
    skeleton = tmp_path / "sandbox"
    skeleton.mkdir()
    (skeleton / "tools").mkdir()
    (skeleton / "tools" / "check_subsystem_boundaries.py").write_text(
        LINT_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (skeleton / "prepare_day_trading.py").write_text(
        "from modules.price_cycle.data_structures import BarSequence\n"
        "from modules.price_cycle.feature_pipeline import build_cycle_features\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "tools/check_subsystem_boundaries.py"],
        capture_output=True, text=True, cwd=skeleton,
    )
    assert result.returncode == 0, (
        f"Shared helpers should be allowed:\n{result.stdout}"
    )
