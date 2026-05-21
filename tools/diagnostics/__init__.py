"""Diagnostic scripts for QuantSystem.

Each script in this package adds the project root to sys.path automatically
so legacy imports like ``from regime_config import ...`` and
``from modules.xxx import ...`` keep working when scripts are executed
directly (``python tools/diagnostics/diagnose_xxx.py``).
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
