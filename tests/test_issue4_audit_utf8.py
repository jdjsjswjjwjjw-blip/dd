"""Issue 4 — event-gate audit writes are utf-8 (Windows-safe).

`tools/diagnostics/audit_event_gate.write_report` emits non-ASCII (a `═` rule
and Arabic strings). Without explicit `encoding="utf-8"`, Path.write_text uses
the OS default — cp1252 on Windows — and raises:
    UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-69
which `_run_event_gate_audit_safe` swallows, silently losing the audit on
Windows. quant-rigor-guard R8.

Tests lock that every write_text / to_csv in audit_event_gate.py passes
`encoding="utf-8"`, with the report itself round-tripping as utf-8.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics import audit_event_gate as A


def _call_write_report(path):
    A.write_report(
        path,
        verdict="HEALTHY",
        diag={"n_rows": 100},
        overall_rate=0.1,
        per_regime={},
        per_session={},
        components={},
        cvd_contam={},
        warmup={},
        n_rows=100,
    )


def test_write_report_passes_utf8(tmp_path, monkeypatch):
    """TEETH: write_report MUST call write_text with encoding='utf-8'."""
    captured = {}

    def _spy(self, data, *args, **kwargs):
        captured["path"] = self
        captured["data"] = data
        captured["encoding"] = kwargs.get("encoding")
        return len(data)

    monkeypatch.setattr(Path, "write_text", _spy)
    _call_write_report(tmp_path / "rpt.txt")
    assert captured.get("encoding") == "utf-8", (
        f"write_text was called without encoding='utf-8' (got {captured.get('encoding')!r}) "
        f"— Windows will UnicodeEncodeError on the report's `═` / Arabic"
    )


def test_source_writes_carry_utf8():
    """All write_text / to_csv calls inside the audit module declare utf-8.
    A regression that drops the encoding would fail this static check before
    a Windows user re-hits the silently-skipped audit.

    Multi-line aware: each call is parsed up to its closing paren so a kwarg
    on a continuation line still counts."""
    import re
    src = (REPO_ROOT / "tools/diagnostics/audit_event_gate.py").read_text(encoding="utf-8")

    def _calls_have_utf8(name: str) -> None:
        for m in re.finditer(rf"\.{re.escape(name)}\s*\(", src):
            i, depth = m.end(), 1
            while i < len(src) and depth:
                c = src[i]
                if c == "(":   depth += 1
                elif c == ")": depth -= 1
                i += 1
            call_text = src[m.start():i]
            assert 'encoding="utf-8"' in call_text or "encoding='utf-8'" in call_text, (
                f"{name} call missing utf-8 encoding:\n{call_text}"
            )

    _calls_have_utf8("write_text")
    _calls_have_utf8("to_csv")


def test_report_round_trips_as_utf8(tmp_path):
    """smoke: write_report produces a file readable as utf-8 and containing
    the non-ASCII separator that triggered the original error."""
    out = tmp_path / "rpt.txt"
    _call_write_report(out)
    text = out.read_text(encoding="utf-8")
    assert "═" in text
