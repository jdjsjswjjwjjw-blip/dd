"""
modules/regime_alpha_library.py
───────────────────────────────
Sprint 13 (Regime Analysis Report v2، التعديل ③ — الفكرة الذهبية):
Per-Regime Alpha Library — مكتبة alphas منفصلة لكل regime.

═════════════════════════════════════════════════════════════════════════════
الفكرة العميقة (التقرير ص 12)
═════════════════════════════════════════════════════════════════════════════

    alpha ≠ alpha
    alpha في trending ≠ alpha في ranging
    لذا: كل alpha يجب أن تختبر منفصلة في كل regime

المشكلة في discover_alphas الحالي:
    يبحث في كل الـ 12K شمعة معا.
    لو alpha تعمل في trending فقط (10K) لكن تخسر في ranging (2K):
        WR_total = (10K × 70%) + (2K × 30%) / 12K = 63%
    قد ترفض إحصائيا رغم أنها مربحة جدا في trending.

الحل: per-regime discovery + per-regime library.

═════════════════════════════════════════════════════════════════════════════
Schema (JSON)
═════════════════════════════════════════════════════════════════════════════

{
  "version": 1,
  "created_at": "2026-05-21T...",
  "n_samples": 12169,
  "regimes": {
    "trending": [
      {
        "name": "alpha_001",
        "zone": "T_REG_LONDON_Q1",
        "level": "PDH",
        "event": "touch",
        "combo": "LONG_dp_pos",
        "direction": 1,
        "horizon": 6,
        "filters": ["depth_pressure_pos"],
        "stats": {
          "wr": 0.72,
          "sharpe": 1.85,
          "perm_p": 0.018,
          "n_trades": 145,
          "n_days": 22
        }
      }
    ],
    "ranging": [...],
    "volatile": []
  }
}

═════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LIBRARY_SCHEMA_VERSION = 1

# الـ regimes المدعومة (يطابق REGIMES في live_predictor.py / regime_config.py)
DEFAULT_REGIMES: tuple[str, ...] = ("trending", "ranging", "volatile")


@dataclass
class RegimeAlphaLibrary:
    """Per-regime alpha library — wrapper حول dict {regime: [alphas]}."""

    alphas: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = LIBRARY_SCHEMA_VERSION

    def __post_init__(self):
        # default regimes موجودة (حتى لو فاضية)
        for r in DEFAULT_REGIMES:
            if r not in self.alphas:
                self.alphas[r] = []

    # ── Build / Modify ──────────────────────────────────────────────────────

    def add_alpha(self, regime: str, alpha: dict[str, Any]) -> None:
        """يضيف alpha لـ regime معين."""
        if regime not in self.alphas:
            self.alphas[regime] = []
        # validation: required keys
        required = ("name", "direction", "horizon")
        missing = [k for k in required if k not in alpha]
        if missing:
            raise ValueError(f"alpha missing required keys: {missing}")
        self.alphas[regime].append(dict(alpha))

    def add_from_scan_result(self, regime: str, scan_result: dict) -> int:
        """يستورد نتائج edge_scanner_v2 لـ regime معين.

        Returns: عدد الـ alphas المُضافة.
        """
        candidates = scan_result.get("candidates", [])
        n_added = 0
        for i, cand in enumerate(candidates):
            alpha = {
                "name": cand.get("name", f"{regime}_alpha_{i:03d}"),
                "zone": cand.get("zone"),
                "level": cand.get("level"),
                "event": cand.get("event"),
                "combo": cand.get("combo"),
                "direction": cand.get("direction", 1),
                "horizon": cand.get("horizon", 6),
                "filters": cand.get("filters", []),
                "stats": {
                    "wr": float(cand.get("wr", 0.0)),
                    "sharpe": float(cand.get("sharpe", 0.0)),
                    "perm_p": float(cand.get("perm_p", 1.0)),
                    "n_trades": int(cand.get("n_trades", 0)),
                    "n_days": int(cand.get("n_days", 0)),
                },
            }
            self.add_alpha(regime, alpha)
            n_added += 1
        return n_added

    # ── Query ───────────────────────────────────────────────────────────────

    def get(self, regime: str) -> list[dict[str, Any]]:
        """alphas لـ regime معين (empty list لو غير موجود)."""
        return self.alphas.get(regime, [])

    def has(self, regime: str) -> bool:
        """فيها alphas غير فارغة لـ regime؟"""
        return len(self.alphas.get(regime, [])) > 0

    def regimes(self) -> list[str]:
        """قائمة الـ regimes الموجودة."""
        return list(self.alphas.keys())

    def total_alphas(self) -> int:
        """مجموع الـ alphas عبر كل الـ regimes."""
        return sum(len(v) for v in self.alphas.values())

    def summary(self) -> dict[str, Any]:
        """ملخص للتشخيص."""
        return {
            "version": self.version,
            "total_alphas": self.total_alphas(),
            "per_regime_counts": {r: len(a) for r, a in self.alphas.items()},
            "regimes_with_alphas": [r for r, a in self.alphas.items() if a],
            "metadata": dict(self.metadata),
        }

    # ── Serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "metadata": dict(self.metadata),
            "regimes": {r: list(a) for r, a in self.alphas.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegimeAlphaLibrary":
        version = data.get("version", LIBRARY_SCHEMA_VERSION)
        if version != LIBRARY_SCHEMA_VERSION:
            warnings.warn(
                f"Library version {version} != current {LIBRARY_SCHEMA_VERSION}. "
                f"Schema migration قد يكون ضروري.",
                stacklevel=2,
            )
        return cls(
            alphas={r: list(a) for r, a in data.get("regimes", {}).items()},
            metadata=dict(data.get("metadata", {})),
            version=version,
        )

    def save(self, path: str | Path) -> None:
        """يحفظ JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        # timestamp tracking
        data.setdefault("metadata", {})["saved_at"] = datetime.now(timezone.utc).isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeAlphaLibrary":
        """يحمّل JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Library not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ── Filtering & Selection ──────────────────────────────────────────────

    def filter_by_stats(
        self,
        min_wr: float = 0.0,
        min_sharpe: float = 0.0,
        max_perm_p: float = 1.0,
        min_n_trades: int = 0,
    ) -> "RegimeAlphaLibrary":
        """يُرجع library جديدة بـ alphas تستوفي الشروط."""
        filtered: dict[str, list[dict[str, Any]]] = {}
        for regime, alphas in self.alphas.items():
            kept = []
            for a in alphas:
                stats = a.get("stats", {})
                if (
                    stats.get("wr", 0.0) >= min_wr
                    and stats.get("sharpe", 0.0) >= min_sharpe
                    and stats.get("perm_p", 1.0) <= max_perm_p
                    and stats.get("n_trades", 0) >= min_n_trades
                ):
                    kept.append(a)
            filtered[regime] = kept
        return RegimeAlphaLibrary(
            alphas=filtered,
            metadata={
                **self.metadata,
                "filtered_with": {
                    "min_wr": min_wr, "min_sharpe": min_sharpe,
                    "max_perm_p": max_perm_p, "min_n_trades": min_n_trades,
                },
            },
            version=self.version,
        )

    def top_k_per_regime(self, k: int = 10, sort_by: str = "sharpe") -> "RegimeAlphaLibrary":
        """يحتفظ بـ top-k لكل regime مرتّبة بـ stat معيّن."""
        top: dict[str, list[dict[str, Any]]] = {}
        for regime, alphas in self.alphas.items():
            sorted_a = sorted(
                alphas,
                key=lambda a: a.get("stats", {}).get(sort_by, 0.0),
                reverse=True,
            )
            top[regime] = sorted_a[:k]
        return RegimeAlphaLibrary(
            alphas=top,
            metadata={**self.metadata, "top_k": k, "sort_by": sort_by},
            version=self.version,
        )


# ════════════════════════════════════════════════════════════════════════════
# Convenience: merge multiple libraries
# ════════════════════════════════════════════════════════════════════════════


def merge_libraries(*libraries: RegimeAlphaLibrary) -> RegimeAlphaLibrary:
    """يدمج عدة libraries (يحافظ على كل الـ alphas دون duplicates by name)."""
    merged: dict[str, list[dict[str, Any]]] = {r: [] for r in DEFAULT_REGIMES}
    seen_names: dict[str, set] = {r: set() for r in DEFAULT_REGIMES}

    for lib in libraries:
        for regime, alphas in lib.alphas.items():
            if regime not in merged:
                merged[regime] = []
                seen_names[regime] = set()
            for a in alphas:
                name = a.get("name")
                if name and name in seen_names[regime]:
                    continue
                merged[regime].append(a)
                if name:
                    seen_names[regime].add(name)

    return RegimeAlphaLibrary(
        alphas=merged,
        metadata={"merged_from": len(libraries)},
    )
