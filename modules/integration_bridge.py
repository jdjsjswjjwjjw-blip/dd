"""
modules/integration_bridge.py — Phase 6: V19.2 alphas ⇄ DL ensemble bridge.
══════════════════════════════════════════════════════════════════════

الفكرة (من التقرير، المرحلة 6):
  ملف يربط V19.2 alphas (statistical) مع DL ensemble (CatBoost + DeepLOB + LSTM).

  السيناريو:
    1. V19.2 يكتشف 5-15 alphas موثوقة (FDR + permutation + 3-way validated)
    2. DL ensemble يولّد probability per-row (LONG/SHORT/NEUTRAL)
    3. الـ bridge:
       - Entry rule: alpha matches → DL confirms (>= threshold)
       - Wall exit: V19.2 wall_consumed > 0.7 → exit
       - Stop: ATR-based via TripleBarrier
    4. trade decision نهائي للـ paper/live trading

API:
    TradingDecision dataclass
    IntegrationBridge class
    make_trading_decision(df_row, alpha_set, dl_proba, ...) -> TradingDecision
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class Action(Enum):
    HOLD = 0
    OPEN_LONG = 1
    OPEN_SHORT = 2
    CLOSE = 3


@dataclass
class TradingDecision:
    """قرار trade واحد للـ paper/live execution."""
    action: Action
    reason: str
    confidence: float = 0.0    # [0, 1]
    matched_alpha: Optional[str] = None
    dl_proba: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action.name,
            "reason": self.reason,
            "confidence": self.confidence,
            "matched_alpha": self.matched_alpha,
            "dl_proba": self.dl_proba,
            "metadata": self.metadata,
        }


@dataclass
class BridgeConfig:
    """thresholds للـ entry/exit gates."""
    dl_confirm_threshold: float = 0.55   # DL يجب أن يثبّت >= 0.55
    wall_consumed_exit: float = 0.70     # wall_consumed > 0.7 → close
    iceberg_buy_threshold: float = 0.5   # iceberg_side > 0.5 → buy signal
    iceberg_sell_threshold: float = -0.5
    min_alpha_confidence: float = 0.6


class IntegrationBridge:
    """يربط V19.2 alphas مع DL predictions ويُخرج trading decisions.

    Usage:
        bridge = IntegrationBridge(alpha_set, config=BridgeConfig())
        decision = bridge.evaluate_row(row, dl_proba)
        # OR batch:
        decisions = bridge.evaluate_batch(df, dl_proba_arr)
    """

    def __init__(
        self,
        alpha_set,
        config: BridgeConfig = BridgeConfig(),
        regime_library=None,
    ):
        """
        Parameters
        ----------
        alpha_set : AlphaSet (legacy, global alphas)
        config : BridgeConfig
        regime_library : RegimeAlphaLibrary | None (Sprint 13)
            لو مُمرَّر، الـ Bridge يفلتر alphas حسب row['regime_label']
            بدل استخدام alpha_set الموحد. الفكرة الذهبية من
            Regime Analysis Report v2 (③).
        """
        self.alpha_set = alpha_set
        self.config = config
        self.regime_library = regime_library

    def _alphas_for_row(self, row: pd.Series) -> list[dict]:
        """يُرجع candidates relevant لـ row الحالي.

        - لو regime_library متاح: يفلتر حسب row['regime_label']
        - وإلا: يستخدم alpha_set.candidates (السلوك القديم)
        """
        if self.regime_library is not None:
            regime = row.get("regime_label")
            if regime is None:
                # fallback: نستخدم all alphas من كل الـ regimes
                all_a = []
                for r in self.regime_library.regimes():
                    all_a.extend(self.regime_library.get(r))
                return all_a
            return self.regime_library.get(regime)
        return list(self.alpha_set.candidates)

    def _row_matches_alpha(self, row: pd.Series, alpha: dict) -> bool:
        """يفحص لو row يطابق alpha specs (zone + level event + combo filters)."""
        # zone match
        if row.get("zone_full") != alpha.get("zone"):
            return False
        # level event match
        level = alpha.get("level")
        event = alpha.get("event")
        ev_col = f"{level}_event"
        if row.get(ev_col) != event:
            return False
        # combo filters match (basic check on first filter)
        from edge_scanner import SIMULATOR_FILTERS
        filts = alpha.get("filters", [])
        for fk in filts:
            if fk not in SIMULATOR_FILTERS:
                return False
            col, op, thr = SIMULATOR_FILTERS[fk]
            val = pd.to_numeric(row.get(col, 0), errors="coerce")
            if pd.isna(val):
                return False
            if op == ">" and not (val > thr):
                return False
            if op == "<" and not (val < thr):
                return False
        return True

    def _check_wall_exit(self, row: pd.Series) -> bool:
        """wall_consumed > threshold → close signal."""
        wall_consumed = pd.to_numeric(row.get("sim_wall_consumed", 0), errors="coerce")
        if pd.isna(wall_consumed):
            return False
        return float(wall_consumed) > self.config.wall_consumed_exit

    def evaluate_row(
        self,
        row: pd.Series,
        dl_proba: dict[str, float],
        in_position: Optional[Action] = None,
    ) -> TradingDecision:
        """قرار trade لصف واحد.

        Parameters
        ----------
        row : pd.Series - features الصف
        dl_proba : dict('long': p, 'short': p, 'neutral': p)
        in_position : Action.OPEN_LONG/SHORT لو في position، None لو flat

        Returns
        -------
        TradingDecision
        """
        # ── 1. Wall-based exit (priority) ─────────────────────────────────
        if in_position in (Action.OPEN_LONG, Action.OPEN_SHORT):
            if self._check_wall_exit(row):
                return TradingDecision(
                    action=Action.CLOSE,
                    reason="wall_consumed exit",
                    confidence=1.0,
                    metadata={"wall_consumed": float(row.get("sim_wall_consumed", 0))},
                )

        # ── 2. Alpha match + DL confirm ───────────────────────────────────
        # Sprint 13: لو regime_library مُمرَّر، الـ candidates مفلترة حسب
        # row['regime_label'] (الفكرة الذهبية من التقرير ③).
        candidates = self._alphas_for_row(row)
        matched = None
        for alpha in candidates:
            if self._row_matches_alpha(row, alpha):
                matched = alpha
                break

        if matched is None:
            return TradingDecision(
                action=Action.HOLD,
                reason="no alpha match",
                dl_proba=dl_proba,
            )

        direction = matched.get("direction", 0)
        if direction == 1:
            dl_conf = dl_proba.get("long", 0.0)
            if dl_conf >= self.config.dl_confirm_threshold:
                return TradingDecision(
                    action=Action.OPEN_LONG,
                    reason=f"alpha={matched['combo']} + DL long={dl_conf:.2f}",
                    confidence=dl_conf,
                    matched_alpha=matched.get("combo"),
                    dl_proba=dl_proba,
                )
        elif direction == -1:
            dl_conf = dl_proba.get("short", 0.0)
            if dl_conf >= self.config.dl_confirm_threshold:
                return TradingDecision(
                    action=Action.OPEN_SHORT,
                    reason=f"alpha={matched['combo']} + DL short={dl_conf:.2f}",
                    confidence=dl_conf,
                    matched_alpha=matched.get("combo"),
                    dl_proba=dl_proba,
                )

        return TradingDecision(
            action=Action.HOLD,
            reason="alpha matched لكن DL لم يثبّت",
            matched_alpha=matched.get("combo"),
            dl_proba=dl_proba,
        )

    def evaluate_batch(
        self,
        df: pd.DataFrame,
        dl_proba_arr: np.ndarray,
    ) -> list[TradingDecision]:
        """قرارات batch لـ DataFrame كامل.

        Parameters
        ----------
        df : DataFrame مع features
        dl_proba_arr : (n, 3) array - cols = [long, short, neutral]

        Returns
        -------
        list[TradingDecision] طوله = len(df)
        """
        if len(dl_proba_arr) != len(df):
            raise ValueError(
                f"dl_proba_arr length {len(dl_proba_arr)} != df length {len(df)}"
            )
        decisions = []
        for i, (_, row) in enumerate(df.iterrows()):
            proba = {
                "long": float(dl_proba_arr[i, 0]),
                "short": float(dl_proba_arr[i, 1]),
                "neutral": float(dl_proba_arr[i, 2]) if dl_proba_arr.shape[1] > 2 else 0.0,
            }
            decisions.append(self.evaluate_row(row, proba))
        return decisions


__all__ = [
    "Action",
    "TradingDecision",
    "BridgeConfig",
    "IntegrationBridge",
]
