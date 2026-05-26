"""
modules/trade_decision_policy.py
═══════════════════════════════════════════════════════════════════════════════
Trade-decision policy that fuses:
  • day_trade rule outputs (event_flag, event_direction, signal_quality)
  • HybridModel outputs   (event_prob, direction_probs, confidence)
  • AdaptiveTargetHeads outputs (max_R_pred, target_bucket, regime_risk)

into a single ACTIONABLE trading decision per bar.

This layer is deliberately rule-based + transparent — no ML training here.
Every if/elif/else is auditable. That makes the policy reviewable by a
trader and tunable without retraining models.

Returns `TradeDecision`:
  • take_trade           bool
  • direction            +1 / -1 / 0
  • position_size_scale  0.0 ... 1.0 (multiplier on base contract count)
  • tp_mult              adaptive TP multiplier (≥ rules' default 1.5)
  • sl_mult              adaptive SL multiplier (typically 1.0, can be tighter)
  • reason               string explaining the decision

The 4 success criteria from the proposal:
  (1) Raise TP2 dynamically per trade  → tp_mult from max_R_pred
  (2) Improve hit rate                 → confidence-gated entry
  (3) Accurate 1R-2R prediction        → bucket-based selection
  (4) Reduce drawdown in volatile      → regime_risk-based sizing
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeDecision:
    take_trade: bool
    direction: int                  # +1 LONG, -1 SHORT, 0 SKIP
    position_size_scale: float      # 0.0 ... 1.0
    tp_mult: float                  # ATR multiplier for TP (>= base 1.5)
    sl_mult: float                  # ATR multiplier for SL
    reason: str

    def to_dict(self) -> dict:
        return {
            'take_trade': self.take_trade,
            'direction': self.direction,
            'position_size_scale': self.position_size_scale,
            'tp_mult': self.tp_mult,
            'sl_mult': self.sl_mult,
            'reason': self.reason,
        }


@dataclass
class TradeDecisionPolicy:
    """Thresholds & multipliers for the trade-decision rules.

    All values are tunable via direct attribute assignment — no
    retraining needed.
    """
    # Confidence gates
    hybrid_min_confidence: float = 0.55
    hybrid_min_event_prob: float = 0.5
    direction_min_prob: float = 0.50

    # Regime-risk caps on position size
    regime_normal_scale: float = 1.0
    regime_elevated_scale: float = 0.5
    regime_extreme_scale: float = 0.0      # skip extreme by default

    # Bucket-based size boosts (more confident → larger size)
    bucket_size_scale: tuple[float, ...] = (
        0.5,   # <1R bucket — half size or skip
        1.0,   # 1-2R — base size
        1.2,   # 2-3R — 20% more
        1.5,   # 3+R — 50% more
    )

    # Bucket-based TP multiplier overrides (criterion 1: raise TP2)
    # If bucket >= 2, allow TP to extend up to bucket_tp_mult[bucket]
    bucket_tp_mult: tuple[float, ...] = (
        1.5,   # bucket 0 (<1R): no extension, stick with base
        1.5,   # bucket 1 (1-2R): no extension
        2.0,   # bucket 2 (2-3R): extend TP to 2.0 × ATR
        2.5,   # bucket 3 (3+R): extend TP to 2.5 × ATR
    )

    # Conservative SL — never widen, can tighten in low-vol bucket
    base_sl_mult: float = 1.0
    tight_sl_mult: float = 0.7              # used in extreme regime

    # Quality-tier multipliers (uses day_trade signal_quality)
    # signal_quality=2 (premium) → +20% size
    # signal_quality=1 (regular) → base
    # signal_quality=0 (weak)    → skip
    quality_size_scale: tuple[float, ...] = (0.0, 1.0, 1.2)

    # Disagreement penalty — if rules say LONG and hybrid says SHORT, skip
    require_rule_hybrid_agree: bool = True


def decide(
    *,
    # ── day_trade rule outputs (per bar) ──
    rule_event_flag: int,
    rule_event_direction: int,           # +1 / -1 / 0
    rule_signal_quality: int,            # 0 / 1 / 2

    # ── HybridModel outputs ──
    hybrid_event_prob: float,            # [0, 1]
    hybrid_p_long: float,                 # [0, 1]
    hybrid_p_short: float,                # [0, 1]
    hybrid_p_neutral: float,              # [0, 1]
    hybrid_confidence: float,            # [0, 1]

    # ── AdaptiveTargetHeads outputs ──
    adaptive_max_r_pred: float,           # ≥ 0
    adaptive_bucket: int,                 # 0/1/2/3
    adaptive_regime_risk: int,            # 0/1/2

    # ── Policy ──
    policy: TradeDecisionPolicy | None = None,
) -> TradeDecision:
    """Apply the decision tree. Returns a TradeDecision."""
    P = policy or TradeDecisionPolicy()

    # ── Gate 0: day_trade rule must flag an event ──
    if int(rule_event_flag) != 1:
        return TradeDecision(
            take_trade=False, direction=0, position_size_scale=0.0,
            tp_mult=P.bucket_tp_mult[0], sl_mult=P.base_sl_mult,
            reason="no_rule_event",
        )

    # ── Gate 1: regime extreme → skip outright (criterion 4) ──
    if int(adaptive_regime_risk) == 2:
        return TradeDecision(
            take_trade=False, direction=0, position_size_scale=0.0,
            tp_mult=P.bucket_tp_mult[0], sl_mult=P.tight_sl_mult,
            reason="regime_extreme",
        )

    # ── Gate 2: signal_quality=0 (weak) → skip ──
    if int(rule_signal_quality) == 0:
        return TradeDecision(
            take_trade=False, direction=0, position_size_scale=0.0,
            tp_mult=P.bucket_tp_mult[0], sl_mult=P.base_sl_mult,
            reason="weak_signal_quality",
        )

    # ── Gate 3: hybrid confidence floor ──
    if float(hybrid_confidence) < P.hybrid_min_confidence:
        return TradeDecision(
            take_trade=False, direction=0, position_size_scale=0.0,
            tp_mult=P.bucket_tp_mult[0], sl_mult=P.base_sl_mult,
            reason="low_hybrid_confidence",
        )

    # ── Gate 4: event_prob floor ──
    if float(hybrid_event_prob) < P.hybrid_min_event_prob:
        return TradeDecision(
            take_trade=False, direction=0, position_size_scale=0.0,
            tp_mult=P.bucket_tp_mult[0], sl_mult=P.base_sl_mult,
            reason="low_hybrid_event_prob",
        )

    # ── Direction agreement (criterion 2: hit rate) ──
    rule_dir = int(rule_event_direction)
    hybrid_dir = (
        +1 if hybrid_p_long > max(hybrid_p_short, hybrid_p_neutral)
        else -1 if hybrid_p_short > max(hybrid_p_long, hybrid_p_neutral)
        else 0
    )
    if P.require_rule_hybrid_agree and rule_dir != hybrid_dir:
        return TradeDecision(
            take_trade=False, direction=0, position_size_scale=0.0,
            tp_mult=P.bucket_tp_mult[0], sl_mult=P.base_sl_mult,
            reason="rule_hybrid_disagree",
        )

    # ── Compute position size scale (criterion 4 + bucket-boost) ──
    bucket = int(adaptive_bucket)
    bucket = max(0, min(bucket, len(P.bucket_size_scale) - 1))
    base_size = P.bucket_size_scale[bucket]
    # Apply regime cap
    regime = int(adaptive_regime_risk)
    if regime == 1:
        regime_factor = P.regime_elevated_scale
    elif regime == 2:
        regime_factor = P.regime_extreme_scale   # already returned above
    else:
        regime_factor = P.regime_normal_scale
    # Apply signal_quality boost
    sq = int(rule_signal_quality)
    sq = max(0, min(sq, len(P.quality_size_scale) - 1))
    quality_factor = P.quality_size_scale[sq]
    # Apply hybrid confidence scaling
    conf_factor = float(hybrid_confidence)

    size = base_size * regime_factor * quality_factor * conf_factor
    size = max(0.0, min(size, 1.5))   # cap at 1.5× base contracts

    # ── Adaptive TP (criterion 1: raise TP2) ──
    # Use bucket-based TP override; clamp by max_r_pred to avoid greed
    bucket_tp = P.bucket_tp_mult[bucket]
    # If max_r_pred is high, allow extra extension up to bucket_tp
    pred_tp = min(float(adaptive_max_r_pred), bucket_tp)
    tp_mult = max(P.bucket_tp_mult[0], pred_tp)

    # SL stays at base unless regime elevated → slightly tighter
    sl_mult = P.tight_sl_mult if regime == 1 else P.base_sl_mult

    if size <= 0:
        return TradeDecision(
            take_trade=False, direction=0, position_size_scale=0.0,
            tp_mult=tp_mult, sl_mult=sl_mult, reason="zero_size_post_filter",
        )

    return TradeDecision(
        take_trade=True,
        direction=rule_dir,
        position_size_scale=size,
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        reason=(
            f"ok|bucket={bucket}|regime={regime}|sq={sq}"
            f"|conf={hybrid_confidence:.2f}|max_r={adaptive_max_r_pred:.2f}"
        ),
    )
