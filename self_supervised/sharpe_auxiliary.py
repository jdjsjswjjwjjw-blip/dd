"""
self_supervised/sharpe_auxiliary.py
═══════════════════════════════════════════════════════════════════════════════
Sharpe-aware auxiliary loss for the SSL pretraining loop.

Motivation
──────────
The existing six SSL tasks (next_price, next_imbalance, next_volatility,
next_regime, wall_persist, time_to_event) are all MSE / cross-entropy
losses on heuristic targets. The model is rewarded for accurate
forecasts in MSE units — which is NOT the same as rewarding forecasts
that translate into profitable trades.

The Sharpe-aware auxiliary task closes this gap. It takes the model's
predicted next-period return, converts it into a soft position via
tanh(temperature * pred), and computes a differentiable Sharpe ratio
against the realised return. Minimising −Sharpe encourages the encoder
to learn representations that are both predictive AND tradable.

Architecture decisions
──────────────────────
1. Pure utility — does NOT modify the SSL model architecture. Called
   from `compute_ssl_loss` only when `cfg.enabled and cfg.weight > 0`.

2. Off by default — `SharpeAuxConfig.enabled=False` keeps the existing
   six-task setup byte-identical.

3. Direction-aware via tanh → reuses the existing
   `differentiable_sharpe_loss` from trading_intel by converting a
   continuous prediction into the (p_long, p_short, magnitude) trinity
   the existing loss expects.

4. Temperature is the single knob that controls how aggressively the
   model commits to a direction. Low temperature → soft / neutral
   positions (the loss looks like an MSE in the limit). High
   temperature → hard long/short positions (the loss is closer to a
   classification Sharpe).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.trading_intel.anti_collapse.sharpe_loss import (
    SharpeLossConfig,
    differentiable_sharpe_loss,
)


@dataclass
class SharpeAuxConfig:
    """Configuration for the Sharpe-aware auxiliary loss in pretrain_lob.

    Field semantics
    ───────────────
    enabled                On/off switch. False → no-op, preserves the
                           existing six-task SSL loss exactly.
    weight                 Multiplier on the auxiliary loss in the
                           total. Should be in [0.05, 0.5] for most
                           setups — smaller than the forecasting heads
                           so the Sharpe doesn't dominate early when
                           predictions are still random.
    temperature            tanh temperature for converting pred → soft
                           position. 1.0 keeps pred-as-position;
                           values > 1 amplify; < 1 dampen.
    cost_per_unit_position Pass-through to differentiable_sharpe_loss.
                           Matches the typical 0.5–1 pip per side cost.
    return_clip            Pass-through. Clips realised return to
                           ±return_clip before the Sharpe calc to dampen
                           outliers.
    objective              "sharpe" or "sortino". Sortino penalises only
                           downside variance — closer to what a trader
                           cares about.
    """

    enabled: bool = False
    weight: float = 0.10
    temperature: float = 1.0
    cost_per_unit_position: float = 1e-4
    return_clip: float | None = 0.10
    objective: str = "sharpe"

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError(f"weight must be ≥ 0, got {self.weight}")
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0, got {self.temperature}"
            )
        if self.cost_per_unit_position < 0.0:
            raise ValueError(
                f"cost_per_unit_position must be ≥ 0, got "
                f"{self.cost_per_unit_position}"
            )
        if self.objective not in ("sharpe", "sortino"):
            raise ValueError(
                f"objective must be 'sharpe' or 'sortino', "
                f"got {self.objective!r}"
            )


def _to_sharpe_trinity(
    pred_return: torch.Tensor, temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a continuous prediction into (p_long, p_short, magnitude).

    tanh squashes to [-1, +1] — positive becomes p_long, negative
    becomes p_short, |value| becomes magnitude. For any single sample
    exactly one of p_long / p_short is nonzero — this matches a
    "either long OR short" interpretation rather than a "both at once"
    one. The existing differentiable_sharpe_loss handles either.

    Pure: no tensors are mutated.
    """
    if pred_return.dim() != 1:
        raise ValueError(
            f"pred_return must be 1-D (B,), got shape "
            f"{tuple(pred_return.shape)}"
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    signed = torch.tanh(pred_return * float(temperature))
    p_long = torch.clamp(signed, min=0.0)
    p_short = torch.clamp(-signed, min=0.0)
    magnitude = signed.abs()
    return p_long, p_short, magnitude


def compute_sharpe_aux_loss(
    pred_return: torch.Tensor,
    target_return: torch.Tensor,
    config: SharpeAuxConfig,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute the Sharpe-aware auxiliary loss.

    Parameters
    ──────────
    pred_return   (B,)  model output — predicted next-period return.
                        Can be on any scale; temperature handles the
                        rescaling into the [-1, +1] position range.
    target_return (B,)  realised next-period return (same row index).
    config              SharpeAuxConfig instance.
    valid_mask   (B,) bool optional — drops samples where the target
                        is missing/truncated. When None, all samples
                        are used.

    Returns
    ───────
    (loss, diagnostics) — diagnostics dict has emp_sharpe, mean_pnl,
                          std_pnl, active_fraction, n_samples_used.

    When `config.enabled` is False, returns (tensor(0.0), {}) so the
    caller can unconditionally add the result to total_loss.
    """
    if not config.enabled:
        return pred_return.new_zeros(()), {"loss_kind": "disabled"}

    if pred_return.shape != target_return.shape:
        raise ValueError(
            f"shape mismatch: pred_return {tuple(pred_return.shape)} "
            f"vs target_return {tuple(target_return.shape)}"
        )

    if valid_mask is not None:
        if valid_mask.shape != pred_return.shape:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} must match "
                f"pred_return {tuple(pred_return.shape)}"
            )
        pred_return = pred_return[valid_mask]
        target_return = target_return[valid_mask]

    if pred_return.numel() < 2:
        return pred_return.new_zeros(()), {
            "loss_kind": "too_few_samples",
            "n_samples_used": int(pred_return.numel()),
        }

    p_long, p_short, magnitude = _to_sharpe_trinity(
        pred_return, config.temperature,
    )
    sharpe_cfg = SharpeLossConfig(
        cost_per_unit_position=config.cost_per_unit_position,
        return_clip=config.return_clip,
        objective=config.objective,
    )
    loss, diag = differentiable_sharpe_loss(
        p_long=p_long,
        p_short=p_short,
        magnitude=magnitude,
        target_ret=target_return,
        config=sharpe_cfg,
    )
    diag["n_samples_used"] = int(pred_return.numel())
    return loss, diag
