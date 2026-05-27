"""
modules/trading_intel/anti_collapse/sharpe_loss.py
═══════════════════════════════════════════════════════════════════════════════
Differentiable Sharpe / Sortino Loss — direct utility optimization.

Addresses Mechanism #1 (Loss Landscape Mismatch): standard SSL losses
optimize SURROGATE objectives (next-price MSE, next-regime CE) that are
loosely correlated with the actual trading goal (risk-adjusted return).
The model can minimize the surrogate perfectly while still losing money.

Solution (Western Decision-Focused Learning):
  Skip the surrogate entirely. Define a DIFFERENTIABLE trading objective
  directly on the model's outputs and the realized forward returns. The
  optimizer then aligns gradients with the actual P&L manifold instead of
  with a proxy.

Architecture of the loss:

  Inputs from the model:
    p_long   ∈ [0, 1]      probability of taking a long position
    p_short  ∈ [0, 1]      probability of taking a short
    p_flat   ∈ [0, 1]      probability of doing nothing (1 - p_long - p_short)
    magnitude ≥ 0           size of intended position (in R-multiples)

  Inputs from the data:
    target_ret  ∈ R         realized log return over the trade horizon

  Predicted position (differentiable, soft):
    position = (p_long − p_short) · magnitude     ∈ [-magnitude, +magnitude]

  P&L per sample (with proportional transaction cost):
    pnl = position · target_ret − cost · |position|

  Risk-adjusted objective (default: Sharpe ratio):
    Sharpe = E[pnl] / std(pnl)
    Loss   = −Sharpe       (maximization → minimization)

  Sortino (alternative, penalizes only downside):
    DownsideStd = std(min(pnl, 0))
    Sortino = E[pnl] / DownsideStd
    Loss = −Sortino

Why this is "different" from MSE/CE:
  • The gradient w.r.t. the model now points TOWARD higher Sharpe directly
  • A model that minimizes MSE on returns may still bet poorly if its
    miscalibrated probabilities ignore the asymmetry of P&L distributions
  • Sharpe loss correctly penalizes confident-but-wrong predictions more
    than uncertain ones, because variance contribution dominates

Numerical stability:
  • Std is computed with a small ε to avoid division by zero
  • Returns are clipped to ±max_clip to limit the contribution of outliers
  • Variance is computed across the batch — large batches give more stable
    gradients (recommended batch size: ≥ 256)

Practical notes:
  • Pair with a small CE/MSE regularizer to keep predictions well-formed
  • Use on the holdout-validated label slice only (not on unlabeled data)
  • Cost should reflect REAL transaction costs in the data's return units

References:
  • Moody & Saffell (2001) "Learning to Trade via Direct Reinforcement"
  • Donti, Amos & Kolter (2017) "Task-based End-to-end Model Learning"
  • Wilder, Dilkina & Tambe (2019) "Melding the Data-Decisions Pipeline"
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SharpeLossConfig:
    """Hyperparameters for the differentiable Sharpe / Sortino loss."""
    cost_per_unit_position: float = 1e-4    # 1 bp on a unit-sized position
    eps: float = 1e-6                       # numerical floor on std
    return_clip: float | None = 0.10        # clip returns to ±10% (1000 pips)
    annualization_factor: float | None = None  # set to e.g. sqrt(252*24*4) to
                                            # express the loss in annualized
                                            # Sharpe units. Leave None to
                                            # match the natural batch scale.
    objective: str = "sharpe"               # 'sharpe' or 'sortino'
    min_active_fraction: float = 0.05       # if fewer than this fraction of
                                            # samples have a non-zero position,
                                            # return a regularizer-only loss
                                            # to avoid degenerate gradients


def _safe_std(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Standard deviation with a small ε added inside the sqrt for
    numerical stability on near-constant tensors."""
    if x.numel() <= 1:
        return torch.tensor(eps, device=x.device, dtype=x.dtype)
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return torch.sqrt(var + eps)


def differentiable_sharpe_loss(
    p_long: torch.Tensor,      # (B,) in [0, 1]
    p_short: torch.Tensor,     # (B,) in [0, 1]
    magnitude: torch.Tensor,   # (B,) ≥ 0
    target_ret: torch.Tensor,  # (B,) realized forward return
    *,
    config: SharpeLossConfig | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute −Sharpe (or −Sortino) for a batch of (prediction, outcome) pairs.

    Returns (loss, diagnostics_dict).

    The diagnostics dict reports the empirical Sharpe (positive number),
    mean PnL, std PnL, fraction of active trades, etc. — these are NOT
    differentiable, just for logging.
    """
    cfg = config or SharpeLossConfig()

    # ── Validate shapes ───────────────────────────────────────────────
    B = p_long.shape[0]
    for name, t in (("p_short", p_short), ("magnitude", magnitude),
                    ("target_ret", target_ret)):
        if t.shape != (B,):
            raise ValueError(
                f"shape mismatch: {name} has {tuple(t.shape)}, expected ({B},)"
            )

    # ── Predicted position (signed, soft) ─────────────────────────────
    # position ∈ [-magnitude, +magnitude]; positive = long, negative = short
    position = (p_long - p_short) * magnitude

    # ── Realized return (clipped to dampen outliers) ──────────────────
    ret = target_ret
    if cfg.return_clip is not None:
        ret = torch.clamp(ret, min=-cfg.return_clip, max=cfg.return_clip)

    # ── P&L net of transaction cost ───────────────────────────────────
    pnl = position * ret - cfg.cost_per_unit_position * position.abs()

    # ── Empirical diagnostics (non-differentiable, for logging) ───────
    with torch.no_grad():
        mean_pnl = float(pnl.mean())
        std_pnl = float(_safe_std(pnl, cfg.eps))
        emp_sharpe = mean_pnl / std_pnl if std_pnl > cfg.eps else 0.0
        active_frac = float((position.abs() > cfg.eps).float().mean())
        if cfg.annualization_factor is not None:
            emp_sharpe = emp_sharpe * cfg.annualization_factor

    diagnostics = {
        "emp_sharpe": emp_sharpe,
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "active_fraction": active_frac,
        "n_samples": B,
    }

    # ── Degenerate-case guard ─────────────────────────────────────────
    if active_frac < cfg.min_active_fraction:
        # Too few active trades — return a small regularizer that
        # encourages the model to take SOME positions
        reg_loss = -position.abs().mean()   # gently push magnitude up
        diagnostics["loss_kind"] = "degenerate_regularizer"
        return reg_loss, diagnostics

    # ── Sharpe / Sortino ──────────────────────────────────────────────
    if cfg.objective == "sortino":
        # Downside std: penalize only negative P&L variance
        downside = torch.clamp(pnl, max=0.0)
        denom = _safe_std(downside, cfg.eps)
    elif cfg.objective == "sharpe":
        denom = _safe_std(pnl, cfg.eps)
    else:
        raise ValueError(f"Unknown objective: {cfg.objective}")

    # Differentiable ratio. Note: torch.std is differentiable everywhere
    # except at the exact mean; _safe_std handles that with the ε floor.
    sharpe = pnl.mean() / denom

    # Annualization (just a constant multiplier — gradient direction
    # is preserved either way)
    if cfg.annualization_factor is not None:
        sharpe = sharpe * cfg.annualization_factor

    loss = -sharpe
    diagnostics["loss_kind"] = cfg.objective
    return loss, diagnostics


class SharpeRegularizer:
    """Wrapper that adds a Sharpe-loss term to a primary loss with a
    configurable weight. Use this when you want CE/BCE to keep the
    predictions well-calibrated AND Sharpe to align them with the
    decision objective.

    Usage:
        sharpe_reg = SharpeRegularizer(weight=0.3)

        # In training step:
        primary = compute_hybrid_loss(...)  # standard CE/BCE
        sharpe_term, diag = sharpe_reg(p_long, p_short, magnitude, target_ret)
        total = primary + sharpe_term
        total.backward()
    """

    def __init__(
        self, weight: float = 0.5, config: SharpeLossConfig | None = None,
    ):
        self.weight = weight
        self.config = config or SharpeLossConfig()

    def __call__(
        self,
        p_long: torch.Tensor, p_short: torch.Tensor,
        magnitude: torch.Tensor, target_ret: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        loss, diag = differentiable_sharpe_loss(
            p_long, p_short, magnitude, target_ret, config=self.config,
        )
        return self.weight * loss, diag
