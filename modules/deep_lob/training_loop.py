"""
modules/deep_lob/training_loop.py
──────────────────────────────────
Production-grade training loop for HierarchicalLOBTransformer.

Features:
    - Mixed precision (fp16/bf16)
    - Gradient accumulation
    - Gradient clipping
    - Cosine warmup schedule
    - Early stopping
    - Top-K checkpoint keeping
    - Reproducible (seed control)
    - Comprehensive metrics logging
    - Validation interleaving

Not auto-runs — designed for orchestration by user (CLI / notebook / pipeline).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import TrainingConfig
from .hierarchical_model import HierarchicalLOBTransformer
from .multi_task_heads import MultiTaskTargets


@dataclass
class TrainingMetrics:
    """Per-step metrics."""

    step: int
    epoch: int
    train_loss: float
    train_losses_per_task: dict[str, float] = field(default_factory=dict)
    val_loss: Optional[float] = None
    val_losses_per_task: Optional[dict[str, float]] = None
    learning_rate: float = 0.0
    grad_norm: Optional[float] = None
    seconds_per_step: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "train_loss": float(self.train_loss),
            "train_losses": {k: float(v) for k, v in self.train_losses_per_task.items()},
            "val_loss": float(self.val_loss) if self.val_loss is not None else None,
            "val_losses": (
                {k: float(v) for k, v in self.val_losses_per_task.items()}
                if self.val_losses_per_task else None
            ),
            "learning_rate": float(self.learning_rate),
            "grad_norm": float(self.grad_norm) if self.grad_norm is not None else None,
            "seconds_per_step": float(self.seconds_per_step),
        }


def _set_seed(seed: int, deterministic: bool = True):
    """Reproducibility setup."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _cosine_warmup_schedule(
    step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr_ratio: float = 0.1,
) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(1.0, max(0.0, progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def _build_optimizer(
    model: nn.Module, config: TrainingConfig,
) -> torch.optim.Optimizer:
    """Build optimizer with weight decay applied selectively."""
    # No weight decay on biases / norms
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name or "norm" in name.lower() or "embed" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            param_groups, lr=config.learning_rate, betas=config.betas, eps=config.eps,
        )
    if config.optimizer == "adam":
        return torch.optim.Adam(
            param_groups, lr=config.learning_rate, betas=config.betas, eps=config.eps,
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(param_groups, lr=config.learning_rate, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


# ════════════════════════════════════════════════════════════════════════════
# Trainer
# ════════════════════════════════════════════════════════════════════════════


class HierarchicalLOBTrainer:
    """Production training driver."""

    def __init__(
        self,
        model: HierarchicalLOBTransformer,
        config: TrainingConfig,
        output_dir: str,
        device: str = "auto",
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        # Reproducibility
        _set_seed(config.seed, config.deterministic)

        # Optimizer
        self.optimizer = _build_optimizer(model, config)

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if (
            config.use_amp and self.device.type == "cuda"
        ) else None

        # Metrics tracking
        self.metrics_history: list[TrainingMetrics] = []
        self.best_val_loss = float("inf")
        self.best_step = 0
        self.patience_counter = 0

        # Checkpoint tracking
        self.checkpoint_scores: list[tuple[int, float, str]] = []  # (step, score, path)

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        targets: MultiTaskTargets,
        accumulation: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Single forward/backward step."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        targets = MultiTaskTargets(**{
            k: (v.to(self.device) if v is not None else None)
            for k, v in targets.__dict__.items()
        })

        # Forward + loss (with optional AMP)
        if self.scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = self.model(**batch)
                losses = self.model.heads.compute_loss(outputs, targets)
            self.scaler.scale(losses["total"]).backward()
        else:
            outputs = self.model(**batch)
            losses = self.model.heads.compute_loss(outputs, targets)
            losses["total"].backward()

        return losses

    def _optimizer_step(self) -> Optional[float]:
        """Optimizer step + scheduler + grad clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        grad_norm = None
        if self.config.max_grad_norm > 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm,
                )
            )

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    def _update_lr(self, step: int):
        """Apply learning rate schedule."""
        if self.config.scheduler == "constant":
            return self.config.learning_rate
        if self.config.scheduler == "cosine_warmup":
            lr = _cosine_warmup_schedule(
                step, self.config.warmup_steps, self.config.max_steps,
                self.config.learning_rate,
            )
        else:
            lr = self.config.learning_rate

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def train_step_batch(
        self,
        batch: dict[str, torch.Tensor],
        targets: MultiTaskTargets,
        step: int,
        epoch: int = 0,
    ) -> TrainingMetrics:
        """Train on a single batch.

        Useful when caller manages iteration (DataLoader external).

        Returns
        -------
        TrainingMetrics
        """
        self.model.train()
        t0 = time.perf_counter()

        # Update LR
        lr = self._update_lr(step)

        # Forward + backward
        losses = self._step(batch, targets, accumulation=False)

        # Optimize
        grad_norm = self._optimizer_step()

        elapsed = time.perf_counter() - t0
        metric = TrainingMetrics(
            step=step,
            epoch=epoch,
            train_loss=float(losses["total"].detach().cpu()),
            train_losses_per_task={
                k: float(v.detach().cpu()) for k, v in losses.items() if k != "total"
                and not k.startswith("weighted_")
            },
            learning_rate=lr,
            grad_norm=grad_norm,
            seconds_per_step=elapsed,
        )
        self.metrics_history.append(metric)
        return metric

    @torch.no_grad()
    def validate(
        self,
        val_batches: list[tuple[dict, MultiTaskTargets]],
    ) -> dict[str, float]:
        """Run validation on a list of (batch, targets) tuples."""
        self.model.eval()
        total_loss = 0.0
        per_task_losses: dict[str, list] = {}
        n_batches = 0

        for batch, targets in val_batches:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            targets_d = MultiTaskTargets(**{
                k: (v.to(self.device) if v is not None else None)
                for k, v in targets.__dict__.items()
            })

            outputs = self.model(**batch)
            losses = self.model.heads.compute_loss(outputs, targets_d)

            total_loss += float(losses["total"])
            for k, v in losses.items():
                if k == "total" or k.startswith("weighted_"):
                    continue
                per_task_losses.setdefault(k, []).append(float(v))
            n_batches += 1

        return {
            "val_loss": total_loss / max(n_batches, 1),
            **{f"val_{k}": float(np.mean(v)) for k, v in per_task_losses.items()},
        }

    def save_checkpoint(self, step: int, score: float, tag: str = ""):
        """Save model + trainer state."""
        ckpt_name = f"ckpt_step{step:08d}{('_' + tag) if tag else ''}.pt"
        path = self.output_dir / ckpt_name
        self.model.save_checkpoint(
            str(path),
            extra_state={
                "step": step,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                "best_val_loss": self.best_val_loss,
                "best_step": self.best_step,
            },
        )

        # Track top-K
        self.checkpoint_scores.append((step, score, str(path)))
        self.checkpoint_scores.sort(key=lambda x: x[1])  # low score first
        if len(self.checkpoint_scores) > self.config.keep_top_k_checkpoints:
            # Remove worst
            _, _, worst_path = self.checkpoint_scores.pop()
            try:
                os.remove(worst_path)
            except OSError:
                pass

    def save_metrics(self):
        """Save metrics history to JSON."""
        path = self.output_dir / "metrics.jsonl"
        with open(path, "w") as f:
            for m in self.metrics_history:
                f.write(json.dumps(m.to_dict()) + "\n")

    def early_stopping_check(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_step = self.metrics_history[-1].step if self.metrics_history else 0
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience
