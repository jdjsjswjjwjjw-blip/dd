"""
ssl/pretrain_cycle.py
═══════════════════════════════════════════════════════════
SSL Pretraining للـ PriceCycleModel (PR #18).

يدرّب 4 SSL heads (rules-based weak supervision):
    phase (Wyckoff 4 classes)
    maturity (3 classes)
    swing (3 classes)
    cycle_position (regression [0,1])

الـ direction head في PriceCycleModel غير موجود (التصميم يفترض ده).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.price_cycle.config import PriceCycleConfig
from modules.price_cycle.cycle_model import PriceCycleModel, CycleTargets

from self_supervised.data_loader import build_ssl_loaders


MAX_SKIP_RATIO = 0.05  # >5% = silent corruption, fail hard


def _build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """No weight decay on biases / LayerNorm / Embeddings (C7)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_no_decay = (
            name.endswith('.bias')
            or '.ln' in name or 'layer_norm' in name.lower()
            or 'layernorm' in name.lower()
            or 'norm.' in name or 'bn.' in name
            or 'embed' in name.lower()
        )
        (no_decay if is_no_decay else decay).append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def make_cycle_targets(batch: dict, device: torch.device) -> CycleTargets:
    """Build CycleTargets from SSL batch (no direction)."""
    return CycleTargets(
        phase=batch['phase_target'].to(device),
        maturity=batch['maturity_target'].to(device),
        swing_direction=batch['swing_target'].to(device),
        cycle_position=batch['cycle_position_target'].to(device),
        reversal_proximity=None,
    )


def _masked_mean_loss(loss_per_sample: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    w = valid.to(loss_per_sample.dtype)
    denom = w.sum().clamp(min=1.0)
    return (loss_per_sample * w).sum() / denom


def _params_are_finite(model) -> bool:
    return all(torch.isfinite(p.data).all() for p in model.parameters())


def train_cycle_epoch(model, loader, optimizer, device, scaler, scheduler,
                      use_amp: bool) -> dict:
    model.train()
    losses_sum: dict[str, float] = {}
    n_batches = 0
    n_skipped = 0
    n_total = 0
    for batch in loader:
        n_total += 1
        cycle_input = batch['cycle_window'].to(device, non_blocking=True)
        targets = make_cycle_targets(batch, device)
        cycle_valid = batch['cycle_valid'].to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(cycle_input)
                losses = model.heads.compute_loss(outputs, targets, reduction='none')
        else:
            outputs = model(cycle_input)
            losses = model.heads.compute_loss(outputs, targets, reduction='none')

        # Validity-aware mean over per-sample losses
        total = torch.zeros((), device=device)
        per_task_log: dict[str, float] = {}
        for k, v in losses.items():
            if k == 'total' or k.startswith('weighted_'):
                continue
            if v.dim() == 0:                # already reduced — fall back
                total = total + v
                per_task_log[k] = float(v.item())
                continue
            masked = _masked_mean_loss(v, cycle_valid)
            total = total + masked
            per_task_log[k] = float(masked.item())

        if not torch.isfinite(total):
            n_skipped += 1
            continue

        if use_amp:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_ok = all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if grad_ok:
                scaler.step(optimizer)
            else:
                n_skipped += 1
            scaler.update()
        else:
            total.backward()
            grad_ok = all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if not grad_ok:
                n_skipped += 1
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        per_task_log['total'] = float(total.item())
        for k, v in per_task_log.items():
            losses_sum[k] = losses_sum.get(k, 0.0) + v
        n_batches += 1

    if n_total > 0 and n_skipped / n_total > MAX_SKIP_RATIO:
        raise RuntimeError(
            f"Too many skipped batches: {n_skipped}/{n_total} "
            f"({100*n_skipped/n_total:.1f}%). Aborting."
        )
    if n_skipped > 0:
        print(f"           ⚠️  train: skipped {n_skipped}/{n_total}")
    if not _params_are_finite(model):
        raise RuntimeError("Model params became NaN/Inf — aborting (no silent reset).")
    return {k: v / max(n_batches, 1) for k, v in losses_sum.items()}


@torch.no_grad()
def eval_cycle_epoch(model, loader, device, use_amp: bool) -> dict:
    model.eval()
    losses_sum: dict[str, float] = {}
    n_batches = 0
    n_skipped = 0
    n_total = 0
    for batch in loader:
        n_total += 1
        cycle_input = batch['cycle_window'].to(device, non_blocking=True)
        targets = make_cycle_targets(batch, device)
        cycle_valid = batch['cycle_valid'].to(device)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(cycle_input)
                losses = model.heads.compute_loss(outputs, targets, reduction='none')
        else:
            outputs = model(cycle_input)
            losses = model.heads.compute_loss(outputs, targets, reduction='none')

        total = torch.zeros((), device=device)
        per_task_log: dict[str, float] = {}
        for k, v in losses.items():
            if k == 'total' or k.startswith('weighted_'):
                continue
            if v.dim() == 0:
                total = total + v
                per_task_log[k] = float(v.item())
                continue
            masked = _masked_mean_loss(v, cycle_valid)
            total = total + masked
            per_task_log[k] = float(masked.item())

        if not torch.isfinite(total):
            n_skipped += 1
            continue
        per_task_log['total'] = float(total.item())
        for k, v in per_task_log.items():
            losses_sum[k] = losses_sum.get(k, 0.0) + v
        n_batches += 1
    if n_total > 0 and n_skipped / n_total > MAX_SKIP_RATIO:
        raise RuntimeError(
            f"eval: too many NaN batches ({n_skipped}/{n_total}) — aborting"
        )
    if n_skipped > 0:
        print(f"           ⚠️  eval: skipped {n_skipped}/{n_total}")
    return {k: v / max(n_batches, 1) for k, v in losses_sum.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--lob-tensors', required=True)
    p.add_argument('--lob-timestamps', default=None)
    p.add_argument('--order-batches-dir', default=None)
    p.add_argument('--output', default='checkpoints/ssl_cycle')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--train-split', type=float, default=0.75)
    p.add_argument('--lookback-bars', type=int, default=50)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--early-stopping-patience', type=int, default=8)
    p.add_argument('--embargo-bars', type=int, default=24)
    p.add_argument('--use-amp', action='store_true')
    args = p.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    use_amp = bool(args.use_amp) and device.type == 'cuda'
    print(f"═══ SSL Pretraining: PriceCycleModel ═══")
    print(f"Device: {device} (AMP={use_amp})\nOutput: {args.output}")

    train_loader, holdout_loader = build_ssl_loaders(
        args.features, args.lob_tensors, args.lob_timestamps,
        order_batches_dir=args.order_batches_dir,
        batch_size=args.batch_size, train_split=args.train_split,
        num_workers=args.num_workers, lookback_bars=args.lookback_bars,
        embargo_bars=args.embargo_bars,
    )

    print("🏗️  Building PriceCycleModel...")
    sample_batch = next(iter(train_loader))
    cycle_dim = sample_batch['cycle_window'].shape[-1]
    config = PriceCycleConfig()
    if hasattr(config, 'encoder') and hasattr(config.encoder, 'n_input_features'):
        config.encoder.n_input_features = cycle_dim
    model = PriceCycleModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,} | Input dim: {cycle_dim}")
    print()

    optimizer = torch.optim.AdamW(
        _build_param_groups(model, args.weight_decay),
        lr=args.lr,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    patience = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_metrics = train_cycle_epoch(
            model, train_loader, optimizer, device, scaler, scheduler, use_amp,
        )
        val_metrics = eval_cycle_epoch(model, holdout_loader, device, use_amp)
        elapsed = time.time() - t0

        train_total = train_metrics.get('total', 0.0)
        val_total = val_metrics.get('total', 0.0)
        history.append({
            'epoch': epoch + 1,
            'train': train_metrics, 'val': val_metrics, 'time_sec': elapsed,
        })

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train={train_total:.4f} | val={val_total:.4f} | {elapsed:.1f}s")

        if val_total < best_val_loss:
            best_val_loss = val_total
            patience = 0
            try:
                cfg_dict = asdict(config)
            except TypeError:
                cfg_dict = None
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg_dict if cfg_dict is not None else config,
                'epoch': epoch + 1, 'val_loss': val_total, 'val_metrics': val_metrics,
                'args': vars(args),
                'input_dim': cycle_dim,
            }, Path(args.output) / 'best_ssl_cycle.pt')
            print(f"           💾 saved best")
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                print(f"⏹️  Early stopping (patience={patience})")
                break

    with open(Path(args.output) / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Cycle SSL Pretraining complete | Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
