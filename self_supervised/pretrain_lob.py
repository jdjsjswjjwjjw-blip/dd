"""
ssl/pretrain_lob.py
═══════════════════════════════════════════════════════════
SSL pretraining for HierarchicalLOBTransformer (PR #17).

Six SSL heads (no direction):
    next_price, next_imbalance, next_volatility, next_regime,
    wall_persist, time_to_event

Quant-grade training loop:
  * Per-task validity masks (next_price_valid, wall_persist_valid, ...)
    are pulled from each batch and applied to per-task losses so
    truncated-target samples don't pollute gradients.
  * Real AMP (autocast + GradScaler.scale/step/update) when CUDA.
  * AdamW param groups: weight_decay = 0 on bias / LayerNorm / Embeddings.
  * Skipped-batch ratio reported and HARD-FAILS if > 5% (prevents the
    "best val_loss = epoch with the most NaN-skips" selection bias).
  * Checkpoint persists the architecture config (DeepLOBConfig) for safe
    reload by extract_embeddings.
  * No silent param-reset hack — NaN params now raise instead of zeroing
    silently destroying trained weights.
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

from modules.deep_lob.config import DeepLOBConfig
from modules.deep_lob.hierarchical_model import HierarchicalLOBTransformer
from modules.deep_lob.multi_task_heads import MultiTaskTargets

from self_supervised.data_loader import build_ssl_loaders


MAX_SKIP_RATIO = 0.05    # >5% skipped batches = bug, fail hard


# ════════════════════════════════════════════════════════════════════════════
# Param groups (C7): no weight decay on biases / LayerNorm / Embedding
# ════════════════════════════════════════════════════════════════════════════

def _build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # No WD on bias, LayerNorm, BatchNorm, Embedding
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


# ════════════════════════════════════════════════════════════════════════════
# Validity-aware multi-task loss
# ════════════════════════════════════════════════════════════════════════════

def _masked_loss(loss_per_sample: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Average loss over valid samples only.

    loss_per_sample : (B,) float
    valid_mask      : (B,) float in {0, 1}

    Returns scalar mean over valid samples; 0 if no valid samples.
    """
    w = valid_mask.to(loss_per_sample.dtype)
    denom = w.sum().clamp(min=1.0)
    return (loss_per_sample * w).sum() / denom


def compute_ssl_loss(
    outputs, batch: dict, device: torch.device, config_weights,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-task SSL loss with per-sample validity masking.

    Returns (total_loss, per_task_loss_floats).
    """
    # Per-task validity masks (from batch dict)
    v_price  = batch['next_price_valid'].to(device)
    v_imb    = batch['next_imbalance_valid'].to(device)
    v_vol    = batch['next_volatility_valid'].to(device)
    v_reg    = batch['next_regime_valid'].to(device)
    v_wall   = batch['wall_persist_valid'].to(device)
    v_tte    = batch['time_to_event_valid'].to(device)

    # Targets
    t_price = batch['next_price'].to(device)
    t_imb   = batch['next_imbalance'].to(device)
    t_vol   = batch['next_volatility'].to(device).clamp(min=1e-6)
    t_reg   = batch['next_regime'].to(device).long().clamp(0, 3)
    t_wall  = batch['wall_persist'].to(device)
    t_tte   = batch['time_to_event'].to(device)

    # Per-task per-sample losses (reduction='none')
    l_price = F.mse_loss(outputs.next_price, t_price, reduction='none')
    l_imb   = F.mse_loss(outputs.next_imbalance, t_imb, reduction='none')
    # Log-MSE on volatility (robust to outliers); clamp inputs to safe range
    log_pred = torch.log(outputs.next_volatility.clamp(min=1e-6))
    log_tgt  = torch.log(t_vol.clamp(min=1e-6))
    l_vol   = F.mse_loss(log_pred, log_tgt, reduction='none')
    l_reg   = F.cross_entropy(outputs.next_regime_logits, t_reg, reduction='none')
    l_wall  = F.smooth_l1_loss(outputs.wall_persist, t_wall, reduction='none')
    l_tte   = F.smooth_l1_loss(outputs.time_to_event, t_tte, reduction='none')

    # Mask + average
    losses = {
        'next_price':      _masked_loss(l_price, v_price),
        'next_imbalance':  _masked_loss(l_imb,   v_imb),
        'next_volatility': _masked_loss(l_vol,   v_vol),
        'next_regime':     _masked_loss(l_reg,   v_reg),
        'wall_persist':    _masked_loss(l_wall,  v_wall),
        'time_to_event':   _masked_loss(l_tte,   v_tte),
    }

    # Weighted sum
    w = config_weights
    weighted_sum = (
        w.next_price_weight      * losses['next_price']
        + w.next_imbalance_weight  * losses['next_imbalance']
        + w.next_volatility_weight * losses['next_volatility']
        + w.next_regime_weight     * losses['next_regime']
        + w.wall_persist_weight    * losses['wall_persist']
        + w.time_to_event_weight   * losses['time_to_event']
    )

    return weighted_sum, {
        'total': float(weighted_sum.item()),
        **{k: float(v.item()) for k, v in losses.items()},
    }


# ════════════════════════════════════════════════════════════════════════════
# Train / eval epochs
# ════════════════════════════════════════════════════════════════════════════

def _params_are_finite(model) -> bool:
    return all(torch.isfinite(p.data).all() for p in model.parameters())


def train_epoch(model, loader, optimizer, device, scaler, scheduler,
                config_weights, use_amp: bool) -> dict:
    model.train()
    losses_sum: dict[str, float] = {}
    n_batches = 0
    n_skipped = 0
    n_total = 0
    for batch in loader:
        n_total += 1
        inputs = {
            'order_features': batch['order_features'].to(device, non_blocking=True),
            'order_masks':    batch['order_masks'].to(device, non_blocking=True),
            'bar_mask':       batch['bar_mask'].to(device, non_blocking=True),
            'context':        batch['context'].to(device, non_blocking=True),
        }

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(**inputs)
                total_loss, per_task = compute_ssl_loss(outputs, batch, device, config_weights)
        else:
            outputs = model(**inputs)
            total_loss, per_task = compute_ssl_loss(outputs, batch, device, config_weights)

        if not torch.isfinite(total_loss):
            n_skipped += 1
            continue

        if use_amp:
            scaler.scale(total_loss).backward()
            # Unscale before grad-norm clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Check grad finite AFTER unscale (scaler handles inf on its own,
            # but we want to catch NaN explicitly)
            grad_ok = all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if grad_ok:
                scaler.step(optimizer)
            else:
                n_skipped += 1
                # Still must call update to track scaler stats
            scaler.update()
        else:
            total_loss.backward()
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

        for k, v in per_task.items():
            losses_sum[k] = losses_sum.get(k, 0.0) + v
        n_batches += 1

    # Fail-fast: too many skipped batches = silent corruption somewhere
    if n_total > 0 and n_skipped / n_total > MAX_SKIP_RATIO:
        raise RuntimeError(
            f"Too many batches skipped: {n_skipped}/{n_total} = "
            f"{100*n_skipped/n_total:.1f}% (limit {100*MAX_SKIP_RATIO}%). "
            f"NaN-corruption detected; aborting before mean-loss becomes biased."
        )

    if n_skipped > 0:
        print(f"           ⚠️  train: skipped {n_skipped}/{n_total} batches")
    if not _params_are_finite(model):
        raise RuntimeError(
            "Model parameters contain NaN/Inf after epoch — aborting. "
            "Previously this silently zeroed bad params which destroyed "
            "trained weights without warning."
        )
    return {k: v / max(n_batches, 1) for k, v in losses_sum.items()}


@torch.no_grad()
def eval_epoch(model, loader, device, config_weights, use_amp: bool) -> dict:
    model.eval()
    losses_sum: dict[str, float] = {}
    n_batches = 0
    n_skipped = 0
    n_total = 0
    for batch in loader:
        n_total += 1
        inputs = {
            'order_features': batch['order_features'].to(device, non_blocking=True),
            'order_masks':    batch['order_masks'].to(device, non_blocking=True),
            'bar_mask':       batch['bar_mask'].to(device, non_blocking=True),
            'context':        batch['context'].to(device, non_blocking=True),
        }
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(**inputs)
                total_loss, per_task = compute_ssl_loss(outputs, batch, device, config_weights)
        else:
            outputs = model(**inputs)
            total_loss, per_task = compute_ssl_loss(outputs, batch, device, config_weights)

        if not torch.isfinite(total_loss):
            n_skipped += 1
            continue
        for k, v in per_task.items():
            losses_sum[k] = losses_sum.get(k, 0.0) + v
        n_batches += 1

    if n_total > 0 and n_skipped / n_total > MAX_SKIP_RATIO:
        raise RuntimeError(
            f"eval: too many NaN batches ({n_skipped}/{n_total}) — "
            f"validation metric would be biased. Aborting."
        )
    if n_skipped > 0:
        print(f"           ⚠️  eval: skipped {n_skipped}/{n_total}")
    return {k: v / max(n_batches, 1) for k, v in losses_sum.items()}


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--lob-tensors', required=True)
    p.add_argument('--lob-timestamps', default=None)
    p.add_argument('--order-batches-dir', default=None)
    p.add_argument('--output', default='checkpoints/ssl_lob')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--train-split', type=float, default=0.75)
    p.add_argument('--lookback-bars', type=int, default=50)
    p.add_argument('--embargo-bars', type=int, default=24)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--early-stopping-patience', type=int, default=10)
    p.add_argument('--save-every', type=int, default=5)
    p.add_argument('--use-amp', action='store_true',
                  help='Enable real AMP (autocast + GradScaler). Default: off')
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

    print(f"═══ SSL Pretraining: HierarchicalLOBTransformer ═══")
    print(f"Device:       {device} (AMP={use_amp})")
    print(f"Features:     {args.features}")
    print(f"LOB tensors:  {args.lob_tensors}")
    print(f"Output:       {args.output}")
    print(f"Epochs:       {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}")
    print(f"Embargo:      {args.embargo_bars} bars (target-horizon safety)")
    print()

    train_loader, holdout_loader = build_ssl_loaders(
        args.features, args.lob_tensors, args.lob_timestamps,
        order_batches_dir=args.order_batches_dir,
        batch_size=args.batch_size, train_split=args.train_split,
        num_workers=args.num_workers, lookback_bars=args.lookback_bars,
        embargo_bars=args.embargo_bars,
    )
    print(f"Train batches: {len(train_loader)} | Holdout batches: {len(holdout_loader)}")
    print()

    print("🏗️  Building HierarchicalLOBTransformer...")
    config = DeepLOBConfig()
    config.multi_task_heads.direction_weight = 0.0
    model = HierarchicalLOBTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,}")
    print()

    # C7 fix: param groups with weight-decay split
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

    # C6 fix: scaler only constructed when AMP is actually used
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    patience = 0
    history = []
    cfg_weights = config.multi_task_heads

    for epoch in range(args.epochs):
        t0 = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, scaler, scheduler,
            cfg_weights, use_amp,
        )
        val_metrics = eval_epoch(model, holdout_loader, device, cfg_weights, use_amp)
        elapsed = time.time() - t0

        train_total = train_metrics.get('total', 0.0)
        val_total = val_metrics.get('total', 0.0)
        history.append({
            'epoch': epoch + 1,
            'train': train_metrics, 'val': val_metrics, 'time_sec': elapsed,
        })

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train={train_total:.4f} | val={val_total:.4f} | {elapsed:.1f}s")
        ssl_tasks = ['next_price', 'next_imbalance', 'next_volatility',
                     'next_regime', 'wall_persist', 'time_to_event']
        task_str = ' | '.join(
            f"{t}={val_metrics.get(t, 0.0):.4f}" for t in ssl_tasks if t in val_metrics
        )
        print(f"           val tasks: {task_str}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            patience = 0
            best_path = Path(args.output) / 'best_ssl_lob.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': asdict(config),
                'epoch': epoch + 1,
                'val_loss': val_total, 'val_metrics': val_metrics,
                'args': vars(args),
            }, best_path)
            print(f"           💾 saved best to {best_path}")
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                print(f"⏹️  Early stopping (patience={patience})")
                break

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': asdict(config),
                'epoch': epoch + 1,
            }, Path(args.output) / f'ssl_lob_epoch_{epoch+1}.pt')

    with open(Path(args.output) / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print()
    print(f"✅ SSL Pretraining complete | Best val loss: {best_val_loss:.4f}")
    print(f"   Output: {args.output}/best_ssl_lob.pt")


if __name__ == '__main__':
    main()
