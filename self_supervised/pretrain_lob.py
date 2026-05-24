"""
ssl/pretrain_lob.py
═══════════════════════════════════════════════════════════
SSL Pretraining للـ HierarchicalLOBTransformer (PR #17).

يدرّب 6 SSL heads (بدون direction):
    next_price, next_imbalance, next_volatility, next_regime,
    wall_persist, time_to_event
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from modules.deep_lob.config import DeepLOBConfig
from modules.deep_lob.hierarchical_model import HierarchicalLOBTransformer
from modules.deep_lob.multi_task_heads import MultiTaskTargets

from self_supervised.data_loader import build_ssl_loaders


def make_ssl_targets(batch: dict, device: torch.device) -> MultiTaskTargets:
    """Build MultiTaskTargets بدون direction (SSL pretraining only)."""
    return MultiTaskTargets(
        direction=None,
        next_price=batch['next_price'].to(device),
        next_imbalance=batch['next_imbalance'].to(device),
        next_volatility=batch['next_volatility'].to(device),
        next_regime=batch['next_regime'].to(device),
        wall_persist=batch['wall_persist'].to(device),
        time_to_event=batch['time_to_event'].to(device),
    )


def train_epoch(model, loader, optimizer, device, scaler=None, scheduler=None) -> dict:
    model.train()
    losses_sum = {}
    n_batches = 0
    for batch in loader:
        targets = make_ssl_targets(batch, device)
        inputs = {
            'order_features': batch['order_features'].to(device),
            'order_masks': batch['order_masks'].to(device),
            'bar_mask': batch['bar_mask'].to(device),
            'context': batch['context'].to(device),
        }
        optimizer.zero_grad()
        outputs = model(**inputs)
        losses = model.heads.compute_loss(outputs, targets)
        total_loss = losses['total']

        # NaN guard: skip batch لو الـ loss = NaN/Inf (يحمي من corruption)
        if not torch.isfinite(total_loss):
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        for k, v in losses.items():
            losses_sum[k] = losses_sum.get(k, 0.0) + float(v.item())
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in losses_sum.items()}


def eval_epoch(model, loader, device) -> dict:
    model.eval()
    losses_sum = {}
    n_batches = 0
    n_nan_batches = 0
    with torch.no_grad():
        for batch in loader:
            targets = make_ssl_targets(batch, device)
            inputs = {
                'order_features': batch['order_features'].to(device),
                'order_masks': batch['order_masks'].to(device),
                'bar_mask': batch['bar_mask'].to(device),
                'context': batch['context'].to(device),
            }
            outputs = model(**inputs)
            losses = model.heads.compute_loss(outputs, targets)
            # NaN guard في eval برضو
            if not torch.isfinite(losses['total']):
                n_nan_batches += 1
                continue
            for k, v in losses.items():
                losses_sum[k] = losses_sum.get(k, 0.0) + float(v.item())
            n_batches += 1
    if n_nan_batches > 0:
        print(f"           ⚠️  eval: skipped {n_nan_batches} NaN batches")
    return {k: v / max(n_batches, 1) for k, v in losses_sum.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--lob-tensors', required=True)
    p.add_argument('--lob-timestamps', default=None)
    p.add_argument('--order-batches-dir', default=None,
                  help='Optional: path to order_batches dir (real MBO orders).'
                       ' Recommended for iceberg detection.')
    p.add_argument('--output', default='checkpoints/ssl_lob')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--train-split', type=float, default=0.75)
    p.add_argument('--lookback-bars', type=int, default=50)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--early-stopping-patience', type=int, default=10)
    p.add_argument('--save-every', type=int, default=5)
    args = p.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"═══ SSL Pretraining: HierarchicalLOBTransformer ═══")
    print(f"Device: {device}")
    print(f"Features: {args.features}")
    print(f"LOB tensors: {args.lob_tensors}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")
    print()

    train_loader, holdout_loader = build_ssl_loaders(
        args.features, args.lob_tensors, args.lob_timestamps,
        order_batches_dir=args.order_batches_dir,
        batch_size=args.batch_size, train_split=args.train_split,
        num_workers=args.num_workers, lookback_bars=args.lookback_bars,
    )
    print(f"Train batches: {len(train_loader)} | Holdout batches: {len(holdout_loader)}")
    print()

    print("🏗️  Building HierarchicalLOBTransformer...")
    config = DeepLOBConfig()
    config.multi_task_heads.direction_weight = 0.0  # SSL: skip direction
    model = HierarchicalLOBTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    # Linear warmup للـ 5% الأولى — يمنع gradient explosions في أول epochs
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_loss = float('inf')
    patience = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device, scaler, scheduler)
        val_metrics = eval_epoch(model, holdout_loader, device)
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
                'epoch': epoch + 1, 'val_loss': val_total, 'val_metrics': val_metrics,
            }, best_path)
            print(f"           💾 saved best to {best_path}")
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                print(f"⏹️  Early stopping (patience={patience})")
                break

        if (epoch + 1) % args.save_every == 0:
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch + 1},
                      Path(args.output) / f'ssl_lob_epoch_{epoch+1}.pt')

    with open(Path(args.output) / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print(f"✅ SSL Pretraining complete | Best val loss: {best_val_loss:.4f}")
    print(f"   Output: {args.output}/best_ssl_lob.pt")


if __name__ == '__main__':
    main()
