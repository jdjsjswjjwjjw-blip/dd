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
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.price_cycle.config import PriceCycleConfig
from modules.price_cycle.cycle_model import PriceCycleModel, CycleTargets

from ssl.data_loader import build_ssl_loaders


def make_cycle_targets(batch: dict, device: torch.device) -> CycleTargets:
    """Build CycleTargets من SSL batch (no direction)."""
    return CycleTargets(
        phase=batch['phase_target'].to(device),
        maturity=batch['maturity_target'].to(device),
        swing=batch['swing_target'].to(device),
        cycle_position=batch['cycle_position_target'].to(device),
        reversal_proximity=None,
    )


def train_cycle_epoch(model, loader, optimizer, device, scaler=None) -> dict:
    model.train()
    losses_sum = {}
    n_batches = 0
    for batch in loader:
        cycle_input = batch['cycle_window'].to(device)
        targets = make_cycle_targets(batch, device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(cycle_input)
                losses = model.heads.compute_loss(outputs, targets)
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(cycle_input)
            losses = model.heads.compute_loss(outputs, targets)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        for k, v in losses.items():
            losses_sum[k] = losses_sum.get(k, 0.0) + float(v.item())
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in losses_sum.items()}


def eval_cycle_epoch(model, loader, device) -> dict:
    model.eval()
    losses_sum = {}
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            cycle_input = batch['cycle_window'].to(device)
            targets = make_cycle_targets(batch, device)
            outputs = model(cycle_input)
            losses = model.heads.compute_loss(outputs, targets)
            for k, v in losses.items():
                losses_sum[k] = losses_sum.get(k, 0.0) + float(v.item())
            n_batches += 1
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
    args = p.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"═══ SSL Pretraining: PriceCycleModel ═══")
    print(f"Device: {device}\nOutput: {args.output}")

    train_loader, holdout_loader = build_ssl_loaders(
        args.features, args.lob_tensors, args.lob_timestamps,
        order_batches_dir=args.order_batches_dir,
        batch_size=args.batch_size, train_split=args.train_split,
        num_workers=args.num_workers, lookback_bars=args.lookback_bars,
    )

    print("🏗️  Building PriceCycleModel...")
    sample_batch = next(iter(train_loader))
    cycle_dim = sample_batch['cycle_window'].shape[-1]
    config = PriceCycleConfig()
    # Adjust input dim if needed
    if hasattr(config, 'encoder') and hasattr(config.encoder, 'input_dim'):
        config.encoder.input_dim = cycle_dim
    model = PriceCycleModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,} | Input dim: {cycle_dim}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_loss = float('inf')
    patience = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_metrics = train_cycle_epoch(model, train_loader, optimizer, device, scaler)
        val_metrics = eval_cycle_epoch(model, holdout_loader, device)
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1, 'val_loss': val_total, 'val_metrics': val_metrics,
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
