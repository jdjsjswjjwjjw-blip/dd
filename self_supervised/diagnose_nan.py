"""
self_supervised/diagnose_nan.py
═══════════════════════════════════════════════════════════
يشخّص مصدر الـ NaN في SSL training:
   1. يفحص order_features.npy / order_masks.npy / lob_tensors.npy لكل bar
   2. يبني SSLDataset (holdout split) ويفحص كل batch
   3. يشغّل forward pass واحد ويحدد أي input/output فيه NaN
   4. يطبع تقرير دقيق

الاستخدام:
   python self_supervised/diagnose_nan.py \\
       --features pipeline_15min/day_trading_features.parquet \\
       --lob-tensors pipeline_15min/lob_tensors.npy \\
       --order-batches-dir checkpoints/ssl_15min/order_batches \\
       --train-split 0.75
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from self_supervised.data_loader import SSLDataset, build_ssl_loaders
from modules.deep_lob.config import DeepLOBConfig
from modules.deep_lob.hierarchical_model import HierarchicalLOBTransformer
from modules.deep_lob.multi_task_heads import MultiTaskTargets


def scan_array(name: str, arr: np.ndarray, sample_n: int = 100) -> dict:
    """يفحص array لـ NaN/Inf، يطبع توزيع لكل bar."""
    print(f"\n═══ Scanning {name} ═══")
    print(f"  shape: {arr.shape} | dtype: {arr.dtype}")

    # Per-bar NaN/Inf counts (sample to avoid full memory load)
    n_bars = arr.shape[0]
    sample_idx = np.linspace(0, n_bars - 1, min(sample_n, n_bars)).astype(int)
    nan_per_bar = []
    inf_per_bar = []
    for bi in sample_idx:
        x = np.asarray(arr[bi])
        nan_per_bar.append(int(np.isnan(x).sum()))
        inf_per_bar.append(int(np.isinf(x).sum()))
    nan_per_bar = np.array(nan_per_bar)
    inf_per_bar = np.array(inf_per_bar)

    total_nan = nan_per_bar.sum()
    total_inf = inf_per_bar.sum()
    bars_with_nan = int((nan_per_bar > 0).sum())
    bars_with_inf = int((inf_per_bar > 0).sum())

    print(f"  sampled {len(sample_idx)} bars (every {n_bars // len(sample_idx)}th)")
    print(f"  Total NaN in sample: {total_nan:,} | bars affected: {bars_with_nan}/{len(sample_idx)}")
    print(f"  Total Inf in sample: {total_inf:,} | bars affected: {bars_with_inf}/{len(sample_idx)}")

    # Show worst bars
    if bars_with_nan > 0:
        worst_idx = np.argsort(-nan_per_bar)[:5]
        print(f"  Worst NaN bars (sample idx → actual idx):")
        for wi in worst_idx:
            actual = sample_idx[wi]
            print(f"    bar {actual}: NaN={nan_per_bar[wi]}, Inf={inf_per_bar[wi]}")

    # Stats on first non-NaN bar
    for bi in sample_idx:
        x = np.asarray(arr[bi])
        finite = x[np.isfinite(x)]
        if len(finite) > 0:
            print(f"  First valid bar {bi}: min={finite.min():.4f}, max={finite.max():.4f}, "
                  f"mean={finite.mean():.4f}")
            break

    return {
        'n_bars': n_bars, 'total_nan': total_nan, 'total_inf': total_inf,
        'bars_with_nan': bars_with_nan, 'bars_with_inf': bars_with_inf,
    }


def diagnose_batch(batch: dict, label: str) -> dict:
    """يفحص batch dict من DataLoader."""
    print(f"\n  ── {label} ──")
    issues = {}
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            continue
        n_nan = int(torch.isnan(v).sum().item())
        n_inf = int(torch.isinf(v).sum().item())
        if n_nan > 0 or n_inf > 0:
            issues[k] = (n_nan, n_inf)
            print(f"    🚨 {k:25s} shape={tuple(v.shape)} dtype={v.dtype}: NaN={n_nan}, Inf={n_inf}")
        else:
            print(f"    ✅ {k:25s} shape={tuple(v.shape)} dtype={v.dtype}: "
                  f"min={v.float().min().item():.4f}, max={v.float().max().item():.4f}")
    return issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--lob-tensors', required=True)
    p.add_argument('--lob-timestamps', default=None)
    p.add_argument('--order-batches-dir', default=None)
    p.add_argument('--train-split', type=float, default=0.75)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    print("═══════════════════════════════════════════════════════════════")
    print("  NaN DIAGNOSTIC")
    print("═══════════════════════════════════════════════════════════════")

    # ── Phase 1: Scan raw arrays ──
    print("\n>>> PHASE 1: Raw array scan <<<")
    lob = np.load(args.lob_tensors, mmap_mode='r')
    scan_array('lob_tensors', lob)

    if args.order_batches_dir:
        of_path = os.path.join(args.order_batches_dir, 'order_features.npy')
        om_path = os.path.join(args.order_batches_dir, 'order_masks.npy')
        if os.path.exists(of_path):
            of = np.load(of_path, mmap_mode='r')
            scan_array('order_features', of)
        if os.path.exists(om_path):
            om = np.load(om_path, mmap_mode='r')
            scan_array('order_masks', om)

    # ── Phase 2: Build DataLoader & inspect batches ──
    print("\n\n>>> PHASE 2: DataLoader batches <<<")
    train_loader, holdout_loader = build_ssl_loaders(
        args.features, args.lob_tensors, args.lob_timestamps,
        order_batches_dir=args.order_batches_dir,
        batch_size=args.batch_size, train_split=args.train_split,
        num_workers=0,
    )

    print(f"\n📦 First TRAIN batch:")
    train_batch = next(iter(train_loader))
    train_issues = diagnose_batch(train_batch, "train batch[0]")

    print(f"\n📦 First HOLDOUT batch:")
    holdout_batch = next(iter(holdout_loader))
    holdout_issues = diagnose_batch(holdout_batch, "holdout batch[0]")

    # Scan more holdout batches to find pattern
    print(f"\n📦 Scanning ALL HOLDOUT batches for NaN:")
    holdout_nan_batches = 0
    holdout_total_batches = 0
    holdout_first_nan_idx = -1
    for bi, batch in enumerate(holdout_loader):
        holdout_total_batches += 1
        has_nan = False
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and (torch.isnan(v).any() or torch.isinf(v).any()):
                has_nan = True
                break
        if has_nan:
            holdout_nan_batches += 1
            if holdout_first_nan_idx < 0:
                holdout_first_nan_idx = bi
    print(f"   {holdout_nan_batches}/{holdout_total_batches} holdout batches have NaN/Inf")
    if holdout_first_nan_idx >= 0:
        print(f"   First NaN batch index: {holdout_first_nan_idx}")

    # ── Phase 3: Forward pass diagnostic ──
    print("\n\n>>> PHASE 3: Forward pass diagnostic <<<")
    device = torch.device(args.device)
    config = DeepLOBConfig()
    config.multi_task_heads.direction_weight = 0.0
    model = HierarchicalLOBTransformer(config).to(device).eval()

    # Use first holdout batch
    inputs = {
        'order_features': holdout_batch['order_features'].to(device),
        'order_masks': holdout_batch['order_masks'].to(device),
        'bar_mask': holdout_batch['bar_mask'].to(device),
        'context': holdout_batch['context'].to(device),
    }
    targets = MultiTaskTargets(
        direction=None,
        next_price=holdout_batch['next_price'].to(device),
        next_imbalance=holdout_batch['next_imbalance'].to(device),
        next_volatility=holdout_batch['next_volatility'].to(device),
        next_regime=holdout_batch['next_regime'].to(device),
        wall_persist=holdout_batch['wall_persist'].to(device),
        time_to_event=holdout_batch['time_to_event'].to(device),
    )

    with torch.no_grad():
        outputs = model(**inputs)
        print(f"\n  Outputs:")
        for k, v in outputs.to_dict().items():
            if not isinstance(v, torch.Tensor):
                continue
            n_nan = int(torch.isnan(v).sum().item())
            n_inf = int(torch.isinf(v).sum().item())
            marker = '🚨' if n_nan + n_inf > 0 else '✅'
            print(f"    {marker} {k:25s} shape={tuple(v.shape)}: NaN={n_nan}, Inf={n_inf}")

        losses = model.heads.compute_loss(outputs, targets)
        print(f"\n  Losses:")
        for k, v in losses.items():
            val = float(v.item())
            marker = '🚨' if not np.isfinite(val) else '✅'
            print(f"    {marker} {k:30s}: {val:.6f}")

    print("\n═══════════════════════════════════════════════════════════════")
    print("  DIAGNOSTIC COMPLETE")
    print("═══════════════════════════════════════════════════════════════")


if __name__ == '__main__':
    main()
