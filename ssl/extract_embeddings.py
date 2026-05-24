"""
ssl/extract_embeddings.py
═══════════════════════════════════════════════════════════
استخراج embeddings من الـ pretrained SSL encoders لكل bar.

Output:
    embeddings.npy   (N, 64+64+12+21) = combined embedding per bar
    metadata.json    column descriptions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch

from modules.deep_lob.config import DeepLOBConfig
from modules.deep_lob.hierarchical_model import HierarchicalLOBTransformer
from modules.price_cycle.config import PriceCycleConfig
from modules.price_cycle.cycle_model import PriceCycleModel

from ssl.data_loader import SSLDataset


def load_lob_model(checkpoint_path: str, device: torch.device) -> HierarchicalLOBTransformer:
    config = DeepLOBConfig()
    model = HierarchicalLOBTransformer(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def load_cycle_model(checkpoint_path: str, device: torch.device,
                     input_dim: int = 14) -> PriceCycleModel:
    config = PriceCycleConfig()
    if hasattr(config, 'encoder') and hasattr(config.encoder, 'input_dim'):
        config.encoder.input_dim = input_dim
    model = PriceCycleModel(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--lob-tensors', required=True)
    p.add_argument('--lob-timestamps', default=None)
    p.add_argument('--order-batches-dir', default=None)
    p.add_argument('--lob-checkpoint', required=True, help='best_ssl_lob.pt')
    p.add_argument('--cycle-checkpoint', required=True, help='best_ssl_cycle.pt')
    p.add_argument('--output-dir', default='checkpoints/embeddings')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--lookback-bars', type=int, default=50)
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"═══ Extract Embeddings ═══")
    print(f"Device: {device}")

    # Build dataset (full range, no split)
    df = pd.read_parquet(args.features)
    n = len(df)
    print(f"Loading dataset: {n} bars")

    dataset = SSLDataset(
        args.features, args.lob_tensors, args.lob_timestamps,
        order_batches_dir=args.order_batches_dir,
        lookback_bars=args.lookback_bars,
        min_idx=args.lookback_bars,
        max_idx=n,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
    )

    # Load models
    print(f"Loading LOB Transformer from {args.lob_checkpoint}...")
    lob_model = load_lob_model(args.lob_checkpoint, device)
    print(f"Loading Cycle Model from {args.cycle_checkpoint}...")
    sample_batch = next(iter(loader))
    cycle_dim = sample_batch['cycle_window'].shape[-1]
    cycle_model = load_cycle_model(args.cycle_checkpoint, device, input_dim=cycle_dim)

    # Determine embedding dims
    print("Determining embedding dims...")
    with torch.no_grad():
        outs_lob = lob_model(
            sample_batch['order_features'].to(device),
            sample_batch['order_masks'].to(device),
            sample_batch['bar_mask'].to(device),
            sample_batch['context'].to(device),
        )
        if hasattr(outs_lob, 'shared_embedding'):
            lob_emb_dim = outs_lob.shared_embedding.shape[-1]
        else:
            lob_emb_dim = 64
        outs_cycle = cycle_model(sample_batch['cycle_window'].to(device))
        if hasattr(outs_cycle, 'cycle_embedding'):
            cycle_emb_dim = outs_cycle.cycle_embedding.shape[-1]
        elif hasattr(outs_cycle, 'shared_embedding'):
            cycle_emb_dim = outs_cycle.shared_embedding.shape[-1]
        else:
            cycle_emb_dim = 64

    print(f"   LOB embedding dim: {lob_emb_dim}")
    print(f"   Cycle embedding dim: {cycle_emb_dim}")

    # Extract embeddings
    print(f"Extracting embeddings for {len(dataset):,} bars...")
    micro_embs = []
    macro_embs = []
    indices = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            outs_lob = lob_model(
                batch['order_features'].to(device),
                batch['order_masks'].to(device),
                batch['bar_mask'].to(device),
                batch['context'].to(device),
            )
            outs_cycle = cycle_model(batch['cycle_window'].to(device))

            if hasattr(outs_lob, 'shared_embedding'):
                micro_e = outs_lob.shared_embedding.cpu().numpy()
            else:
                micro_e = outs_lob['shared_embedding'].cpu().numpy() if isinstance(outs_lob, dict) else np.zeros((batch['order_features'].shape[0], lob_emb_dim))

            if hasattr(outs_cycle, 'cycle_embedding'):
                macro_e = outs_cycle.cycle_embedding.cpu().numpy()
            elif hasattr(outs_cycle, 'shared_embedding'):
                macro_e = outs_cycle.shared_embedding.cpu().numpy()
            else:
                macro_e = np.zeros((batch['cycle_window'].shape[0], cycle_emb_dim))

            micro_embs.append(micro_e)
            macro_embs.append(macro_e)
            if batch_idx % 50 == 0:
                print(f"   batch {batch_idx}/{len(loader)}")

    micro_arr = np.concatenate(micro_embs, axis=0).astype(np.float32)
    macro_arr = np.concatenate(macro_embs, axis=0).astype(np.float32)

    # Pad to full dataset length (first lookback_bars rows have zero embeddings)
    full_micro = np.zeros((n, micro_arr.shape[1]), dtype=np.float32)
    full_macro = np.zeros((n, macro_arr.shape[1]), dtype=np.float32)
    full_micro[args.lookback_bars: args.lookback_bars + len(micro_arr)] = micro_arr
    full_macro[args.lookback_bars: args.lookback_bars + len(macro_arr)] = macro_arr

    # Concat with seasonal + cycle structural features
    seasonal_cols = [c for c in df.columns if c.startswith(('session_phase', 'time_since_', 'time_to_', 'dow_', 'is_', 'dom', 'woy_'))]
    cycle_struct_cols = [c for c in df.columns if c.startswith('cycle_') and c not in
                         ('cycle_position',)]

    seasonal_arr = df[seasonal_cols].fillna(0).to_numpy(dtype=np.float32) if seasonal_cols else np.zeros((n, 0), dtype=np.float32)
    cycle_struct_arr = df[cycle_struct_cols].fillna(0).to_numpy(dtype=np.float32) if cycle_struct_cols else np.zeros((n, 0), dtype=np.float32)

    combined = np.concatenate([full_micro, full_macro, seasonal_arr, cycle_struct_arr], axis=1)
    print(f"Combined embedding shape: {combined.shape}")

    # Save
    out_path = Path(args.output_dir) / 'embeddings.npy'
    np.save(out_path, combined)

    meta = {
        'n_rows': int(n),
        'embedding_dim': int(combined.shape[1]),
        'micro_dim': int(full_micro.shape[1]),
        'macro_dim': int(full_macro.shape[1]),
        'seasonal_cols': seasonal_cols,
        'cycle_structural_cols': cycle_struct_cols,
        'lob_checkpoint': args.lob_checkpoint,
        'cycle_checkpoint': args.cycle_checkpoint,
    }
    with open(Path(args.output_dir) / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Embeddings saved to {out_path}")
    print(f"   shape: {combined.shape} ({combined.dtype})")


if __name__ == '__main__':
    main()
