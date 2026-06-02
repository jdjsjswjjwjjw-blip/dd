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

from self_supervised.data_loader import SSLDataset


def load_lob_model(checkpoint_path: str, device: torch.device) -> HierarchicalLOBTransformer:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Prefer the config persisted in the checkpoint over a fresh
    # DeepLOBConfig() — defaults can drift between training and extraction
    # runs and silently produce a different architecture.
    saved_cfg = ckpt.get('config', None)
    if saved_cfg is None:
        print("⚠️  Checkpoint has no 'config' — using current DeepLOBConfig() defaults. "
              "If defaults have changed since training, embeddings will differ.")
        # D1 Phase 2: the normal path uses the checkpoint's saved config, which
        # carries the DERIVED context dim — so extraction rebuilds at the right
        # size. This fallback assumes the default dim; a derived-dim checkpoint
        # without a saved config will fail loudly at load_state_dict(strict=True),
        # which is correct (better a hard error than a silently wrong architecture).
        config = DeepLOBConfig()
    elif isinstance(saved_cfg, DeepLOBConfig):
        config = saved_cfg
    else:
        # Saved as a dict (asdict) — reconstruct
        try:
            model_cls = HierarchicalLOBTransformer
            model = model_cls.from_checkpoint(checkpoint_path, map_location=device)
            model.eval()
            return model.to(device)
        except Exception as e:
            print(f"⚠️  from_checkpoint failed ({e}); using fresh DeepLOBConfig()")
            config = DeepLOBConfig()

    model = HierarchicalLOBTransformer(config)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def load_cycle_model(checkpoint_path: str, device: torch.device,
                     input_dim: int = 14) -> PriceCycleModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get('config', None)
    if isinstance(saved_cfg, PriceCycleConfig):
        config = saved_cfg
    else:
        config = PriceCycleConfig()
        if hasattr(config, 'encoder') and hasattr(config.encoder, 'n_input_features'):
            config.encoder.n_input_features = input_dim
    model = PriceCycleModel(config)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


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
    p.add_argument('--embargo-bars', type=int, default=24,
                  help='Right-edge embargo to match SSL training. Last embargo_bars '
                       'rows produce no embedding (target lookahead not observable).')
    p.add_argument('--train-split', type=float, default=0.75,
                  help='Used for normalization-stats fit (must match SSL training)')
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"═══ Extract Embeddings ═══")
    print(f"Device: {device}")

    df = pd.read_parquet(args.features)
    n = len(df)
    print(f"Loading dataset: {n} bars")

    # Sample range: [lookback, n - embargo) — matches what was actually
    # trained against. Bars outside this range cannot produce a valid
    # embedding (input window incomplete or target window truncated).
    sample_lo = args.lookback_bars
    sample_hi = max(sample_lo + 1, n - args.embargo_bars)

    # Stats fit: TRAIN slice only — match the pretraining run.
    stats_lo = args.lookback_bars
    stats_hi = max(stats_lo + 1, int(n * args.train_split) - args.embargo_bars)

    dataset = SSLDataset(
        args.features, args.lob_tensors, args.lob_timestamps,
        order_batches_dir=args.order_batches_dir,
        lookback_bars=args.lookback_bars,
        min_idx=sample_lo,
        max_idx=sample_hi,
        stats_min_idx=stats_lo,
        stats_max_idx=stats_hi,
        embargo_bars=args.embargo_bars,
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
                lob_tensor=batch['lob_image'].to(device) if 'lob_image' in batch else None,
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

    # ── S7 FIX: pad with NaN (not zero) and emit an explicit valid mask ──
    # Previous version silently zero-padded the leading lookback_bars rows
    # AND the trailing rows that the dataset dropped. Downstream
    # fine_tune_direction.py and validate_ssl.py then used those zero
    # embeddings as if they were real — the direction head learned to
    # predict the majority class on zero input, biasing val accuracy and
    # balance metrics.
    #
    # Now: pad with NaN so any downstream consumer that forgets to filter
    # gets a loud failure instead of silently biased results. Also write a
    # `embedding_valid.npy` mask: True iff this row has a real embedding.
    full_micro = np.full((n, micro_arr.shape[1]), np.nan, dtype=np.float32)
    full_macro = np.full((n, macro_arr.shape[1]), np.nan, dtype=np.float32)
    full_micro[sample_lo: sample_lo + len(micro_arr)] = micro_arr
    full_macro[sample_lo: sample_lo + len(macro_arr)] = macro_arr

    embedding_valid = np.zeros(n, dtype=bool)
    embedding_valid[sample_lo: sample_lo + len(micro_arr)] = True

    # Concat seasonal/cycle features ONLY for valid rows (else those columns
    # leak structural info into the NaN region that downstream code might
    # silently overwrite with zeros)
    seasonal_cols = [c for c in df.columns if c.startswith(
        ('session_phase', 'time_since_', 'time_to_', 'dow_', 'is_', 'dom', 'woy_'))]
    # Exclude leakage-prone is_* columns that are forward-looking labels
    seasonal_cols = [c for c in seasonal_cols if c not in {
        'is_event', 'is_train_slice', 'is_holdout_slice', 'is_purged_slice'}]
    cycle_struct_cols = [c for c in df.columns if c.startswith('cycle_') and c not in
                         {'cycle_position',
                          'cycle_phase_acc_prob', 'cycle_phase_markup_prob',
                          'cycle_phase_dist_prob', 'cycle_phase_markdown_prob'}]

    seasonal_arr = (
        df[seasonal_cols].fillna(0).to_numpy(dtype=np.float32)
        if seasonal_cols else np.zeros((n, 0), dtype=np.float32)
    )
    cycle_struct_arr = (
        df[cycle_struct_cols].fillna(0).to_numpy(dtype=np.float32)
        if cycle_struct_cols else np.zeros((n, 0), dtype=np.float32)
    )

    combined = np.concatenate([full_micro, full_macro, seasonal_arr, cycle_struct_arr], axis=1)
    print(f"Combined embedding shape: {combined.shape}")
    print(f"Valid rows: {int(embedding_valid.sum())} / {n} "
          f"({100*embedding_valid.sum()/n:.1f}%)")

    out_path = Path(args.output_dir) / 'embeddings.npy'
    valid_path = Path(args.output_dir) / 'embedding_valid.npy'
    np.save(out_path, combined)
    np.save(valid_path, embedding_valid)

    meta = {
        'n_rows': int(n),
        'embedding_dim': int(combined.shape[1]),
        'micro_dim': int(full_micro.shape[1]),
        'macro_dim': int(full_macro.shape[1]),
        'sample_lo': int(sample_lo),
        'sample_hi': int(sample_hi),
        'stats_lo': int(stats_lo),
        'stats_hi': int(stats_hi),
        'embargo_bars': int(args.embargo_bars),
        'lookback_bars': int(args.lookback_bars),
        'train_split': float(args.train_split),
        'seasonal_cols': seasonal_cols,
        'cycle_structural_cols': cycle_struct_cols,
        'lob_checkpoint': args.lob_checkpoint,
        'cycle_checkpoint': args.cycle_checkpoint,
        'invalid_row_fill': 'NaN — downstream MUST filter on embedding_valid.npy',
    }
    with open(Path(args.output_dir) / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Embeddings saved to {out_path} (shape={combined.shape} dtype={combined.dtype})")
    print(f"✅ Validity mask saved to {valid_path}")


if __name__ == '__main__':
    main()
