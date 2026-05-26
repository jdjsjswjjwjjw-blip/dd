"""
self_supervised/train_hybrid.py
═══════════════════════════════════════════════════════════════════════════════
Train the HybridModel that fuses day_trade rule features + SSL embeddings
into event / direction / confidence predictions.

Inputs (paths):
  • --features          combined_6m/day_trading_features.parquet  (135 cols)
  • --ssl-embeddings    checkpoints/ssl_6m/embeddings/embeddings.npy
  • --embedding-valid   checkpoints/ssl_6m/embeddings/embedding_valid.npy

Outputs (in --output dir):
  • best_hybrid.pt              torch checkpoint
  • predictions.parquet         per-bar (ts, event_prob, dir_probs, confidence)
  • training_report.json        config + per-epoch metrics + final val metrics

The day_trade features are filtered to exclude any leakage-prone columns
(labels, outcomes, etc.). The whitelist is computed at runtime from the
column names — anything matching a known leakage pattern is dropped.

Training split:
  Same as the SSL pipeline: 75% calendar train / embargo / 25% val.
  Compatible with dataset_slice if present in the parquet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# From modules/trading_intel/training/ go up 3 levels to repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.trading_intel.hybrid.model import (
    HybridConfig,
    HybridModel,
    HybridOutput,
    HybridTargets,
    compute_hybrid_loss,
)


# ════════════════════════════════════════════════════════════════════
# Leakage-prone columns we MUST drop from the day_trade feature matrix
# ════════════════════════════════════════════════════════════════════
LEAKAGE_PATTERNS = [
    'event_flag', 'event_direction', 'event_score', 'is_event',
    'path_outcome', 'bias_label', 'label_confidence', 'label_end_ts',
    'label_horizon_steps', 'forward_return', 'trade_duration',
    'soft_label', 'soft_label_long', 'soft_label_short',
    'neutral_reason', 'event_label_tier', 'train_event_flag',
    'is_train_slice', 'is_holdout_slice', 'is_purged_slice',
    'dataset_slice', 'signal_quality',
]


def _is_leakage_col(name: str) -> bool:
    name_low = name.lower()
    return any(p in name_low for p in LEAKAGE_PATTERNS)


def _select_features(df: pd.DataFrame) -> list[str]:
    """Return list of numeric columns minus leakage-prone ones."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if not _is_leakage_col(c)]


def _build_event_label(df: pd.DataFrame) -> np.ndarray:
    """Binary: is this bar a tradeable event? (= rule's event_flag)"""
    return df['event_flag'].fillna(0).astype(np.float32).to_numpy()


def _build_direction_label(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """3-way label: 0=LONG, 1=SHORT, 2=NEUTRAL.

    Returns (label, valid). Only event_flag=1 AND non-zero direction rows are
    valid for training. Rows that are filtered out get label=-1 so any code
    that accidentally consumes them without checking `valid` will crash with
    a clear CE error instead of silently treating them as NEUTRAL.
    """
    ev = df['event_flag'].fillna(0).astype(int).to_numpy()
    dir_raw = df['event_direction'].fillna(0).astype(int).to_numpy()  # 1/-1/0
    label = np.where(dir_raw == 1, 0,
                     np.where(dir_raw == -1, 1, 2)).astype(np.int64)
    valid = (ev == 1) & (dir_raw != 0)
    # Sentinel -1 for invalid rows — surfaces silent leakage as a CE error
    label = np.where(valid, label, -1).astype(np.int64)
    return label, valid


def _build_confidence_label(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Confidence label = label_confidence column, mapped to [0, 1].

    Only event rows are valid.
    """
    if 'label_confidence' not in df.columns:
        return np.zeros(len(df), dtype=np.float32), np.zeros(len(df), dtype=bool)
    conf = df['label_confidence'].fillna(0).astype(np.float32).to_numpy()
    conf = np.clip(conf, 0.0, 1.0)
    ev = df['event_flag'].fillna(0).astype(int).to_numpy()
    valid = ev == 1
    return conf, valid


def _zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-column mu/sigma. sigma floored at 1e-6 to avoid div0."""
    mu = X.mean(axis=0).astype(np.float32)
    sig = X.std(axis=0).astype(np.float32)
    sig = np.maximum(sig, 1e-6)
    return mu, sig


def _zscore_apply(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return ((X - mu) / sig).astype(np.float32)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"═══ HybridModel Training ═══")
    print(f"Device: {device}")

    # ── Load data ──
    df = pd.read_parquet(args.features)
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    n = len(df)
    print(f"Features: {n:,} rows × {df.shape[1]} cols")

    feature_cols = _select_features(df)
    print(f"Selected {len(feature_cols)} non-leakage numeric columns")
    # Materialize feature matrix, replace NaN with 0 after z-score
    X_daytrade = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)

    # ── Load SSL embeddings ──
    ssl_emb = np.load(args.ssl_embeddings).astype(np.float32)
    if ssl_emb.shape[0] != n:
        raise ValueError(f"SSL embeddings rows {ssl_emb.shape[0]} != features rows {n}")
    if args.embedding_valid and Path(args.embedding_valid).exists():
        emb_valid = np.load(args.embedding_valid).astype(bool)
    else:
        emb_valid = np.isfinite(ssl_emb).all(axis=1)
    print(f"SSL embeddings: {ssl_emb.shape}  valid={int(emb_valid.sum()):,}/{n}")

    # ── Optional CNN embeddings ──
    cnn_emb: np.ndarray | None = None
    if args.cnn_embeddings and Path(args.cnn_embeddings).exists():
        cnn_emb = np.load(args.cnn_embeddings).astype(np.float32)
        if cnn_emb.shape[0] != n:
            raise ValueError(f"CNN embeddings rows {cnn_emb.shape[0]} != features rows {n}")
        print(f"CNN embeddings:  {cnn_emb.shape}")

    # ── Build labels ──
    y_event = _build_event_label(df)
    y_dir, dir_valid = _build_direction_label(df)
    y_conf, conf_valid = _build_confidence_label(df)
    print(f"Labels: event=1 → {int(y_event.sum()):,} bars  "
          f"directional={int(dir_valid.sum()):,} bars")

    # ── Sample-validity mask ──
    sample_valid = emb_valid & np.isfinite(X_daytrade).all(axis=1)
    print(f"Sample valid (emb + features): {int(sample_valid.sum()):,}")

    # ── Calendar split (75/25 by default, with embargo) ──
    if 'dataset_slice' in df.columns and args.use_dataset_slice:
        train_mask = (df['dataset_slice'] == 'train').to_numpy() & sample_valid
        val_mask = (df['dataset_slice'] == 'holdout').to_numpy() & sample_valid
        # apply embargo: drop last embargo rows of train + first embargo of val
        if args.embargo_bars > 0:
            train_idxs = np.where(train_mask)[0]
            if len(train_idxs) > args.embargo_bars:
                train_mask[train_idxs[-args.embargo_bars:]] = False
            val_idxs = np.where(val_mask)[0]
            if len(val_idxs) > args.embargo_bars:
                val_mask[val_idxs[:args.embargo_bars]] = False
        split_desc = f"dataset_slice (train→{int(train_mask.sum())} val→{int(val_mask.sum())})"
    else:
        split_row = int(n * args.train_split)
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        train_mask[:split_row - args.embargo_bars] = True
        val_mask[split_row + args.embargo_bars:] = True
        train_mask &= sample_valid
        val_mask &= sample_valid
        split_desc = f"calendar @ {args.train_split} (train→{int(train_mask.sum())} val→{int(val_mask.sum())})"
    print(f"Split: {split_desc}")

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError(
            f"Empty split: train={int(train_mask.sum())}, val={int(val_mask.sum())}. "
            f"Check that dataset_slice contains both 'train' and 'holdout' rows, "
            f"or use --train-split with a non-extreme value."
        )
    if train_mask.sum() < 200 or val_mask.sum() < 50:
        print("⚠️  Very few samples; results may be noisy.")

    # ── Z-score normalization fit on train slice ──
    print("Fitting z-score on train slice...")
    mu_d, sig_d = _zscore_fit(X_daytrade[train_mask])
    X_daytrade_norm = _zscore_apply(X_daytrade, mu_d, sig_d)
    # SSL embeddings: also z-score
    mu_s, sig_s = _zscore_fit(ssl_emb[train_mask])
    ssl_emb_norm = _zscore_apply(ssl_emb, mu_s, sig_s)
    # Replace any remaining NaN with 0
    X_daytrade_norm = np.nan_to_num(X_daytrade_norm, nan=0.0)
    ssl_emb_norm = np.nan_to_num(ssl_emb_norm, nan=0.0)
    if cnn_emb is not None:
        mu_c, sig_c = _zscore_fit(cnn_emb[train_mask])
        cnn_emb_norm = np.nan_to_num(_zscore_apply(cnn_emb, mu_c, sig_c), nan=0.0)
    else:
        cnn_emb_norm = None

    # ── Build model ──
    cfg = HybridConfig(
        daytrade_feature_dim=X_daytrade_norm.shape[1],
        ssl_embed_dim=ssl_emb_norm.shape[1],
        cnn_embed_dim=(cnn_emb_norm.shape[1] if cnn_emb_norm is not None else 0),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        n_layers=args.n_layers,
        w_event=args.w_event,
        w_direction=args.w_direction,
        w_confidence=args.w_confidence,
    )
    model = HybridModel(cfg).to(device)
    print(f"Model: {model.num_parameters():,} params  "
          f"(input dim {cfg.total_input_dim})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ── Tensorize and move to device ──
    Xd_t = torch.from_numpy(X_daytrade_norm).to(device)
    Xs_t = torch.from_numpy(ssl_emb_norm).to(device)
    Xc_t = torch.from_numpy(cnn_emb_norm).to(device) if cnn_emb_norm is not None else None
    y_e_t = torch.from_numpy(y_event).to(device)
    y_d_t = torch.from_numpy(y_dir).to(device)
    y_c_t = torch.from_numpy(y_conf).to(device)
    dir_valid_t = torch.from_numpy(dir_valid).to(device)
    conf_valid_t = torch.from_numpy(conf_valid).to(device)

    train_idx = torch.from_numpy(np.where(train_mask)[0]).to(device)
    val_idx = torch.from_numpy(np.where(val_mask)[0]).to(device)

    # ── Training loop ──
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    patience_left = args.patience
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xd = Xd_t[idx]
            xs = Xs_t[idx]
            xc = Xc_t[idx] if Xc_t is not None else None
            out = model(xd, xs, xc)
            tgt = HybridTargets(
                event_flag=y_e_t[idx],
                direction=y_d_t[idx] if dir_valid_t[idx].any() else None,
                confidence=y_c_t[idx] if conf_valid_t[idx].any() else None,
            )
            optimizer.zero_grad()
            loss, _ = compute_hybrid_loss(out, tgt, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = epoch_loss / max(1, n_batches)

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            xd_v = Xd_t[val_idx]
            xs_v = Xs_t[val_idx]
            xc_v = Xc_t[val_idx] if Xc_t is not None else None
            out_v = model(xd_v, xs_v, xc_v)
            tgt_v = HybridTargets(
                event_flag=y_e_t[val_idx],
                direction=y_d_t[val_idx] if dir_valid_t[val_idx].any() else None,
                confidence=y_c_t[val_idx] if conf_valid_t[val_idx].any() else None,
            )
            val_loss, val_metrics = compute_hybrid_loss(out_v, tgt_v, cfg)
            # Compute val accuracy
            event_prob = torch.sigmoid(out_v.event_logit).cpu().numpy()
            event_pred = (event_prob > 0.5).astype(int)
            event_acc = float((event_pred == y_e_t[val_idx].cpu().numpy()).mean())
            val_metrics['val_event_acc'] = event_acc
            if out_v.direction_logits is not None:
                dir_idx = val_idx[dir_valid_t[val_idx]]
                if len(dir_idx) > 0:
                    out_dir_v = model(Xd_t[dir_idx], Xs_t[dir_idx],
                                       Xc_t[dir_idx] if Xc_t is not None else None)
                    dir_pred = out_dir_v.direction_logits.argmax(dim=-1)
                    dir_acc = float((dir_pred == y_d_t[dir_idx]).float().mean())
                    val_metrics['val_dir_acc'] = dir_acc
                    val_metrics['val_dir_n'] = int(len(dir_idx))

        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': val_loss.item(), **val_metrics})

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f} | "
                  f"val={val_loss.item():.4f} | "
                  f"event_acc={val_metrics.get('val_event_acc', 0.0):.3f} | "
                  f"dir_acc={val_metrics.get('val_dir_acc', 0.0):.3f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            ckpt = {
                'state_dict': model.state_dict(),
                'config': cfg.__dict__,
                'epoch': epoch,
                'val_loss': best_val_loss,
                'feature_cols': feature_cols,
                'norm_daytrade': {'mu': mu_d.tolist(), 'sigma': sig_d.tolist()},
                'norm_ssl': {'mu': mu_s.tolist(), 'sigma': sig_s.tolist()},
            }
            if cnn_emb_norm is not None:
                ckpt['norm_cnn'] = {'mu': mu_c.tolist(), 'sigma': sig_c.tolist()}
            torch.save(ckpt, out_dir / 'best_hybrid.pt')
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"⏹️  Early stopping at epoch {epoch}")
                break

    print(f"\n✅ Training complete. Best val_loss={best_val_loss:.4f}")

    # ── Inference on full dataset → predictions.parquet ──
    print("Running inference on full dataset...")
    ckpt = torch.load(out_dir / 'best_hybrid.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    with torch.no_grad():
        out_all = model(Xd_t, Xs_t, Xc_t)
        event_prob = torch.sigmoid(out_all.event_logit).cpu().numpy()
        dir_probs = F.softmax(out_all.direction_logits, dim=-1).cpu().numpy()
        confidence = torch.sigmoid(out_all.confidence_logit).cpu().numpy()

    pred = pd.DataFrame({
        'ts_event': df['ts_event'].values,
        'emb_valid': emb_valid,
        'event_prob': event_prob,
        'p_long': dir_probs[:, 0],
        'p_short': dir_probs[:, 1],
        'p_neutral': dir_probs[:, 2],
        'confidence': confidence,
    })
    # Mask invalid rows
    invalid = ~emb_valid
    for c in ('event_prob', 'p_long', 'p_short', 'p_neutral', 'confidence'):
        pred.loc[invalid, c] = np.nan
    pred_path = out_dir / 'predictions.parquet'
    pred.to_parquet(pred_path, index=False)
    print(f"💾 {pred_path} ({len(pred):,} rows)")

    # ── Final report ──
    report = {
        'config': cfg.__dict__,
        'train_size': int(train_mask.sum()),
        'val_size': int(val_mask.sum()),
        'best_val_loss': best_val_loss,
        'history': history,
        'n_features': len(feature_cols),
        'has_cnn_embeddings': cnn_emb is not None,
    }
    report_path = out_dir / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f"💾 {report_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--ssl-embeddings', required=True)
    p.add_argument('--embedding-valid', default='')
    p.add_argument('--cnn-embeddings', default='',
                   help='Optional: human_lob_cnn output .npy. Disabled if empty.')
    p.add_argument('--output', required=True)
    p.add_argument('--train-split', type=float, default=0.75)
    p.add_argument('--embargo-bars', type=int, default=24)
    p.add_argument('--use-dataset-slice', action='store_true',
                   help='Use dataset_slice column if present (else use --train-split)')
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--n-layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--print-every', type=int, default=5)
    p.add_argument('--w-event', type=float, default=1.0)
    p.add_argument('--w-direction', type=float, default=1.5)
    p.add_argument('--w-confidence', type=float, default=0.5)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
