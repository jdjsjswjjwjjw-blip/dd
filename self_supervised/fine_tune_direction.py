"""
ssl/fine_tune_direction.py
═══════════════════════════════════════════════════════════
Phase B: Direction Head Training باستخدام SSL embeddings.

يستخدم الـ embeddings المُستخرجة من Phase A لتدريب MLP صغير
على الـ directional labels فقط (LONG=0، SHORT=1).

الـ encoder المُجمَّد (frozen)، فقط direction head يتدرّب.
بسبب backbone قوي، 100-500 label كافية للتدريب.
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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class DirectionHead(nn.Module):
    """Small MLP for binary direction classification (LONG vs SHORT)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # LONG, SHORT
        )

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True, help='day_trading_features.parquet')
    p.add_argument('--embeddings', required=True, help='embeddings.npy')
    p.add_argument('--output', default='checkpoints/direction_head')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--train-split', type=float, default=0.75)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--early-stopping-patience', type=int, default=15)
    args = p.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"═══ Direction Head Fine-Tuning ═══")
    print(f"Device: {device}")

    print(f"Loading features from {args.features}...")
    df = pd.read_parquet(args.features)
    print(f"   {len(df):,} rows")

    print(f"Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings)
    print(f"   shape={embeddings.shape}")

    # Optional validity mask (recommended path — emitted by extract_embeddings)
    valid_path = Path(args.embeddings).parent / 'embedding_valid.npy'
    if valid_path.exists():
        embedding_valid = np.load(valid_path)
        print(f"   embedding_valid: {int(embedding_valid.sum())} / {len(embedding_valid)} valid")
    else:
        # Conservative fallback: any row containing NaN → invalid
        embedding_valid = np.isfinite(embeddings).all(axis=1)
        print(f"   embedding_valid: derived from NaN-check "
              f"({int(embedding_valid.sum())} / {len(embedding_valid)} valid)")

    if len(df) != len(embeddings):
        raise ValueError(f"Length mismatch: df={len(df)} embeddings={len(embeddings)}")

    # ── A4 FIX: calendar-based split BEFORE directional filter ──
    # Previous version: directional_mask first, then int(len(X)*0.75) split.
    # Because directional rows are sparse and unevenly distributed in time,
    # the boundary "75% of directional rows" did NOT correspond to a fixed
    # calendar cutoff — train and val rows were calendar-interleaved.
    # Now: pick a CALENDAR cutoff time, then filter directional+valid rows
    # on each side. This matches the SSL train/holdout boundary in time.
    ts = pd.to_datetime(df['ts_event']).to_numpy()
    n_total = len(df)
    split_calendar_idx = int(n_total * args.train_split)
    split_ts = ts[split_calendar_idx]
    print(f"   Calendar split @ row {split_calendar_idx} = {pd.Timestamp(split_ts)}")

    # Build full per-row mask: directional + embedding-valid
    bias = df['bias_label'].to_numpy()
    directional_mask = (bias == 0) | (bias == 1)
    base_mask = directional_mask & embedding_valid

    train_row_mask = base_mask & (np.arange(n_total) < split_calendar_idx)
    val_row_mask   = base_mask & (np.arange(n_total) >= split_calendar_idx)

    X_train = embeddings[train_row_mask].astype(np.float32)
    y_train = bias[train_row_mask].astype(np.int64)
    X_val = embeddings[val_row_mask].astype(np.float32)
    y_val = bias[val_row_mask].astype(np.int64)

    n_directional = int(directional_mask.sum())
    print(f"   Directional rows: {n_directional} ({n_directional/n_total*100:.1f}%)")
    print(f"   Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"   y_train: LONG={int((y_train==0).sum())}, SHORT={int((y_train==1).sum())}")
    print(f"   y_val:   LONG={int((y_val==0).sum())}, SHORT={int((y_val==1).sum())}")

    if len(X_train) < 30:
        print(f"❌ FATAL: less than 30 train rows after calendar+valid filter")
        sys.exit(1)
    if len(X_val) < 10:
        print(f"⚠️  Very few validation samples ({len(X_val)}) — bootstrap CIs in "
              f"validation will be wide. Results indicative only.")

    # Normalize features
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-6
    X_train_norm = (X_train - mu) / sigma
    X_val_norm = (X_val - mu) / sigma

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_norm), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_norm), torch.from_numpy(y_val)),
        batch_size=args.batch_size, shuffle=False,
    )

    # Build direction head
    input_dim = X_train.shape[1]
    model = DirectionHead(input_dim, args.hidden_dim, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Direction head parameters: {n_params:,} (input_dim={input_dim})")
    print()

    # Class weights for imbalance
    long_count = float((y_train == 0).sum())
    short_count = float((y_train == 1).sum())
    total = long_count + short_count
    weights = torch.tensor([
        total / max(long_count, 1) / 2.0,
        total / max(short_count, 1) / 2.0,
    ], dtype=torch.float32).to(device)
    print(f"   Class weights: LONG={weights[0]:.3f}, SHORT={weights[1]:.3f}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_acc = 0.0
    patience = 0
    history = []

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_correct += int((logits.argmax(dim=-1) == yb).sum().item())
            train_n += len(xb)
        train_loss /= train_n
        train_acc = train_correct / train_n

        # Val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_n = 0
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(xb)
                val_correct += int((logits.argmax(dim=-1) == yb).sum().item())
                val_n += len(xb)
                all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        val_loss /= max(val_n, 1)
        val_acc = val_correct / max(val_n, 1)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"train: loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val: loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            # S8 fix: save the architecture hyperparameters so validate_ssl
            # can reconstruct the model with the right hidden_dim and
            # dropout. Previously these defaulted to (64, 0.3) at load
            # time → shape mismatch on state_dict when training used
            # --hidden-dim 128.
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mu': mu, 'sigma': sigma,
                'input_dim': input_dim,
                'hidden_dim': args.hidden_dim,
                'dropout': args.dropout,
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'split_calendar_idx': int(split_calendar_idx),
                'split_ts': str(pd.Timestamp(split_ts)),
            }, Path(args.output) / 'best_direction_head.pt')
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                print(f"⏹️  Early stopping at epoch {epoch+1}")
                break

    print()
    print(f"✅ Direction Head Training Complete")
    print(f"   Best val accuracy: {best_val_acc:.3f}")
    print(f"   Baseline (random): 0.500")
    print(f"   Output: {args.output}/best_direction_head.pt")

    # Save history + summary
    with open(Path(args.output) / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    summary = {
        'best_val_acc': float(best_val_acc),
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'input_dim': int(input_dim),
        'epochs_trained': len(history),
    }
    with open(Path(args.output) / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
