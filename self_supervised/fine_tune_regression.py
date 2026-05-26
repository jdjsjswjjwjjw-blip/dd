"""
self_supervised/fine_tune_regression.py
═══════════════════════════════════════════════════════════
Phase D-Regression — multi-horizon trading head على SSL embeddings.

الـ architecture بيوصل بـ 2 use-case في موديل واحد:
  1. standalone trading signal — long/short/flat حسب predicted direction
     + magnitude في الـ horizons المختلفة
  2. confidence score لـ day_trade events — يعمل re-rank لإشارات
     day_trade القائمة باستخدام نفس السكور

التفاصيل:
  • لكل bar فيها embedding صالح، بنحسب 2 targets:
       r6  = log(close[t+6]  / close[t])  ← 1.5 ساعة
       r24 = log(close[t+24] / close[t])  ← 6 ساعات
  • الـ samples اللي forward window بتاعهم بيعدّي session_break بتتفلتر
  • لكل horizon h ∈ {6, 24}، الموديل بيخرج head واحد بـ 2 outputs:
       logit_up_h  → sigmoid → P(r_h > 0)
       mag_h       → |r_h| المتوقع
  • Combined sign+magnitude loss:
       loss_h = α · BCE(logit_up_h, sign(r_h))
              + β · SmoothL1(mag_h, |r_h|)
              + γ · directional_penalty(logit_up_h, r_h)
       حيث directional_penalty = max(0, -pred_return · actual_return)

  • Total loss = loss_6 + loss_24

الـ training set = جميع البارز بـ embedding_valid (مش الـ 414 event فقط)
→ يحل bottleneck الـ direction head القديم.

Output:
  best_regression_head.pt    ← weights
  predictions.parquet        ← per-bar (ts, prob_up_6, mag_6, prob_up_24,
                                mag_24, signed_score_6, signed_score_24)
  validation_report.json     ← metrics: IC, hit-rate, Sharpe (regression-driven)

استخدام:
  python self_supervised/fine_tune_regression.py \
      --features combined_6m/day_trading_features.parquet \
      --embeddings checkpoints/ssl_6m/embeddings/embeddings.npy \
      --embedding-valid checkpoints/ssl_6m/embeddings/embedding_valid.npy \
      --output checkpoints/ssl_6m/regression \
      --train-split 0.75
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


# ════════════════════════════════════════════════════════════════
# Architecture
# ════════════════════════════════════════════════════════════════
class MultiHorizonHead(nn.Module):
    """Shared backbone + per-horizon (direction + magnitude) heads.

    For each horizon h, the head outputs 2 scalars:
       logit_up_h  → P(r_h > 0) via sigmoid
       raw_mag_h   → predicted |r_h| via softplus (always ≥ 0)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        horizons: tuple[int, ...] = (6, 24),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.horizons = horizons
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({
            f'h{h}': nn.Linear(hidden_dim, 2) for h in horizons
        })

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        feat = self.backbone(x)
        out = {}
        for h in self.horizons:
            raw = self.heads[f'h{h}'](feat)
            out[f'h{h}'] = {
                'logit_up': raw[:, 0],
                'mag_raw': raw[:, 1],
                'mag': F.softplus(raw[:, 1]),
            }
        return out


# ════════════════════════════════════════════════════════════════
# Loss
# ════════════════════════════════════════════════════════════════
def combined_loss(
    pred: dict[str, dict[str, torch.Tensor]],
    targets: dict[int, torch.Tensor],
    target_valid: dict[int, torch.Tensor],
    alpha: float = 1.0,
    beta: float = 5.0,
    gamma: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-horizon: BCE(direction) + SmoothL1(magnitude) + directional penalty.

    alpha, beta, gamma — weights for the 3 terms.
    The directional penalty grows when predicted direction differs from
    actual: -pred_signed · actual_return (clamped at 0 so correct
    directions don't get rewarded into the loss).
    """
    total = 0.0
    metrics: dict[str, float] = {}
    for h_key, p in pred.items():
        h = int(h_key[1:])  # strip 'h' prefix
        ret = targets[h]
        valid = target_valid[h]
        if valid.sum() == 0:
            continue
        ret_v = ret[valid]
        logit_up = p['logit_up'][valid]
        mag = p['mag'][valid]

        # ── 1. Direction loss (BCE) ──
        y_up = (ret_v > 0).float()
        loss_dir = F.binary_cross_entropy_with_logits(logit_up, y_up)

        # ── 2. Magnitude loss (SmoothL1 / Huber) ──
        # Target is |actual_return|, weighted so big-move samples matter
        # (loss is bounded so we don't blow up on tail moves)
        loss_mag = F.smooth_l1_loss(mag, ret_v.abs(), beta=0.001)

        # ── 3. Directional penalty ──
        # pred_signed = (P_up - 0.5) * mag * 2 — proportional signed prediction
        p_up = torch.sigmoid(logit_up)
        pred_signed = (p_up * 2 - 1) * mag
        # PnL-style: if pred · actual < 0 we lose; minimize -E[pred · actual]
        loss_dirpen = (-pred_signed * ret_v).clamp(min=0).mean()

        loss_h = alpha * loss_dir + beta * loss_mag + gamma * loss_dirpen
        total = total + loss_h

        # Bookkeeping
        with torch.no_grad():
            pred_up = (logit_up > 0).float()
            acc = (pred_up == y_up).float().mean().item()
            metrics[f'h{h}_acc'] = acc
            metrics[f'h{h}_loss_dir'] = loss_dir.item()
            metrics[f'h{h}_loss_mag'] = loss_mag.item()
            metrics[f'h{h}_loss_dirpen'] = loss_dirpen.item()

    return total, metrics


# ════════════════════════════════════════════════════════════════
# Target builder
# ════════════════════════════════════════════════════════════════
def build_targets(
    df: pd.DataFrame, horizons: tuple[int, ...],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """For each horizon h: returns (log_return[i] = log(close[i+h]/close[i]),
    valid_mask[i]) where valid is True iff:
      - close[i] and close[i+h] are finite & positive
      - no is_session_break in (i, i+h]  (forward window doesn't cross break)
      - i + h < len(df)
    """
    n = len(df)
    close = pd.to_numeric(df['close'], errors='coerce').to_numpy(np.float64)
    close = np.where(np.isfinite(close) & (close > 0), close, np.nan)

    if 'is_session_break' in df.columns:
        sb = df['is_session_break'].astype(bool).to_numpy()
    else:
        sb = np.zeros(n, dtype=bool)
    # Cumulative break count for O(1) window queries
    sb_cs = np.concatenate([[0], np.cumsum(sb)]).astype(np.int64)

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for h in horizons:
        log_ret = np.full(n, np.nan, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        upper = n - h
        for i in range(upper):
            c0 = close[i]
            ch = close[i + h]
            if not (np.isfinite(c0) and np.isfinite(ch) and c0 > 0 and ch > 0):
                continue
            # forward window (i, i+h] — count breaks strictly after i, up to i+h
            n_break_fwd = int(sb_cs[i + h + 1] - sb_cs[i + 1])
            if n_break_fwd > 0:
                continue
            log_ret[i] = float(np.log(ch / c0))
            valid[i] = True
        out[h] = (log_ret, valid)
        print(f"  target h={h}: {valid.sum():,} valid / {n:,} ({100*valid.mean():.1f}%)")
    return out


# ════════════════════════════════════════════════════════════════
# Train
# ════════════════════════════════════════════════════════════════
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"═══ Multi-Horizon Regression Head ═══")
    print(f"Device: {device}")

    # ── Load ──
    df = pd.read_parquet(args.features)
    n = len(df)
    print(f"Features: {n:,} rows")

    emb = np.load(args.embeddings).astype(np.float32)
    print(f"Embeddings: {emb.shape}")
    if emb.shape[0] != n:
        raise ValueError(f"embeddings rows {emb.shape[0]} != features rows {n}")

    valid_mask_path = Path(args.embedding_valid)
    if valid_mask_path.exists():
        emb_valid = np.load(valid_mask_path)
        print(f"embedding_valid: {emb_valid.sum():,} / {n}")
    else:
        emb_valid = np.isfinite(emb).all(axis=1)
        print(f"embedding_valid (inferred from NaN): {emb_valid.sum():,} / {n}")

    # ── Targets ──
    horizons = tuple(int(h) for h in args.horizons)
    print(f"Building targets for horizons={horizons}...")
    targets = build_targets(df, horizons)

    # Per-sample validity: embedding valid AND target valid for ALL horizons we use
    sample_valid = emb_valid.copy()
    for h in horizons:
        sample_valid &= targets[h][1]
    print(f"Final usable samples (emb_valid ∩ all-horizons-valid): {sample_valid.sum():,}")

    # ── Calendar split ──
    split_row = int(n * args.train_split)
    embargo = args.embargo_bars
    train_mask = sample_valid.copy()
    train_mask[split_row - embargo:] = False
    val_mask = sample_valid.copy()
    val_mask[:split_row + embargo] = False

    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())
    print(f"Split @ row {split_row} (ts ≈ {df['ts_event'].iloc[split_row]})")
    print(f"Train: {n_train:,} | Val: {n_val:,} (embargo={embargo})")
    if n_train < 200 or n_val < 50:
        print(f"⚠️  Small sample counts — results may be noisy")

    # ── Build tensors ──
    X_train = torch.from_numpy(emb[train_mask])
    X_val = torch.from_numpy(emb[val_mask])
    # Replace any residual NaN in embeddings with 0 (shouldn't happen after filter)
    X_train = torch.nan_to_num(X_train, nan=0.0)
    X_val = torch.nan_to_num(X_val, nan=0.0)

    y_train = {h: torch.from_numpy(targets[h][0][train_mask]).float() for h in horizons}
    y_val = {h: torch.from_numpy(targets[h][0][val_mask]).float() for h in horizons}
    # All-valid since we masked already, but pass mask of trues to keep code uniform
    v_train = {h: torch.ones(n_train, dtype=torch.bool) for h in horizons}
    v_val = {h: torch.ones(n_val, dtype=torch.bool) for h in horizons}

    # ── Model ──
    model = MultiHorizonHead(
        input_dim=emb.shape[1],
        hidden_dim=args.hidden_dim,
        horizons=horizons,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # Move to device
    X_train, X_val = X_train.to(device), X_val.to(device)
    y_train = {h: t.to(device) for h, t in y_train.items()}
    y_val = {h: t.to(device) for h, t in y_val.items()}
    v_train = {h: t.to(device) for h, t in v_train.items()}
    v_val = {h: t.to(device) for h, t in v_val.items()}

    # ── Training loop ──
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_left = args.patience
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Mini-batch through random permutation
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = X_train[idx]
            yb = {h: y_train[h][idx] for h in horizons}
            vb = {h: v_train[h][idx] for h in horizons}
            optimizer.zero_grad()
            pred = model(xb)
            loss, _ = combined_loss(
                pred, yb, vb,
                alpha=args.alpha_dir, beta=args.beta_mag, gamma=args.gamma_dirpen,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = epoch_loss / max(1, n_batches)

        # Validate
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            val_loss, val_metrics = combined_loss(
                pred_val, y_val, v_val,
                alpha=args.alpha_dir, beta=args.beta_mag, gamma=args.gamma_dirpen,
            )

        log_row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss.item(),
            **{f'val_{k}': v for k, v in val_metrics.items()},
        }
        history.append(log_row)

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            acc_str = ' '.join(f"h{h}_acc={val_metrics.get(f'h{h}_acc', float('nan')):.3f}"
                               for h in horizons)
            print(f"Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f} | "
                  f"val={val_loss.item():.4f} | {acc_str}")

        # Save best
        primary_acc = val_metrics.get(f'h{horizons[0]}_acc', 0.0)
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_val_acc = primary_acc
            torch.save({
                'state_dict': model.state_dict(),
                'config': {
                    'input_dim': emb.shape[1],
                    'hidden_dim': args.hidden_dim,
                    'horizons': list(horizons),
                    'dropout': args.dropout,
                },
                'epoch': epoch,
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
            }, out_dir / 'best_regression_head.pt')
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"⏹️  Early stopping at epoch {epoch}")
                break

    print(f"\n✅ Training complete. Best val_loss={best_val_loss:.4f}, "
          f"primary h{horizons[0]}_acc={best_val_acc:.3f}")

    # ── Final inference on ALL valid bars → per-bar predictions ──
    print(f"\n=== Inference on full dataset ===")
    ckpt = torch.load(out_dir / 'best_regression_head.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    pred_rows = []
    X_all = torch.nan_to_num(torch.from_numpy(emb).to(device), nan=0.0)
    with torch.no_grad():
        # Chunked to save memory on long datasets
        preds_per_h = {h: {'p_up': [], 'mag': []} for h in horizons}
        for i in range(0, n, 1024):
            xb = X_all[i:i + 1024]
            out = model(xb)
            for h_key, p in out.items():
                h = int(h_key[1:])
                preds_per_h[h]['p_up'].append(torch.sigmoid(p['logit_up']).cpu().numpy())
                preds_per_h[h]['mag'].append(p['mag'].cpu().numpy())

    preds_concat: dict[int, dict[str, np.ndarray]] = {}
    for h in horizons:
        preds_concat[h] = {
            'p_up': np.concatenate(preds_per_h[h]['p_up']),
            'mag': np.concatenate(preds_per_h[h]['mag']),
        }

    # Build predictions dataframe
    out_df = pd.DataFrame({'ts_event': df['ts_event'].values})
    out_df['emb_valid'] = emb_valid
    for h in horizons:
        p_up = preds_concat[h]['p_up']
        mag = preds_concat[h]['mag']
        signed_score = (p_up * 2 - 1) * mag
        out_df[f'prob_up_{h}'] = p_up
        out_df[f'mag_{h}'] = mag
        out_df[f'signed_score_{h}'] = signed_score
        out_df[f'target_ret_{h}'] = targets[h][0]
        out_df[f'target_valid_{h}'] = targets[h][1]
        # NaN out invalid embedding rows
        out_df.loc[~emb_valid, f'prob_up_{h}'] = np.nan
        out_df.loc[~emb_valid, f'mag_{h}'] = np.nan
        out_df.loc[~emb_valid, f'signed_score_{h}'] = np.nan

    pred_path = out_dir / 'predictions.parquet'
    out_df.to_parquet(pred_path, index=False)
    print(f"💾 {pred_path} — {len(out_df):,} rows")

    # ── Validation metrics: IC + hit-rate + per-horizon Sharpe ──
    val_metrics_report = {}
    print(f"\n=== Validation Metrics ===")
    for h in horizons:
        msk = val_mask
        if msk.sum() == 0:
            continue
        p_up = preds_concat[h]['p_up'][msk]
        mag = preds_concat[h]['mag'][msk]
        signed = (p_up * 2 - 1) * mag
        target = targets[h][0][msk]
        valid_target = targets[h][1][msk]

        usable = valid_target & np.isfinite(target) & np.isfinite(signed)
        if usable.sum() == 0:
            continue
        p_up_u = p_up[usable]
        signed_u = signed[usable]
        target_u = target[usable]

        # Pearson IC: correlation between signed_score and actual return
        if signed_u.std() > 0 and target_u.std() > 0:
            ic = float(np.corrcoef(signed_u, target_u)[0, 1])
        else:
            ic = 0.0
        # Hit rate: sign match
        hit = float((np.sign(signed_u) == np.sign(target_u)).mean())
        # Spearman (rank-IC)
        rank_ic = float(pd.Series(signed_u).corr(pd.Series(target_u), method='spearman'))
        # Annualized Sharpe (signal * actual_return, non-trade-cost-adjusted)
        pnl_per_bar = np.sign(signed_u) * target_u  # naive: take sign as position
        if pnl_per_bar.std() > 0:
            bars_per_year = 252 * 24 * 4  # 15min bars
            sharpe = float(pnl_per_bar.mean() / pnl_per_bar.std() * np.sqrt(bars_per_year))
        else:
            sharpe = 0.0
        mean_ret = float(pnl_per_bar.mean())

        val_metrics_report[f'h{h}'] = {
            'n_val': int(usable.sum()),
            'pearson_ic': ic,
            'spearman_ic': rank_ic,
            'hit_rate': hit,
            'naive_sharpe': sharpe,
            'mean_return_per_bar': mean_ret,
        }
        print(f"  h={h}: n={int(usable.sum()):,} | IC={ic:+.4f} | "
              f"rank-IC={rank_ic:+.4f} | hit={hit:.3f} | "
              f"Sharpe={sharpe:+.2f} | mean_ret={mean_ret:+.5f}")

    report = {
        'config': vars(args),
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'train_size': n_train,
        'val_size': n_val,
        'split_row': split_row,
        'horizons': list(horizons),
        'val_metrics': val_metrics_report,
        'history': history,
    }
    report_path = out_dir / 'regression_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f"💾 {report_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--embeddings', required=True)
    p.add_argument('--embedding-valid', default='',
                   help='Path to embedding_valid.npy mask. Default: <embeddings_dir>/embedding_valid.npy')
    p.add_argument('--output', required=True)
    p.add_argument('--train-split', type=float, default=0.75)
    p.add_argument('--embargo-bars', type=int, default=24)
    p.add_argument('--horizons', nargs='+', default=[6, 24],
                   help='Forward horizons in bars (e.g. 6 24)')
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--print-every', type=int, default=5)
    p.add_argument('--alpha-dir', type=float, default=1.0,
                   help='Weight on direction (BCE) loss')
    p.add_argument('--beta-mag', type=float, default=5.0,
                   help='Weight on magnitude (SmoothL1) loss')
    p.add_argument('--gamma-dirpen', type=float, default=0.5,
                   help='Weight on directional penalty (pnl-like)')
    args = p.parse_args()

    if not args.embedding_valid:
        args.embedding_valid = str(Path(args.embeddings).parent / 'embedding_valid.npy')

    train(args)


if __name__ == '__main__':
    main()
