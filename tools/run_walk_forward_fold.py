"""
tools/run_walk_forward_fold.py
═══════════════════════════════════════════════════════════════════════════════
One walk-forward fold: train hybrid on train slice, backtest on test slice.

Used by scripts/walk_forward_2022_2024.sh. Each fold:
  1. Slices the combined features parquet + SSL embeddings by date range
  2. Trains a HybridModel on the train slice only (no fold leakage)
  3. Predicts on the test slice
  4. Runs a strict backtest on the test slice
  5. Writes per-fold metrics.json

The day_trade rule labels are unchanged across folds — they're computed by
prepare_day_trading.py once. We're measuring whether the HYBRID model
generalizes from past windows to a future window.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.trading_intel.hybrid.model import (
    HybridConfig, HybridModel, HybridTargets, compute_hybrid_loss,
)


# ────────────────────────────────────────────────────────────────────
# Same leakage blocklist used by train_hybrid.py
# ────────────────────────────────────────────────────────────────────
LEAKAGE_PATTERNS = [
    "event_flag", "event_direction", "event_score", "is_event",
    "path_outcome", "bias_label", "label_confidence", "label_end_ts",
    "label_horizon_steps", "forward_return", "trade_duration",
    "soft_label", "soft_label_long", "soft_label_short",
    "neutral_reason", "event_label_tier", "train_event_flag",
    "is_train_slice", "is_holdout_slice", "is_purged_slice",
    "dataset_slice", "signal_quality",
]


def _is_leakage(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in LEAKAGE_PATTERNS)


def _select_features(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if not _is_leakage(c)]


def _zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0).astype(np.float32)
    sig = np.maximum(X.std(axis=0).astype(np.float32), 1e-6)
    return mu, sig


def _zscore_apply(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return ((X - mu) / sig).astype(np.float32)


def _build_event_label(df: pd.DataFrame) -> np.ndarray:
    return df["event_flag"].fillna(0).astype(np.float32).to_numpy()


def _build_direction_label(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ev = df["event_flag"].fillna(0).astype(int).to_numpy()
    raw = df["event_direction"].fillna(0).astype(int).to_numpy()
    label = np.where(raw == 1, 0, np.where(raw == -1, 1, 2)).astype(np.int64)
    valid = (ev == 1) & (raw != 0)
    return np.where(valid, label, -1).astype(np.int64), valid


def _slice_by_ts(df: pd.DataFrame, start_ym: str, end_ym: str) -> np.ndarray:
    """Return boolean mask for rows in [start_ym, end_ym] (inclusive months)."""
    start = pd.Timestamp(f"{start_ym}-01")
    # End-of-month for end_ym
    end_dt = pd.Timestamp(f"{end_ym}-01") + pd.offsets.MonthEnd(0)
    ts = pd.to_datetime(df["ts_event"])
    # Strip tz if present so comparison works
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(None)
    return ((ts >= start) & (ts <= end_dt)).to_numpy()


def run_fold(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   device: {device}")

    # ── Load ──
    df = pd.read_parquet(args.combined_features)
    df["ts_event"] = pd.to_datetime(df["ts_event"])
    if df["ts_event"].dt.tz is not None:
        df["ts_event"] = df["ts_event"].dt.tz_convert(None)

    emb = np.load(args.ssl_embeddings).astype(np.float32)
    valid_path = Path(args.embedding_valid)
    if valid_path.exists():
        emb_valid = np.load(valid_path).astype(bool)
    else:
        emb_valid = np.isfinite(emb).all(axis=1)

    if emb.shape[0] != len(df):
        raise ValueError(
            f"Shape mismatch: features {len(df)} vs embeddings {emb.shape[0]}"
        )

    # ── Build train / test masks by date ──
    # to_numpy() can return a read-only view; force a writeable copy
    train_mask = _slice_by_ts(df, args.train_start, args.train_end).copy()
    test_mask = _slice_by_ts(df, args.test_start, args.test_end).copy()
    # Embargo: drop last `embargo_bars` rows of train (right before test)
    if args.embargo_bars > 0:
        train_idxs = np.where(train_mask)[0]
        if len(train_idxs) > args.embargo_bars:
            train_mask[train_idxs[-args.embargo_bars:]] = False

    # Apply emb_valid + finite-feature requirement
    feat_cols = _select_features(df)
    X = df[feat_cols].fillna(0).to_numpy(dtype=np.float32)
    sample_valid = emb_valid & np.isfinite(X).all(axis=1)
    train_mask &= sample_valid
    test_mask &= sample_valid

    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    print(f"   train: {n_train:,}  test: {n_test:,}")
    if n_train < 200 or n_test < 50:
        raise RuntimeError(
            f"Fold too small: train={n_train}, test={n_test}"
        )

    # ── Labels ──
    y_event = _build_event_label(df)
    y_dir, dir_valid = _build_direction_label(df)

    # ── Normalize on train slice ──
    mu_d, sig_d = _zscore_fit(X[train_mask])
    X_norm = np.nan_to_num(_zscore_apply(X, mu_d, sig_d), nan=0.0)
    mu_s, sig_s = _zscore_fit(emb[train_mask])
    emb_norm = np.nan_to_num(_zscore_apply(emb, mu_s, sig_s), nan=0.0)

    # ── Model ──
    cfg = HybridConfig(
        daytrade_feature_dim=X_norm.shape[1],
        ssl_embed_dim=emb_norm.shape[1],
        cnn_embed_dim=0,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        n_layers=args.n_layers,
    )
    model = HybridModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # Materialize copies so PyTorch can wrap them as writable tensors
    Xd_t = torch.from_numpy(np.ascontiguousarray(X_norm)).to(device)
    Xs_t = torch.from_numpy(np.ascontiguousarray(emb_norm)).to(device)
    y_e_t = torch.from_numpy(np.ascontiguousarray(y_event)).to(device)
    y_d_t = torch.from_numpy(np.ascontiguousarray(y_dir)).to(device)
    dir_valid_t = torch.from_numpy(np.ascontiguousarray(dir_valid)).to(device)

    train_idx = torch.from_numpy(np.where(train_mask)[0]).to(device)
    test_idx = torch.from_numpy(np.where(test_mask)[0]).to(device)

    def _make_batch_targets(idx_t: torch.Tensor) -> HybridTargets:
        """Build HybridTargets for a batch. Direction labels carry the -1
        sentinel for invalid rows; compute_hybrid_loss now passes
        ignore_index=-1 to cross_entropy so those rows contribute 0 loss."""
        return HybridTargets(
            event_flag=y_e_t[idx_t],
            direction=y_d_t[idx_t] if dir_valid_t[idx_t].any() else None,
        )

    # ── Training loop (simple, no early stop here — fixed epochs) ──
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            out = model(Xd_t[idx], Xs_t[idx])
            tgt = _make_batch_targets(idx)
            optimizer.zero_grad()
            loss, _ = compute_hybrid_loss(out, tgt, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Track loss on a small slice of train for "best" selection (no leakage)
        model.eval()
        with torch.no_grad():
            n_val = min(500, len(train_idx))
            val_idx = train_idx[-n_val:]   # last train rows (chronologically)
            out_v = model(Xd_t[val_idx], Xs_t[val_idx])
            tgt_v = _make_batch_targets(val_idx)
            v_loss, _ = compute_hybrid_loss(out_v, tgt_v, cfg)
            if v_loss.item() < best_val_loss:
                best_val_loss = v_loss.item()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ── Predict on test slice ──
    model.eval()
    with torch.no_grad():
        out_test = model(Xd_t[test_idx], Xs_t[test_idx])
        event_prob = torch.sigmoid(out_test.event_logit).cpu().numpy()
        dir_probs = torch.softmax(out_test.direction_logits, dim=-1).cpu().numpy()
        confidence = torch.sigmoid(out_test.confidence_logit).cpu().numpy()

    test_df = df.loc[test_mask].copy().reset_index(drop=True)
    test_df["hybrid_event_prob"] = event_prob
    test_df["hybrid_p_long"] = dir_probs[:, 0]
    test_df["hybrid_p_short"] = dir_probs[:, 1]
    test_df["hybrid_p_neutral"] = dir_probs[:, 2]
    test_df["hybrid_confidence"] = confidence

    # ── Compute test metrics ──
    metrics = {
        "fold_id": args.fold_id,
        "train_period": f"{args.train_start}..{args.train_end}",
        "test_period": f"{args.test_start}..{args.test_end}",
        "n_train": n_train,
        "n_test": n_test,
        "best_val_loss": best_val_loss,
    }

    # day_trade rule baseline on test slice
    rule_events = test_df[test_df["event_flag"] == 1]
    if len(rule_events) > 0:
        # Forward-window outcome from path_outcome:
        # 0,1 = win (long_tp, short_tp); 2,3 = loss; 4 = timeout
        outcomes = rule_events["path_outcome"]
        wins = outcomes.isin([0, 1]).sum()
        losses = outcomes.isin([2, 3]).sum()
        metrics["rule_baseline"] = {
            "n_events": int(len(rule_events)),
            "wins": int(wins),
            "losses": int(losses),
            "hit_rate": float(wins / max(wins + losses, 1)),
        }

    # Hybrid filter — keep only events where hybrid agrees on direction
    if len(rule_events) > 0:
        re_full = test_df[test_df["event_flag"] == 1].copy()
        re_full["hybrid_dir"] = np.argmax(
            re_full[["hybrid_p_long", "hybrid_p_short", "hybrid_p_neutral"]].to_numpy(),
            axis=1,
        )
        # rule direction codes: 1=long, -1=short → map to 0/1 to match hybrid
        re_full["rule_dir_code"] = re_full["event_direction"].map(
            {1: 0, -1: 1, 0: 2}
        )
        agree = re_full["hybrid_dir"] == re_full["rule_dir_code"]
        re_full = re_full[agree]
        if len(re_full) > 0:
            outcomes2 = re_full["path_outcome"]
            wins2 = outcomes2.isin([0, 1]).sum()
            losses2 = outcomes2.isin([2, 3]).sum()
            metrics["hybrid_filtered"] = {
                "n_events": int(len(re_full)),
                "wins": int(wins2),
                "losses": int(losses2),
                "hit_rate": float(wins2 / max(wins2 + losses2, 1)),
            }
            metrics["lift_hit_rate"] = (
                metrics["hybrid_filtered"]["hit_rate"]
                - metrics["rule_baseline"]["hit_rate"]
            )

    # ── Write outputs ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(out_dir / "test_predictions.parquet", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"   ✅ fold {args.fold_id} done.")
    if "rule_baseline" in metrics:
        print(f"      rule baseline:    hit={metrics['rule_baseline']['hit_rate']:.3f}  "
              f"n={metrics['rule_baseline']['n_events']}")
    if "hybrid_filtered" in metrics:
        print(f"      hybrid filtered: hit={metrics['hybrid_filtered']['hit_rate']:.3f}  "
              f"n={metrics['hybrid_filtered']['n_events']}  "
              f"lift={metrics.get('lift_hit_rate', 0):+.3f}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold-id", required=True)
    p.add_argument("--combined-features", required=True)
    p.add_argument("--ssl-embeddings", required=True)
    p.add_argument("--embedding-valid", required=True)
    p.add_argument("--train-start", required=True)   # YYYY-MM
    p.add_argument("--train-end", required=True)
    p.add_argument("--test-start", required=True)
    p.add_argument("--test-end", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--embargo-bars", type=int, default=24)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()
    run_fold(args)


if __name__ == "__main__":
    main()
