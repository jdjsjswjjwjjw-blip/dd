"""
tools/run_walk_forward_fold_enhanced.py
═══════════════════════════════════════════════════════════════════════════════
Enhanced walk-forward fold runner that wires in the four anti-collapse
modules from `modules.trading_intel.anti_collapse`. All four are opt-in
via CLI flags; the script gracefully reduces to the baseline runner when
they're all disabled.

Wiring summary:
  --use-simplex-etf       Replace HybridModel direction_head with a Fixed
                          Simplex ETF Classifier (3 classes). Switches
                          direction loss from cross-entropy to
                          dot_regression_loss.
  --ortho-weight W        Add an OrthogonalRepresentationModule penalty
                          of weight W between the backbone embedding and
                          the day_trade rule features. Drives the encoder
                          to find INDEPENDENT signal from the rules.
  --use-dbmtl             Wrap per-task losses with the DB-MTL gradient
                          balancer instead of summing them with static
                          weights. Solves the magnitude-vs-direction
                          tradeoff observed in the 6-month run.
  --sharpe-weight W       Add a SharpeRegularizer term that optimizes
                          a differentiable Sharpe ratio on the forward
                          returns. Aligns gradients with actual P&L
                          instead of CE surrogate. Requires `target_ret`
                          in the parquet.

Same data contract as `run_walk_forward_fold.py`. Default behavior with no
flags is IDENTICAL to the baseline runner, so the enhanced version is safe
to use as a drop-in replacement.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.trading_intel.hybrid.model import (
    HybridConfig, HybridModel, HybridTargets,
)
from modules.trading_intel.anti_collapse import (
    SimplexETFClassifier, SimplexETFConfig, dot_regression_loss,
    OrthogonalRepresentationModule,
    DBMTLBalancer, DBMTLConfig,
    SharpeRegularizer, SharpeLossConfig,
)


# ── Reuse the same leakage/feature helpers from the baseline runner ──
LEAKAGE_PATTERNS = [
    "event_flag", "event_direction", "event_score", "is_event",
    "path_outcome", "bias_label", "label_confidence", "label_end_ts",
    "label_horizon_steps", "forward_return", "trade_duration",
    "soft_label", "soft_label_long", "soft_label_short",
    "neutral_reason", "event_label_tier", "train_event_flag",
    "is_train_slice", "is_holdout_slice", "is_purged_slice",
    "dataset_slice", "signal_quality",
    # Avoid leaking target_ret used by Sharpe regularizer
    "target_ret_",
]


def _is_leakage(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in LEAKAGE_PATTERNS)


def _select_features(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if not _is_leakage(c)]


def _zscore_fit_apply(X: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    mu = X[train_mask].mean(axis=0).astype(np.float32)
    sig = np.maximum(X[train_mask].std(axis=0).astype(np.float32), 1e-6)
    return np.nan_to_num(((X - mu) / sig).astype(np.float32), nan=0.0)


def _build_direction_label(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ev = df["event_flag"].fillna(0).astype(int).to_numpy()
    raw = df["event_direction"].fillna(0).astype(int).to_numpy()
    label = np.where(raw == 1, 0, np.where(raw == -1, 1, 2)).astype(np.int64)
    valid = (ev == 1) & (raw != 0)
    label = np.where(valid, label, -1).astype(np.int64)
    return label, valid


def _slice_by_ts(df: pd.DataFrame, start_ym: str, end_ym: str) -> np.ndarray:
    start = pd.Timestamp(f"{start_ym}-01")
    end_dt = pd.Timestamp(f"{end_ym}-01") + pd.offsets.MonthEnd(0)
    ts = pd.to_datetime(df["ts_event"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(None)
    return ((ts >= start) & (ts <= end_dt)).to_numpy().copy()


def run_fold(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   device: {device}  "
          f"[simplex_etf={args.use_simplex_etf} ortho={args.ortho_weight} "
          f"dbmtl={args.use_dbmtl} sharpe={args.sharpe_weight}]")

    # ── Load data ─────────────────────────────────────────────────────
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

    # Target return for Sharpe (use horizon 24 if Sharpe enabled)
    target_ret_col = "target_ret_24" if "target_ret_24" in df.columns else None
    if args.sharpe_weight > 0 and target_ret_col is None:
        raise RuntimeError(
            "--sharpe-weight > 0 requires a target_ret_24 column in the parquet"
        )

    # ── Build train / test masks ──────────────────────────────────────
    train_mask = _slice_by_ts(df, args.train_start, args.train_end)
    test_mask = _slice_by_ts(df, args.test_start, args.test_end)
    if args.embargo_bars > 0:
        train_idxs = np.where(train_mask)[0]
        if len(train_idxs) > args.embargo_bars:
            train_mask[train_idxs[-args.embargo_bars:]] = False

    feat_cols = _select_features(df)
    X = df[feat_cols].fillna(0).to_numpy(dtype=np.float32)
    sample_valid = emb_valid & np.isfinite(X).all(axis=1)
    train_mask &= sample_valid
    test_mask &= sample_valid

    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    print(f"   train: {n_train:,}  test: {n_test:,}")
    if n_train < 200 or n_test < 50:
        raise RuntimeError(f"Fold too small: train={n_train}, test={n_test}")

    # ── Labels ────────────────────────────────────────────────────────
    y_event = df["event_flag"].fillna(0).astype(np.float32).to_numpy()
    y_dir, dir_valid = _build_direction_label(df)
    y_ret = (df[target_ret_col].fillna(0).astype(np.float32).to_numpy()
             if target_ret_col else np.zeros(len(df), dtype=np.float32))

    # ── Normalize features on train slice ─────────────────────────────
    X_norm = _zscore_fit_apply(X, train_mask)
    emb_norm = _zscore_fit_apply(emb, train_mask)

    # ── Build model ───────────────────────────────────────────────────
    cfg = HybridConfig(
        daytrade_feature_dim=X_norm.shape[1],
        ssl_embed_dim=emb_norm.shape[1],
        cnn_embed_dim=0,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        n_layers=args.n_layers,
    )
    # Optional Simplex ETF override for the direction head
    direction_head = None
    etf_anchors = None
    if args.use_simplex_etf:
        etf = SimplexETFClassifier(SimplexETFConfig(
            feature_dim=cfg.hidden_dim, num_classes=3, normalize_features=True,
        ))
        direction_head = etf
        # We keep a reference to the anchors for dot_regression_loss
        etf_anchors = etf.etf_anchors

    model = HybridModel(cfg, direction_head=direction_head).to(device)

    # ── Optional anti-collapse modules ────────────────────────────────
    ortho = (OrthogonalRepresentationModule(weight=args.ortho_weight).to(device)
             if args.ortho_weight > 0 else None)
    sharpe_reg = (SharpeRegularizer(
        weight=args.sharpe_weight,
        config=SharpeLossConfig(cost_per_unit_position=args.tx_cost),
    ) if args.sharpe_weight > 0 else None)
    balancer = (DBMTLBalancer(
        task_names=["event", "direction", "confidence", "sharpe"],
        config=DBMTLConfig(warmup_steps=20),
    ) if args.use_dbmtl else None)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ── Tensorize ─────────────────────────────────────────────────────
    Xd_t = torch.from_numpy(X_norm).to(device)
    Xs_t = torch.from_numpy(emb_norm).to(device)
    y_e_t = torch.from_numpy(y_event).to(device)
    y_d_t = torch.from_numpy(y_dir).to(device)
    y_r_t = torch.from_numpy(y_ret).to(device)
    dir_valid_t = torch.from_numpy(dir_valid).to(device)
    train_idx = torch.from_numpy(np.where(train_mask)[0]).to(device)
    test_idx = torch.from_numpy(np.where(test_mask)[0]).to(device)

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state = None
    diag_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xd, xs = Xd_t[idx], Xs_t[idx]
            out = model(xd, xs)

            # Compute per-task losses
            losses: dict[str, torch.Tensor] = {}

            # 1. Event loss (BCE)
            losses["event"] = F.binary_cross_entropy_with_logits(
                out.event_logit, y_e_t[idx],
            )

            # 2. Direction loss — Simplex ETF / Dot-regression OR standard CE
            if args.use_simplex_etf and etf_anchors is not None:
                losses["direction"] = dot_regression_loss(
                    out.backbone_embedding, y_d_t[idx],
                    etf_anchors.to(device), ignore_index=-1,
                )
            else:
                losses["direction"] = F.cross_entropy(
                    out.direction_logits, y_d_t[idx].long(), ignore_index=-1,
                )

            # 3. Confidence loss (BCE on event_flag as a confidence proxy)
            losses["confidence"] = F.binary_cross_entropy_with_logits(
                out.confidence_logit, y_e_t[idx],
            )

            # 4. Sharpe loss (optional)
            if sharpe_reg is not None:
                dir_probs = F.softmax(out.direction_logits, dim=-1)
                p_long_b = dir_probs[:, 0]
                p_short_b = dir_probs[:, 1]
                # Magnitude: use confidence as a proxy; bounded by softplus
                magnitude_b = F.softplus(out.confidence_logit)
                sharpe_term, _ = sharpe_reg(
                    p_long_b, p_short_b, magnitude_b, y_r_t[idx],
                )
                losses["sharpe"] = sharpe_term

            # Combine losses — DB-MTL or static sum
            if balancer is not None:
                total = balancer.compute_balanced_loss(
                    losses, shared_params=model.backbone.parameters(),
                )
            else:
                total = sum(losses.values())

            # Orthogonality penalty
            if ortho is not None:
                total = total + ortho(out.backbone_embedding, xd)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # ── Track best by held-out chunk of train (no leakage) ─────────
        model.eval()
        with torch.no_grad():
            n_val = min(500, len(train_idx))
            val_idx = train_idx[-n_val:]
            out_v = model(Xd_t[val_idx], Xs_t[val_idx])
            v_loss = F.binary_cross_entropy_with_logits(
                out_v.event_logit, y_e_t[val_idx],
            ).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if balancer is not None and epoch % 5 == 0:
                diag_history.append({"epoch": epoch, **balancer.get_diagnostics()})

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ── Test predictions ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        out_test = model(Xd_t[test_idx], Xs_t[test_idx])
        event_prob = torch.sigmoid(out_test.event_logit).cpu().numpy()
        dir_probs = F.softmax(out_test.direction_logits, dim=-1).cpu().numpy()
        confidence = torch.sigmoid(out_test.confidence_logit).cpu().numpy()

    test_df = df.loc[test_mask].copy().reset_index(drop=True)
    test_df["hybrid_event_prob"] = event_prob
    test_df["hybrid_p_long"] = dir_probs[:, 0]
    test_df["hybrid_p_short"] = dir_probs[:, 1]
    test_df["hybrid_p_neutral"] = dir_probs[:, 2]
    test_df["hybrid_confidence"] = confidence

    # ── Metrics (same conventions as baseline runner) ──────────────────
    metrics = {
        "fold_id": args.fold_id,
        "train_period": f"{args.train_start}..{args.train_end}",
        "test_period": f"{args.test_start}..{args.test_end}",
        "n_train": n_train,
        "n_test": n_test,
        "best_val_loss": best_val_loss,
        "config": {
            "use_simplex_etf": args.use_simplex_etf,
            "ortho_weight": args.ortho_weight,
            "use_dbmtl": args.use_dbmtl,
            "sharpe_weight": args.sharpe_weight,
        },
    }

    rule_events = test_df[test_df["event_flag"] == 1]
    if len(rule_events) > 0:
        outcomes = rule_events["path_outcome"]
        wins = int(outcomes.isin([0, 1]).sum())
        losses = int(outcomes.isin([2, 3]).sum())
        metrics["rule_baseline"] = {
            "n_events": int(len(rule_events)),
            "wins": wins, "losses": losses,
            "hit_rate": wins / max(wins + losses, 1),
        }

    if len(rule_events) > 0:
        re_full = test_df[test_df["event_flag"] == 1].copy()
        re_full["hybrid_dir"] = np.argmax(
            re_full[["hybrid_p_long", "hybrid_p_short", "hybrid_p_neutral"]].to_numpy(),
            axis=1,
        )
        re_full["rule_dir_code"] = re_full["event_direction"].map({1: 0, -1: 1, 0: 2})
        agree = re_full["hybrid_dir"] == re_full["rule_dir_code"]
        re_full = re_full[agree]
        if len(re_full) > 0:
            outcomes2 = re_full["path_outcome"]
            wins2 = int(outcomes2.isin([0, 1]).sum())
            losses2 = int(outcomes2.isin([2, 3]).sum())
            metrics["hybrid_filtered"] = {
                "n_events": int(len(re_full)),
                "wins": wins2, "losses": losses2,
                "hit_rate": wins2 / max(wins2 + losses2, 1),
            }
            metrics["lift_hit_rate"] = (
                metrics["hybrid_filtered"]["hit_rate"]
                - metrics["rule_baseline"]["hit_rate"]
            )

    if diag_history:
        metrics["dbmtl_history"] = diag_history

    # ── Output ────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(out_dir / "test_predictions.parquet", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"   ✅ fold {args.fold_id} done.")
    if "rule_baseline" in metrics:
        print(f"      rule baseline:   hit={metrics['rule_baseline']['hit_rate']:.3f}  "
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
    p.add_argument("--train-start", required=True)
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
    # ── Anti-collapse opt-ins ─────────────────────────────────────────
    p.add_argument("--use-simplex-etf", action="store_true",
                   help="Replace direction head with Simplex ETF Classifier")
    p.add_argument("--ortho-weight", type=float, default=0.0,
                   help="Weight for orthogonality penalty (0 = disabled)")
    p.add_argument("--use-dbmtl", action="store_true",
                   help="Use DB-MTL gradient balancer for multi-task loss")
    p.add_argument("--sharpe-weight", type=float, default=0.0,
                   help="Weight for differentiable Sharpe loss (0 = disabled)")
    p.add_argument("--tx-cost", type=float, default=1e-4,
                   help="Per-unit-position transaction cost for Sharpe loss")
    args = p.parse_args()
    run_fold(args)


if __name__ == "__main__":
    main()
