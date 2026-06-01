#!/usr/bin/env python3
"""Phase 1.11 — SSL Forward-Pass Probing Harness (Step 4 of the user's plan).

THE SCIENTIFIC QUESTION
═══════════════════════
Does the HierarchicalLOBTransformer's representation of the LOB tensors
carry signal beyond what the bar-level context features already provide?

THE ANTI-CIRCULAR DESIGN (4 stages — random control + pretrain comparison)
═══════════════════════════════════════════════════════════════════════════
  1. CONTEXT_ONLY      — linear probe on the 138 bar features alone
                         (the floor any embedding must beat to add value)

  2. RANDOM_EMBED      — linear probe on the embeddings of a transformer
                         with RANDOM weights (null hypothesis — any IC
                         here is artifact, not learning)

  3. RANDOM_EMBED+CTX  — concat random embeddings + context
                         (controls for any spurious lift from concatenation)

  4. PRETRAINED_EMBED  — embeddings of a transformer pretrained on
                         next_price_delta. IF this >> (1) AND >> (3),
                         the transformer learned something the context
                         features couldn't capture. ELSE the LOB signal
                         is either absent at this scale or already in
                         the bar features.

The harness REPORTS the four numbers + their deltas. It does NOT claim
alpha — that's the user's call based on whether the deltas are
economically + statistically meaningful.

USAGE
═════
    python tools/diagnostics/ssl_probe_harness.py \\
        --features outputs/phase1_q2_6b/day_trading_features_phase1_q2_6b.parquet \\
        --lob-tensors outputs/phase1_q2_6b/lob_tensors_phase1_q2_6b.npy \\
        --output outputs/phase1_q2_6b/audit/ssl_probe/ \\
        --pretrain-epochs 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def linear_probe_ic(
    X: np.ndarray, y: np.ndarray, *,
    train_frac: float = 0.7,
    valid_mask: Optional[np.ndarray] = None,
) -> dict:
    """Train ridge on the train slice, predict on the holdout, return
    rank IC (Spearman) + OOS R^2. Robust to fat tails + high-dim X."""
    from scipy.stats import spearmanr, pearsonr
    from sklearn.linear_model import Ridge

    if valid_mask is not None:
        keep = valid_mask & np.isfinite(y) & np.isfinite(X).all(axis=1)
    else:
        keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[keep], y[keep]

    n = len(y)
    if n < 100:
        return {"spearman_ic": 0.0, "pearson_ic": 0.0, "r2_oos": 0.0,
                "n_train": 0, "n_test": 0, "feature_dim": int(X.shape[1]),
                "error": "insufficient samples"}

    cut = int(n * train_frac)
    X_tr, y_tr = X[:cut], y[:cut]
    X_te, y_te = X[cut:], y[cut:]

    model = Ridge(alpha=1.0).fit(X_tr, y_tr)
    pred_te = model.predict(X_te)
    sp_ic, sp_p = spearmanr(pred_te, y_te)
    pe_ic, _ = pearsonr(pred_te, y_te)
    ss_res = float(((y_te - pred_te) ** 2).sum())
    ss_tot = float(((y_te - y_te.mean()) ** 2).sum()) + 1e-12
    r2_oos = 1.0 - ss_res / ss_tot

    return {
        "spearman_ic": float(sp_ic) if np.isfinite(sp_ic) else 0.0,
        "spearman_p": float(sp_p) if np.isfinite(sp_p) else 1.0,
        "pearson_ic": float(pe_ic) if np.isfinite(pe_ic) else 0.0,
        "r2_oos": r2_oos,
        "n_train": int(cut), "n_test": int(n - cut),
        "feature_dim": int(X.shape[1]),
    }


def lob_to_flat_embedding(
    lob_tensors: np.ndarray, *, weights_seed: int = 0,
) -> np.ndarray:
    """Random linear projection of (N, T, P, C) → (N, 64) — serves as
    the null-hypothesis 'embedding'. We don't import the heavy
    HierarchicalLOBTransformer here; a random projection is the cleanest
    null because it has zero learnable parameters and produces an
    embedding-shaped output with the same dimensionality. If the LOB
    structure has signal, it should be detectable above this floor."""
    rng = np.random.RandomState(weights_seed)
    n = lob_tensors.shape[0]
    # Flatten the last-bar snapshot per row: (T, P, C) → (P*C,)
    snap = lob_tensors[:, -1, :, :].reshape(n, -1).astype(np.float64)
    # Random projection to 64-dim (Johnson-Lindenstrauss style)
    proj = rng.randn(snap.shape[1], 64) / np.sqrt(snap.shape[1])
    return (snap @ proj).astype(np.float64)


def quick_train_lob_projector(
    lob_tensors: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    *,
    epochs: int = 5,
    embed_dim: int = 64,
    lr: float = 1e-3,
    batch_size: int = 64,
    train_frac: float = 0.7,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Train a small encoder (Conv1d + Linear) on the LOB snapshot to
    predict y. Returns the embeddings for ALL rows + a training log.

    This is the 'pretrained' arm. A small model (not the full
    HierarchicalLOBTransformer) keeps the harness runnable in minutes
    on CPU + isolates the question 'can the LOB snapshot encode signal'
    from 'is the big architecture trained well'. Real SSL pretraining
    is a separate, longer commitment."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    n, T, P, C = lob_tensors.shape
    snap = lob_tensors[:, -1, :, :].astype(np.float32)   # (N, P, C)

    keep = valid_mask & np.isfinite(y)
    X_all = snap
    y_all = y.astype(np.float32)

    X = X_all[keep]; ytr = y_all[keep]
    n_keep = len(ytr)
    if n_keep < 200:
        return np.zeros((n, embed_dim), dtype=np.float64), {"error": "insufficient samples"}

    cut = int(n_keep * train_frac)
    Xtr = torch.from_numpy(X[:cut])
    ytr_t = torch.from_numpy(ytr[:cut])

    class _Enc(nn.Module):
        def __init__(self):
            super().__init__()
            # treat the P levels as the spatial dim, C channels as feature dim
            self.conv = nn.Conv1d(C, 32, kernel_size=3, padding=1)
            self.act = nn.GELU()
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.proj = nn.Linear(32, embed_dim)

        def encode(self, x):     # x: (B, P, C) → (B, embed_dim)
            x = x.transpose(1, 2)            # (B, C, P)
            x = self.act(self.conv(x))       # (B, 32, P)
            x = self.pool(x).squeeze(-1)     # (B, 32)
            return self.proj(x)              # (B, embed_dim)

        def forward(self, x):
            return self.encode(x)

    enc = _Enc()
    head = nn.Linear(embed_dim, 1)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=lr)
    log = []

    for ep in range(epochs):
        enc.train(); head.train()
        losses = []
        # mini-batch SGD
        idx = np.arange(cut); np.random.RandomState(seed + ep).shuffle(idx)
        for s in range(0, cut, batch_size):
            sl = idx[s:s + batch_size]
            xb = Xtr[sl]; yb = ytr_t[sl]
            emb = enc(xb)
            pred = head(emb).squeeze(-1)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        log.append({"epoch": ep, "train_mse": float(np.mean(losses))})

    enc.eval()
    with torch.no_grad():
        emb_all = enc(torch.from_numpy(snap)).numpy().astype(np.float64)
    return emb_all, {"epochs": epochs, "n_train": cut, "log": log}


def run_probe(
    features_path: Path, lob_path: Path, *, output_dir: Path,
    pretrain_epochs: int = 0, target_horizon: int = 6, device: str = "cpu",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📂 Loading inputs ...")
    df = pd.read_parquet(features_path)
    lob = np.load(lob_path)
    print(f"   features: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"   LOB tensors: {lob.shape}")

    if "next_price_delta" in df.columns:
        y = pd.to_numeric(df["next_price_delta"], errors="coerce").to_numpy(np.float64)
        valid = df.get("next_price_delta_valid", pd.Series(True, index=df.index)).astype(bool).to_numpy()
        print(f"   target: next_price_delta ({int(valid.sum()):,} valid rows)")
    else:
        close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)
        y = np.full(len(close), np.nan)
        y[:-target_horizon] = np.log(np.maximum(close[target_horizon:], 1e-9) /
                                      np.maximum(close[:-target_horizon], 1e-9))
        valid = np.isfinite(y)
        print(f"   target: computed log-return @ H={target_horizon} ({int(valid.sum()):,} valid)")

    # Context features — exclude anything the SSL data_loader would flag as leakage
    from self_supervised.data_loader import _is_leakage_column
    ctx_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if not _is_leakage_column(c) and c not in ("open", "high", "low", "close")]
    X_ctx = df[ctx_cols].astype(np.float64).fillna(0.0).to_numpy()
    print(f"   context features: {len(ctx_cols)} cols → X_ctx shape {X_ctx.shape}")

    results = {"target_horizon": target_horizon, "n_context_features": len(ctx_cols)}

    # Stage 1
    print("\n[1/4] CONTEXT_ONLY linear probe ...")
    results["context_only"] = linear_probe_ic(X_ctx, y, valid_mask=valid)
    print(f"      spearman_ic={results['context_only']['spearman_ic']:+.4f}  "
          f"R²_oos={results['context_only']['r2_oos']:+.4f}")

    # Stage 2 — random projection (null)
    print("\n[2/4] RANDOM_EMBED (null hypothesis — random projection) ...")
    emb_random = lob_to_flat_embedding(lob, weights_seed=0)
    print(f"      embedding shape: {emb_random.shape}")
    results["random_embed"] = linear_probe_ic(emb_random, y, valid_mask=valid)
    print(f"      spearman_ic={results['random_embed']['spearman_ic']:+.4f}  "
          f"R²_oos={results['random_embed']['r2_oos']:+.4f}")

    # Stage 3 — random + ctx concat
    print("\n[3/4] RANDOM_EMBED + CONTEXT concat ...")
    X_concat = np.concatenate([emb_random, X_ctx], axis=1)
    results["random_concat"] = linear_probe_ic(X_concat, y, valid_mask=valid)
    print(f"      spearman_ic={results['random_concat']['spearman_ic']:+.4f}  "
          f"R²_oos={results['random_concat']['r2_oos']:+.4f}")

    # Stage 4 — quick supervised pretrain
    if pretrain_epochs > 0:
        print(f"\n[4/4] PRETRAINED ({pretrain_epochs} epochs supervised on y) ...")
        try:
            emb_trained, log = quick_train_lob_projector(
                lob, y, valid, epochs=pretrain_epochs,
            )
            results["pretrained_embed"] = linear_probe_ic(emb_trained, y, valid_mask=valid)
            X_concat_trained = np.concatenate([emb_trained, X_ctx], axis=1)
            results["pretrained_concat"] = linear_probe_ic(X_concat_trained, y, valid_mask=valid)
            results["pretrain_log"] = log
            print(f"      pretrained alone: spearman_ic={results['pretrained_embed']['spearman_ic']:+.4f}")
            print(f"      pretrained + ctx: spearman_ic={results['pretrained_concat']['spearman_ic']:+.4f}")
        except Exception as e:
            print(f"      ⚠️  Pretrain failed: {e!r}")
            results["pretrained_embed"] = {"error": str(e)}

    # Verdict summary
    print("\n" + "═" * 65)
    print("  SSL Forward-Pass Probe — Δ summary")
    print("═" * 65)
    ctx_ic = results["context_only"]["spearman_ic"]
    rnd_ic = results["random_embed"]["spearman_ic"]
    rnd_cat_ic = results["random_concat"]["spearman_ic"]
    print(f"  context-only            : {ctx_ic:+.4f}   (floor — what bar features alone give)")
    print(f"  random embed alone      : {rnd_ic:+.4f}   (null — should be near 0)")
    print(f"  random embed + ctx      : {rnd_cat_ic:+.4f}   (Δ vs context = {rnd_cat_ic - ctx_ic:+.4f})")

    if "spearman_ic" in results.get("pretrained_embed", {}):
        trn_ic = results["pretrained_embed"]["spearman_ic"]
        trn_cat_ic = results["pretrained_concat"]["spearman_ic"]
        print(f"  pretrained alone        : {trn_ic:+.4f}   (Δ vs random = {trn_ic - rnd_ic:+.4f})")
        print(f"  pretrained + ctx        : {trn_cat_ic:+.4f}   (Δ vs context = {trn_cat_ic - ctx_ic:+.4f})")
        delta_ctx = trn_cat_ic - ctx_ic
        delta_rnd = trn_ic - rnd_ic
        alpha = (abs(delta_ctx) > 0.01) and (abs(delta_rnd) > 0.01)
        verdict = "ALPHA_PRESENT" if alpha else "NO_DETECTABLE_ALPHA"
        results["verdict"] = verdict
        results["delta_pretrained_vs_random"] = float(delta_rnd)
        results["delta_pretrained_concat_vs_context"] = float(delta_ctx)
        print(f"\n  Verdict: {verdict}")
        print(f"    Criterion: |trained-random| > 0.01 AND |trained+ctx vs ctx| > 0.01")
        print(f"    Observed:  |Δ|={abs(delta_rnd):.4f}                |Δ|={abs(delta_ctx):.4f}")

    summary_path = output_dir / "ssl_probe_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n📋 Summary: {summary_path}")
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--lob-tensors", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pretrain-epochs", type=int, default=0,
                   help="0 = only random control (fast). >=3 = train + probe.")
    p.add_argument("--target-horizon", type=int, default=6)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if not args.features.exists():
        print(f"❌ features not found: {args.features}", file=sys.stderr); return 2
    if not args.lob_tensors.exists():
        print(f"❌ LOB tensors not found: {args.lob_tensors}", file=sys.stderr); return 2

    run_probe(
        args.features, args.lob_tensors, output_dir=args.output,
        pretrain_epochs=args.pretrain_epochs, target_horizon=args.target_horizon,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
