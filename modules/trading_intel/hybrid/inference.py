"""
modules/trading_intel/hybrid/inference.py
═══════════════════════════════════════════════════════════════════════════════
Inference helpers — split out so the live serving path is independent of
the training path. Use these in `integration_bridge` or any future live-
trading wiring to avoid the failure modes described in
docs/PIPELINE_ISSUES_AUDIT.md:

  • Issue 2 — Normalization drift
      apply_saved_scalers() loads the FROZEN mu/sigma stats from a
      checkpoint and applies transform-only. NEVER call fit() at
      inference time. Mismatch between training-time and inference-time
      normalization is the #1 cause of "great backtest, terrible live"
      failures.

  • Issue 3 — Latency
      load_hybrid_for_inference() returns a model in eval() mode with
      gradients off. Pair with NumPy/torch tensor inputs — never feed
      pandas DataFrames into the forward path.

Usage:

    from modules.trading_intel.hybrid.inference import (
        load_hybrid_for_inference, apply_saved_scalers,
    )

    # Load once at startup:
    model, scalers = load_hybrid_for_inference("best_hybrid.pt")

    # Per tick:
    daytrade_norm = apply_saved_scalers(daytrade_feats_raw, scalers['daytrade'])
    ssl_norm      = apply_saved_scalers(ssl_embedding_raw, scalers['ssl'])
    cnn_norm      = apply_saved_scalers(cnn_embedding_raw, scalers['cnn'])
    with torch.no_grad():
        out = model(daytrade_norm, ssl_norm, cnn_norm)

The function below DOES NOT do feature extraction — that's the upstream
preprocessing layer's job. It assumes the caller has already produced
the raw feature vectors in the expected order. The corresponding
training-time feature order is stored in checkpoint['feature_cols'].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from modules.trading_intel.hybrid.model import HybridConfig, HybridModel


@dataclass
class FrozenScaler:
    """A frozen mu/sigma scaler — never re-fit at inference time."""
    mu: np.ndarray
    sigma: np.ndarray

    @classmethod
    def from_checkpoint_dict(cls, d: dict) -> "FrozenScaler":
        return cls(
            mu=np.asarray(d["mu"], dtype=np.float32),
            sigma=np.asarray(d["sigma"], dtype=np.float32),
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply (X − mu) / sigma. NO fit, no recomputation."""
        if X.shape[-1] != self.mu.shape[0]:
            raise ValueError(
                f"Scaler expected {self.mu.shape[0]} features, "
                f"got {X.shape[-1]}. Check feature_cols alignment."
            )
        # Floor sigma to avoid divide-by-zero (matches training-time floor)
        sig = np.maximum(self.sigma, 1e-6)
        return ((X - self.mu) / sig).astype(np.float32)


def apply_saved_scalers(
    X: np.ndarray, scaler: FrozenScaler,
) -> np.ndarray:
    """Inference-time wrapper. Equivalent to scaler.transform(X)."""
    return scaler.transform(X)


def load_hybrid_for_inference(
    ckpt_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[HybridModel, dict[str, FrozenScaler]]:
    """Load a trained HybridModel + its frozen scalers.

    Returns (model_in_eval_mode, scalers_dict). The model has gradients
    disabled and is ready for forward-only inference.

    Raises:
        KeyError: if the checkpoint is missing scaler stats — i.e., the
        training run forgot to save them. Better to fail loudly here
        than silently apply no normalization at inference.
    """
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    cfg_dict = ckpt.get("config")
    if cfg_dict is None:
        raise KeyError("Checkpoint is missing 'config' — cannot rebuild HybridModel")
    # Strip non-HybridConfig keys (in case extra metadata was stored)
    cfg_keys = {f.name for f in HybridConfig.__dataclass_fields__.values()}
    cfg = HybridConfig(**{k: v for k, v in cfg_dict.items() if k in cfg_keys})

    model = HybridModel(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    scalers: dict[str, FrozenScaler] = {}
    for name, ckpt_key in (("daytrade", "norm_daytrade"),
                            ("ssl", "norm_ssl"),
                            ("cnn", "norm_cnn")):
        if ckpt_key in ckpt:
            scalers[name] = FrozenScaler.from_checkpoint_dict(ckpt[ckpt_key])

    if "norm_daytrade" not in ckpt or "norm_ssl" not in ckpt:
        raise KeyError(
            f"Checkpoint at {ckpt_path} is missing normalization stats "
            f"(norm_daytrade and/or norm_ssl). The training run was incomplete "
            f"OR the checkpoint is from a pre-fix version. Re-train with the "
            f"current train_hybrid.py."
        )

    return model, scalers


def predict_one(
    model: HybridModel,
    scalers: dict[str, FrozenScaler],
    daytrade_features: np.ndarray,    # (D_daytrade,) raw, unnormalized
    ssl_embedding: np.ndarray,        # (D_ssl,) raw
    cnn_embedding: Optional[np.ndarray] = None,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Single-row inference helper. Returns a dict with event_prob,
    p_long, p_short, p_neutral, confidence. NEVER pandas; pure NumPy/torch.

    For batch inference, call model() directly with stacked inputs and
    pre-applied scalers — this helper is for the single-tick live loop.
    """
    # Apply frozen scalers
    xd = scalers["daytrade"].transform(daytrade_features[None, :])
    xs = scalers["ssl"].transform(ssl_embedding[None, :])
    xc = (scalers["cnn"].transform(cnn_embedding[None, :])
          if cnn_embedding is not None and "cnn" in scalers
          else None)

    xd_t = torch.from_numpy(xd).to(device)
    xs_t = torch.from_numpy(xs).to(device)
    xc_t = torch.from_numpy(xc).to(device) if xc is not None else None

    with torch.no_grad():
        out = model(xd_t, xs_t, xc_t)
        event_prob = float(torch.sigmoid(out.event_logit)[0])
        dir_probs = torch.softmax(out.direction_logits, dim=-1)[0]
        confidence = float(torch.sigmoid(out.confidence_logit)[0])

    return {
        "event_prob": event_prob,
        "p_long": float(dir_probs[0]),
        "p_short": float(dir_probs[1]),
        "p_neutral": float(dir_probs[2]),
        "confidence": confidence,
    }
