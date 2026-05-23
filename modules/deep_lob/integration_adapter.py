"""
modules/deep_lob/integration_adapter.py
─────────────────────────────────────────
Integration adapters connecting HierarchicalLOBTransformer to the existing system.

Two adapters:
    1. DeepLOBCNNAdapter — drop-in replacement لـ modules.deeplob_cnn.DeepLOBCNN
       Same interface (visual embedding 8-dim), backed by Transformer.

    2. BridgeAdapter — converts Transformer outputs to IntegrationBridge DL proba
       (used by paper_v19/live_predictor when --use-bridge is set)

Design: zero changes to existing code. Adapters wrap the Transformer to satisfy
existing APIs.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from .config import DeepLOBConfig
from .data_structures import BarLOB
from .hierarchical_model import HierarchicalLOBTransformer


class DeepLOBCNNAdapter:
    """Drop-in replacement for `modules.deeplob_cnn.DeepLOBCNN`.

    Backs the 8-dim visual embedding interface with the Transformer instead of CNN.

    Usage (zero change to train_v19.py Stage 2):
        # Before:
        from modules.deeplob_cnn import DeepLOBCNN
        cnn = DeepLOBCNN(channels=7)
        emb = cnn.predict(lob_tensor)

        # After:
        from modules.deep_lob.integration_adapter import DeepLOBCNNAdapter
        cnn = DeepLOBCNNAdapter(channels=7)
        emb = cnn.predict(lob_tensor)
    """

    def __init__(
        self,
        time_steps: int = 50,
        price_levels: int = 20,
        channels: int = 7,
        emb_dim: int = 8,
        brain_file: str = "outputs/hierarchical_lob_transformer.pt",
        config: Optional[DeepLOBConfig] = None,
        device: str = "auto",
    ):
        self.T = time_steps
        self.P = price_levels
        self.C = channels
        self.emb_dim = emb_dim
        self.brain_file = brain_file

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load or initialize
        import os
        if os.path.exists(brain_file):
            try:
                self.model = HierarchicalLOBTransformer.from_checkpoint(
                    brain_file, map_location=str(self.device),
                )
                self._fitted = True
                print(f"[Transformer] loaded from {brain_file}")
            except Exception as exc:
                print(f"[Transformer] failed to load ({exc}) — initializing new")
                self.model = HierarchicalLOBTransformer(config or DeepLOBConfig())
                self._fitted = False
        else:
            print("[Transformer] no checkpoint — initializing new")
            self.model = HierarchicalLOBTransformer(config or DeepLOBConfig())
            self._fitted = False

        self.model.to(self.device)

    def predict(self, lob_tensor: np.ndarray) -> np.ndarray:
        """Predict 8-dim visual embedding from LOB tensor.

        Parameters
        ----------
        lob_tensor : (N, T, P, C) numpy — N samples, T time, P levels, C channels

        Returns
        -------
        embeddings : (N, 8) numpy
        """
        if not isinstance(lob_tensor, np.ndarray):
            lob_tensor = np.asarray(lob_tensor)
        if lob_tensor.ndim == 3:
            lob_tensor = lob_tensor[np.newaxis]  # add batch dim

        N = lob_tensor.shape[0]

        # NOTE: This adapter requires order-level input. For LOB-tensor-only mode,
        # we use a degenerate representation: treat each (T, P) snapshot as a bar
        # with fake order features. This loses information but preserves the API.
        order_features, order_masks, bar_mask = self._lob_to_orders(lob_tensor)
        context = torch.zeros(N, 138, device=self.device)  # no context

        with torch.no_grad():
            embeddings = self.model.get_visual_embedding(
                order_features.to(self.device),
                order_masks.to(self.device),
                bar_mask.to(self.device),
                context,
            )
        return embeddings.cpu().numpy()

    def _lob_to_orders(
        self, lob_tensor: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert classic LOB tensor (N, T, P, C) to pseudo-order representation.

        Each price level → one synthetic "order" per timestep.
        Lossy but preserves the API. For full benefit, use BarLOB-based input.
        """
        N, T, P, C = lob_tensor.shape
        n_orders_per_bar = P  # treat each level as one order

        # Convert: (N, T, P, C) → (N, T, P, 7)
        # Map channels to order features (approximate):
        #   channel 0 (depth) → log_size
        #   channel 1 (buy_fp) → side=0
        #   channel 2 (sell_fp) → side=1
        order_features = np.zeros((N, T, n_orders_per_bar, 7), dtype=np.float32)
        for n in range(N):
            for t in range(T):
                for p in range(P):
                    side = 0 if p < P // 2 else 1  # bid vs ask
                    depth_val = float(lob_tensor[n, t, p, 0]) if C > 0 else 0.0
                    log_size = float(np.log1p(abs(depth_val) + 1e-6))
                    price_dist_ticks = float(p - P // 2)
                    time_off_ms = float(t * 1000)  # 1s per step (placeholder)
                    order_features[n, t, p] = [
                        side, 0, log_size, price_dist_ticks, time_off_ms, 0, 0
                    ]

        order_features_t = torch.from_numpy(order_features)
        order_masks = torch.ones(N, T, n_orders_per_bar, dtype=torch.bool)
        bar_mask = torch.ones(N, T, dtype=torch.bool)
        return order_features_t, order_masks, bar_mask


class BridgeAdapter:
    """Adapter للـ IntegrationBridge: Transformer outputs → DL proba dict.

    Bridge expects: dl_proba = np.array([p_long, p_short, p_neutral])
    Transformer outputs: direction_logits → softmax → these probs

    Usage:
        adapter = BridgeAdapter(model)
        dl_proba = adapter.predict_proba(bar_data)
        decision = bridge.evaluate_row(row, dl_proba)
    """

    def __init__(self, model: HierarchicalLOBTransformer, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    def predict_proba(
        self,
        bar: BarLOB,
        max_orders: int = 500,
    ) -> np.ndarray:
        """Single-bar prediction → 3-dim DL proba.

        Returns
        -------
        np.ndarray shape (3,): [p_long, p_short, p_neutral]
        """
        features = bar.to_features(max_orders=max_orders)
        # Add batch and time dims
        order_features = torch.from_numpy(features["order_features"]).unsqueeze(0).unsqueeze(0)
        order_masks = torch.from_numpy(features["order_mask"]).unsqueeze(0).unsqueeze(0)
        bar_mask = torch.ones(1, 1, dtype=torch.bool)
        if "context" in features:
            context = torch.from_numpy(features["context"]).unsqueeze(0).float()
            # Pad/truncate to 138
            n_ctx = context.shape[1]
            if n_ctx < 138:
                context = torch.cat([
                    context, torch.zeros(1, 138 - n_ctx)
                ], dim=1)
            elif n_ctx > 138:
                context = context[:, :138]
        else:
            context = torch.zeros(1, 138)

        with torch.no_grad():
            out = self.model(
                order_features.to(self.device),
                order_masks.to(self.device),
                bar_mask.to(self.device),
                context.to(self.device),
            )
            probs = out.direction_probs().cpu().numpy()[0]  # (3,)

        # Map [LONG, SHORT, NEUTRAL] order
        return probs

    def predict_with_targets(self, bar: BarLOB) -> dict[str, Any]:
        """Full multi-task prediction للـ Bridge decision context."""
        features = bar.to_features(max_orders=500)
        order_features = torch.from_numpy(features["order_features"]).unsqueeze(0).unsqueeze(0)
        order_masks = torch.from_numpy(features["order_mask"]).unsqueeze(0).unsqueeze(0)
        bar_mask = torch.ones(1, 1, dtype=torch.bool)
        context = (
            torch.from_numpy(features["context"]).unsqueeze(0).float()
            if "context" in features else torch.zeros(1, 138)
        )
        if context.shape[1] != 138:
            if context.shape[1] < 138:
                context = torch.cat([context, torch.zeros(1, 138 - context.shape[1])], dim=1)
            else:
                context = context[:, :138]

        with torch.no_grad():
            out = self.model(
                order_features.to(self.device),
                order_masks.to(self.device),
                bar_mask.to(self.device),
                context.to(self.device),
            )

        return {
            "dl_proba": out.direction_probs().cpu().numpy()[0],
            "next_price": float(out.next_price.cpu().item()),
            "next_imbalance": float(out.next_imbalance.cpu().item()),
            "next_volatility": float(out.next_volatility.cpu().item()),
            "next_regime_probs": out.regime_probs().cpu().numpy()[0],
            "wall_persist": float(out.wall_persist.cpu().item()),
            "time_to_event": float(out.time_to_event.cpu().item()),
        }


def compute_dynamic_tp_sl(
    transformer_predictions: dict[str, Any],
    base_tp_mult: float = 2.0,
    base_sl_mult: float = 1.0,
    wall_caution_factor: float = 0.5,
    volatility_scale: float = 1.0,
) -> tuple[float, float]:
    """Compute dynamic TP/SL multipliers based on Transformer outputs.

    Logic (التقرير ④):
        - High predicted volatility → wider TP/SL
        - Strong wall ahead → cap TP near wall
        - Low time_to_event → tighter stops

    Parameters
    ----------
    transformer_predictions : dict from BridgeAdapter.predict_with_targets
    base_tp_mult : starting TP multiplier (from regime)
    base_sl_mult : starting SL multiplier

    Returns
    -------
    (tp_mult, sl_mult)
    """
    vol = transformer_predictions["next_volatility"]
    wall_persist = transformer_predictions["wall_persist"]
    time_to_event = transformer_predictions["time_to_event"]

    # Volatility scaling
    vol_factor = float(np.clip(vol / (vol + 0.5), 0.5, 2.0))
    tp_mult = base_tp_mult * vol_factor * volatility_scale
    sl_mult = base_sl_mult * vol_factor

    # Wall caution: short wall life → tighten TP
    if wall_persist > 5.0:  # wall lives long → reduce TP
        tp_mult *= (1.0 - wall_caution_factor * 0.5)

    # Time-to-event caution
    if time_to_event < 3.0:  # event coming soon → tighten everything
        tp_mult *= 0.8
        sl_mult *= 1.2

    return float(tp_mult), float(sl_mult)
