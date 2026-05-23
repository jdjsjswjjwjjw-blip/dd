"""
modules/deep_lob/hierarchical_model.py
───────────────────────────────────────
Hierarchical LOB Transformer + Multi-Task Heads — the full model.

Architecture (top-down):

    Input:
        order_features  : (B, T, N, 7)   — T bars, N orders/bar, 7 features
        order_masks     : (B, T, N)      — valid order mask
        bar_mask        : (B, T)         — valid bar mask
        context         : (B, n_context) — 138 existing features

    Stage 1: Order Embedding
        (B, T, N, 7) → (B, T, N, D_order=32)

    Stage 2: Order-Level Transformer (per bar)
        Self-attention على orders داخل كل bar
        (B, T, N, D_order) → (B, T, N, D_order)

    Stage 3: Event Aggregation (per bar)
        Soft-cluster orders into E event types
        (B, T, N, D_order) → (B, T, E, D_event)

    Stage 4: Bar-Level LSTM
        Sequence model عبر الـ T bars
        (B, T, E, D_event) → (B, output_dim)

    Stage 5: Context Encoder
        (B, n_context) → (B, context_dim)

    Stage 6: Cross-Attention Fusion
        Book + Context → Shared Representation
        (B, output_dim) ⊗ (B, context_dim) → (B, shared_dim)

    Stage 7: Multi-Task Heads
        Shared Rep → 7 predictions

Total parameter count target: 2-10M (production scale, trainable on small GPU).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import DeepLOBConfig
from .order_embedder import OrderEmbedder
from .transformer_blocks import LOBTransformerEncoder
from .event_aggregator import EventAggregator
from .bar_lstm import BarLevelLSTM
from .context_encoder import ContextEncoder, CrossAttentionFusion
from .multi_task_heads import MultiTaskHeads, MultiTaskOutput


class HierarchicalLOBTransformer(nn.Module):
    """Hierarchical LOB Transformer with Multi-Task heads."""

    def __init__(self, config: DeepLOBConfig):
        super().__init__()
        self.config = config

        # Stage 1: Order embedder
        self.order_embedder = OrderEmbedder(config.order_embedder)

        # Stage 2: Order-level transformer
        self.order_transformer = LOBTransformerEncoder(config.transformer)

        # Stage 3: Event aggregator
        self.event_aggregator = EventAggregator(
            config.event_aggregator,
            order_embed_dim=config.transformer.embed_dim,
        )

        # Stage 4: Bar-level LSTM
        self.bar_lstm = BarLevelLSTM(config.bar_lstm)

        # Stage 5: Context encoder
        self.context_encoder = ContextEncoder(config.context_encoder)

        # Stage 6: Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            book_dim=self.bar_lstm.output_dim,
            context_dim=config.context_encoder.output_dim,
            output_dim=config.multi_task_heads.shared_dim,
            n_heads=4,
            dropout=0.1,
        )

        # Stage 7: Multi-task heads
        self.heads = MultiTaskHeads(config.multi_task_heads)

    def forward(
        self,
        order_features: torch.Tensor,
        order_masks: torch.Tensor,
        bar_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> MultiTaskOutput | tuple[MultiTaskOutput, dict]:
        """Full forward pass.

        Parameters
        ----------
        order_features : (B, T, N, 7)   — order features per bar
        order_masks    : (B, T, N) bool — valid order indicators
        bar_mask       : (B, T) bool    — valid bar indicators
        context        : (B, n_context) — 138 existing features

        Returns
        -------
        outputs : MultiTaskOutput
        intermediates : dict (optional, with debug tensors)
        """
        B, T, N, _ = order_features.shape

        # ── Stage 1: Order embeddings ──────────────────────────────────────
        # Reshape: (B, T, N, 7) → (B*T, N, 7)
        order_feat_flat = order_features.reshape(B * T, N, -1)
        order_mask_flat = order_masks.reshape(B * T, N)

        order_emb = self.order_embedder(order_feat_flat, order_mask_flat)  # (B*T, N, D)

        # ── Stage 2: Order-level Transformer ───────────────────────────────
        order_emb, attentions = self.order_transformer(
            order_emb, key_padding_mask=order_mask_flat,
            return_attentions=return_intermediates,
        )  # (B*T, N, D)

        # ── Stage 3: Event aggregation ─────────────────────────────────────
        events = self.event_aggregator(
            order_emb, order_mask=order_mask_flat,
            return_assignment=return_intermediates,
        )  # (B*T, E, D_event)

        # Reshape back: (B*T, E, D) → (B, T, E, D)
        E = self.event_aggregator.n_event_types
        D_event = self.event_aggregator.event_dim
        events = events.reshape(B, T, E, D_event)

        # ── Stage 4: Bar-level LSTM ────────────────────────────────────────
        if bar_mask is None:
            # Infer bar_mask: a bar is valid if it has at least one valid order
            bar_mask = order_masks.any(dim=-1)  # (B, T)

        bar_sequence_output, bar_final_state = self.bar_lstm(events, bar_mask)
        # bar_final_state : (B, output_dim_lstm)

        # ── Stage 5: Context encoding ──────────────────────────────────────
        if context is None:
            # Use zeros if context not provided (backward compat)
            context = torch.zeros(
                B, self.config.context_encoder.n_input_features,
                device=order_features.device,
                dtype=order_features.dtype,
            )
        context_emb = self.context_encoder(context)  # (B, context_dim)

        # ── Stage 6: Fusion ────────────────────────────────────────────────
        shared = self.fusion(bar_final_state, context_emb)  # (B, shared_dim)

        # ── Stage 7: Multi-task heads ──────────────────────────────────────
        outputs = self.heads(shared)

        if return_intermediates:
            intermediates = {
                "order_embeddings": order_emb.reshape(B, T, N, -1),
                "attentions": attentions,
                "event_assignment": self.event_aggregator.get_last_assignment(),
                "events": events,
                "bar_sequence": bar_sequence_output,
                "bar_final": bar_final_state,
                "context_embedding": context_emb,
                "shared": shared,
            }
            return outputs, intermediates

        return outputs

    def predict(
        self,
        order_features: torch.Tensor,
        order_masks: torch.Tensor,
        bar_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> dict:
        """Inference (eval mode + no_grad).

        Returns dict — production-ready output.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                order_features=order_features,
                order_masks=order_masks,
                bar_mask=bar_mask,
                context=context,
            )
        return outputs.to_dict()

    def get_visual_embedding(
        self,
        order_features: torch.Tensor,
        order_masks: torch.Tensor,
        bar_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Backward-compatible interface with DeepLOB CNN.

        Returns the 8-dim visual embedding that Stage 3 (MetaLearner LSTM) expects.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                order_features=order_features,
                order_masks=order_masks,
                bar_mask=bar_mask,
                context=context,
            )
        # Project shared embedding to 8-dim
        shared = outputs.shared_embedding  # (B, shared_dim)
        # Use direction logits + key features as 8-dim
        # P(LONG), P(SHORT), next_imbalance, next_volatility, etc.
        embed_8 = torch.cat([
            outputs.direction_probs(),                 # 3 dims
            outputs.next_imbalance.unsqueeze(-1),      # 1
            outputs.next_volatility.unsqueeze(-1),     # 1
            outputs.wall_persist.unsqueeze(-1),        # 1
            outputs.regime_probs()[:, :1],             # 1 (trending prob)
            outputs.time_to_event.unsqueeze(-1),       # 1
        ], dim=-1)
        # = 8-dim
        return embed_8

    def count_parameters(self) -> dict[str, int]:
        """Parameter count breakdown by component."""
        return {
            "order_embedder": sum(p.numel() for p in self.order_embedder.parameters()),
            "order_transformer": sum(p.numel() for p in self.order_transformer.parameters()),
            "event_aggregator": sum(p.numel() for p in self.event_aggregator.parameters()),
            "bar_lstm": self.bar_lstm.count_parameters(),
            "context_encoder": sum(p.numel() for p in self.context_encoder.parameters()),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "heads": self.heads.count_parameters(),
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def save_checkpoint(self, path: str, extra_state: Optional[dict] = None):
        """Save model + config + extra state."""
        from dataclasses import asdict
        state = {
            "model_state_dict": self.state_dict(),
            "config": asdict(self.config),
            "model_name": self.config.model_name,
            "version": self.config.version,
        }
        if extra_state:
            state.update(extra_state)
        torch.save(state, path)

    @classmethod
    def from_checkpoint(cls, path: str, map_location: str = "cpu") -> "HierarchicalLOBTransformer":
        """Load model from checkpoint."""
        from .config import (
            OrderEmbedderConfig, TransformerConfig, EventAggregatorConfig,
            BarLSTMConfig, ContextEncoderConfig, MultiTaskHeadsConfig, TrainingConfig,
        )

        state = torch.load(path, map_location=map_location, weights_only=False)
        config_data = state["config"]

        if isinstance(config_data, DeepLOBConfig):
            config = config_data
        else:
            # Reconstruct nested configs from dict
            config = DeepLOBConfig(
                order_embedder=OrderEmbedderConfig(**config_data["order_embedder"]),
                transformer=TransformerConfig(**config_data["transformer"]),
                event_aggregator=EventAggregatorConfig(**config_data["event_aggregator"]),
                bar_lstm=BarLSTMConfig(**config_data["bar_lstm"]),
                context_encoder=ContextEncoderConfig(**config_data["context_encoder"]),
                multi_task_heads=MultiTaskHeadsConfig(**config_data["multi_task_heads"]),
                training=TrainingConfig(**config_data["training"]),
                model_name=config_data.get("model_name", "loaded"),
                version=config_data.get("version", 1),
            )

        model = cls(config)
        model.load_state_dict(state["model_state_dict"])
        return model
