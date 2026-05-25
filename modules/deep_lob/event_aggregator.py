"""
modules/deep_lob/event_aggregator.py
─────────────────────────────────────
Event-level aggregation: clusters orders into discoverable events.

Soft clustering via attention-based pooling:
    Each event type (sweep, absorb, spoof, iceberg, ...) is a learnable query.
    Orders attend to each query → soft assignment + weighted pooling.

This is the layer where the model discovers "what kinds of events exist".
Post-training, we can inspect which orders contribute to each event type.

Mathematical formulation:
    Q_e ∈ R^D for each event type e ∈ {1..E}
    α(e, i) = softmax over orders i of <Q_e, K_i>
    Event_e = Σ_i α(e, i) * V_i

This is essentially "queries against keys", inspired by:
    - Set Transformer (Lee et al. 2019)
    - Perceiver (Jaegle et al. 2021) - cross-attention pooling
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EventAggregatorConfig


class EventAggregator(nn.Module):
    """Soft-clustering aggregator: orders → events.

    Architecture:
        - N learnable event queries (Q_e)
        - Cross-attention: events attend to orders
        - Output: E event embeddings per bar
    """

    def __init__(self, config: EventAggregatorConfig, order_embed_dim: int):
        super().__init__()
        self.config = config
        self.order_embed_dim = order_embed_dim
        E = config.n_event_types
        D_event = config.event_dim
        D_order = order_embed_dim

        # Learnable event queries
        self.event_queries = nn.Parameter(torch.randn(E, D_event) * 0.02)

        # Project orders to event dim (for K, V)
        self.key_proj = nn.Linear(D_order, D_event)
        self.value_proj = nn.Linear(D_order, D_event)

        # Scaling
        self.scale = D_event ** -0.5

        # Temperature للـ softmax (controls sharpness)
        self.log_temperature = nn.Parameter(
            torch.tensor(0.0)  # exp(0) = 1.0
        )

        # Final projection per event
        self.event_proj = nn.Sequential(
            nn.LayerNorm(D_event),
            nn.Linear(D_event, D_event),
            nn.GELU(),
        )

        # Attention weights cache for pattern discovery
        self._last_assignment: Optional[torch.Tensor] = None

    def forward(
        self,
        order_embeddings: torch.Tensor,
        order_mask: Optional[torch.Tensor] = None,
        return_assignment: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        order_embeddings : (B, N, D_order)
        order_mask : (B, N) bool — True for valid

        Returns
        -------
        events : (B, E, D_event) — E event embeddings per bar
        """
        B, N, D = order_embeddings.shape
        E = self.config.n_event_types

        # K, V from orders
        keys = self.key_proj(order_embeddings)        # (B, N, D_event)
        values = self.value_proj(order_embeddings)    # (B, N, D_event)

        # Q = learnable event queries (expanded to batch)
        queries = self.event_queries.unsqueeze(0).expand(B, -1, -1)  # (B, E, D_event)

        # Cross-attention: events ↔ orders
        # logits: (B, E, N)
        # Clamp log_temperature ∈ [-3, 3] → temperature ∈ [0.05, 20] so
        # exp() drift cannot collapse attention to one-hot (temp→0) or
        # uniform garbage (temp→∞).
        log_temp = self.log_temperature.clamp(min=-3.0, max=3.0)
        temperature = torch.exp(log_temp)
        logits = (queries @ keys.transpose(-2, -1)) * self.scale / temperature

        # Mask invalid orders
        if order_mask is not None:
            # Bug fix: لو bar مفيهاش أي order، order_mask = all False
            # → softmax(all -inf) = NaN
            # Solution: rows الفاضية، عاملها كأنها كلها valid (events للـ bars دي
            # هتتصفر downstream بـ bar_mask في LSTM)
            row_has_valid = order_mask.any(dim=-1, keepdim=True)  # (B, 1)
            safe_mask = order_mask | (~row_has_valid)             # (B, N)
            invalid = ~safe_mask
            invalid_expanded = invalid.unsqueeze(1).expand(-1, E, -1)
            logits = logits.masked_fill(invalid_expanded, float("-inf"))

        # Soft assignment
        assignment = F.softmax(logits, dim=-1)  # (B, E, N)

        if return_assignment:
            self._last_assignment = assignment.detach()

        # Weighted pooling
        events = assignment @ values  # (B, E, D_event)

        # Final projection
        events = self.event_proj(events)

        return events

    def get_last_assignment(self) -> Optional[torch.Tensor]:
        """Returns last soft-assignment matrix (B, E, N).

        Use to identify which orders contribute to which event types
        (pattern discovery).
        """
        return self._last_assignment

    @property
    def n_event_types(self) -> int:
        return self.config.n_event_types

    @property
    def event_dim(self) -> int:
        return self.config.event_dim
