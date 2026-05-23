"""
modules/deep_lob/bar_lstm.py
─────────────────────────────
Bar-level sequence modeling.

Aggregates events across bars → narrative understanding.
Each bar contributes an event sequence; LSTM tracks temporal dynamics.

Causal architecture (unidirectional LSTM) للـ real-time inference.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import BarLSTMConfig


class BarLevelLSTM(nn.Module):
    """LSTM aggregator على bar sequences.

    Two-stage:
        1. Event pooling per bar: events (E, D) → bar embedding (D)
        2. LSTM across bar sequence: (T, D) → (T, H) → final hidden state
    """

    def __init__(self, config: BarLSTMConfig):
        super().__init__()
        self.config = config

        # Per-bar pooling: combine E events into single bar representation
        # Use attention pooling for adaptive weighting
        self.event_pool_query = nn.Parameter(torch.randn(1, config.input_dim) * 0.02)
        self.event_pool_key = nn.Linear(config.input_dim, config.input_dim)
        self.event_pool_value = nn.Linear(config.input_dim, config.input_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )

        # Output dim depends on bidirectional
        self.output_dim = config.hidden_dim * (2 if config.bidirectional else 1)

        # Final layer norm
        self.ln = nn.LayerNorm(self.output_dim)

    def _pool_events(
        self,
        events: torch.Tensor,
    ) -> torch.Tensor:
        """Attention-pool events into single bar embedding.

        Parameters
        ----------
        events : (B, E, D)

        Returns
        -------
        bar_emb : (B, D)
        """
        B = events.size(0)
        # query (1, D) → expand to batch
        q = self.event_pool_query.expand(B, -1).unsqueeze(1)  # (B, 1, D)
        k = self.event_pool_key(events)                        # (B, E, D)
        v = self.event_pool_value(events)                      # (B, E, D)

        scale = events.size(-1) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale               # (B, 1, E)
        attn = torch.softmax(attn, dim=-1)
        pooled = (attn @ v).squeeze(1)                         # (B, D)
        return pooled

    def forward(
        self,
        bar_events_sequence: torch.Tensor,
        bar_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        bar_events_sequence : (B, T, E, D)
            B = batch, T = time steps (bars), E = events per bar, D = event_dim
        bar_mask : (B, T) bool — True for valid bars

        Returns
        -------
        sequence_output : (B, T, output_dim) — LSTM outputs at each step
        final_state : (B, output_dim) — last hidden state (last valid bar)
        """
        B, T, E, D = bar_events_sequence.shape

        # Pool events per bar: (B, T, E, D) → (B, T, D)
        bars_flat = bar_events_sequence.reshape(B * T, E, D)
        bars_pooled = self._pool_events(bars_flat)        # (B*T, D)
        bars_pooled = bars_pooled.reshape(B, T, D)

        # LSTM
        sequence_output, (h_n, c_n) = self.lstm(bars_pooled)
        sequence_output = self.ln(sequence_output)

        # Get final state per batch element (respect bar_mask)
        if bar_mask is not None:
            # Find last valid index per batch
            valid_counts = bar_mask.sum(dim=1)            # (B,)
            last_valid_idx = (valid_counts - 1).clamp(min=0)
            # Gather
            batch_idx = torch.arange(B, device=sequence_output.device)
            final_state = sequence_output[batch_idx, last_valid_idx]  # (B, output_dim)
        else:
            final_state = sequence_output[:, -1, :]

        return sequence_output, final_state

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
