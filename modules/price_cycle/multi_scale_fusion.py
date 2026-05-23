"""
modules/price_cycle/multi_scale_fusion.py
───────────────────────────────────────────
Multi-Scale Fusion — يربط النموذجين:
    - LOB Transformer  (micro: order book, ثوانٍ-دقائق)   [Sprint 17]
    - Price Cycle Model (macro: price cycles, ساعات-أيام)  [Sprint 18]

الفكرة العلمية:
    السوق له ديناميكا على مقاييس متعددة (Lo, "Adaptive Markets Hypothesis").
    قرار سليم يحتاج محاذاة الـ micro مع الـ macro:

        micro bullish + macro young trend   → strong LONG
        micro bullish + macro exhausted     → weak LONG (small size, tight TP)
        micro bullish + macro markdown      → SKIP (counter-cycle)

3 fusion methods:
    - cross_attention : كل مقياس يحضر إلى الآخر (الأقوى)
    - gating          : الـ macro يتحكم في وزن الـ micro
    - concat          : أبسط، concat + MLP

التصميم: model-agnostic — يقبل embeddings فقط، لا يستورد الـ models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MultiScaleFusionConfig


@dataclass
class FusionOutput:
    """Multi-scale fusion result."""

    decision_logits: torch.Tensor       # (B, 3) — LONG/SHORT/NEUTRAL
    fused_embedding: torch.Tensor       # (B, fusion_dim)
    micro_gate: torch.Tensor            # (B,) — كم النموذج اعتمد على micro
    macro_gate: torch.Tensor            # (B,) — كم اعتمد على macro
    alignment_score: torch.Tensor       # (B,) — توافق micro/macro [-1,+1]

    def decision_probs(self) -> torch.Tensor:
        return F.softmax(self.decision_logits, dim=-1)

    def to_dict(self) -> dict:
        return {
            "decision_probs": self.decision_probs(),
            "micro_gate": self.micro_gate,
            "macro_gate": self.macro_gate,
            "alignment_score": self.alignment_score,
        }


# ════════════════════════════════════════════════════════════════════════════
# Fusion modules
# ════════════════════════════════════════════════════════════════════════════


class CrossAttentionFusion(nn.Module):
    """Cross-attention: micro ↔ macro mutual attention."""

    def __init__(self, config: MultiScaleFusionConfig):
        super().__init__()
        D = config.fusion_dim
        # common dim that's divisible by n_heads
        if D % config.n_heads != 0:
            D = ((D // config.n_heads) + 1) * config.n_heads

        self.micro_proj = nn.Linear(config.micro_dim, D)
        self.macro_proj = nn.Linear(config.macro_dim, D)

        self.attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=config.n_heads,
            dropout=config.dropout, batch_first=True,
        )
        self.ln_micro = nn.LayerNorm(D)
        self.ln_macro = nn.LayerNorm(D)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self._common_dim = D

    def forward(
        self, micro_emb: torch.Tensor, macro_emb: torch.Tensor,
    ) -> torch.Tensor:
        """micro_emb (B, micro_dim), macro_emb (B, macro_dim) → fused (B, fusion_dim)."""
        m = self.ln_micro(self.micro_proj(micro_emb)).unsqueeze(1)   # (B,1,D)
        M = self.ln_macro(self.macro_proj(macro_emb)).unsqueeze(1)   # (B,1,D)

        # micro attends to macro
        micro_att, _ = self.attn(query=m, key=M, value=M)
        # macro attends to micro
        macro_att, _ = self.attn(query=M, key=m, value=m)

        fused = (m + M + micro_att + macro_att).squeeze(1) / 4.0
        return self.out_proj(fused)


class GatingFusion(nn.Module):
    """Gating: macro context modulates how much micro signal to trust."""

    def __init__(self, config: MultiScaleFusionConfig):
        super().__init__()
        D = config.fusion_dim
        self.micro_proj = nn.Linear(config.micro_dim, D)
        self.macro_proj = nn.Linear(config.macro_dim, D)

        # Gate: macro decides micro weight
        self.gate_net = nn.Sequential(
            nn.Linear(config.macro_dim, D),
            nn.GELU(),
            nn.Linear(D, 1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, micro_emb: torch.Tensor, macro_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.micro_proj(micro_emb)
        M = self.macro_proj(macro_emb)
        gate = self.gate_net(macro_emb)              # (B, 1)
        fused = gate * m + (1.0 - gate) * M
        return self.out_proj(fused), gate.squeeze(-1)


class ConcatFusion(nn.Module):
    """Simplest: concat + MLP."""

    def __init__(self, config: MultiScaleFusionConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.micro_dim + config.macro_dim, config.fusion_dim * 2),
            nn.LayerNorm(config.fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim * 2, config.fusion_dim),
        )

    def forward(
        self, micro_emb: torch.Tensor, macro_emb: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([micro_emb, macro_emb], dim=-1))


# ════════════════════════════════════════════════════════════════════════════
# Full Multi-Scale Fusion Model
# ════════════════════════════════════════════════════════════════════════════


class MultiScaleFusion(nn.Module):
    """Combines micro (LOB) + macro (Cycle) embeddings into a trading decision.

    Model-agnostic: takes EMBEDDINGS, not models. Works with:
        - HierarchicalLOBTransformer.shared_embedding  (micro)
        - PriceCycleModel cycle_embedding              (macro)
        - أو أي embeddings أخرى بنفس الأبعاد
    """

    def __init__(self, config: MultiScaleFusionConfig):
        super().__init__()
        self.config = config

        if config.fusion_method == "cross_attention":
            self.fusion = CrossAttentionFusion(config)
        elif config.fusion_method == "gating":
            self.fusion = GatingFusion(config)
        else:
            self.fusion = ConcatFusion(config)

        # Decision head
        self.decision_head = nn.Sequential(
            nn.LayerNorm(config.fusion_dim),
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim // 2, config.decision_n_classes),
        )

        # Alignment estimator: cosine sim between micro/macro
        self.micro_align_proj = nn.Linear(config.micro_dim, 32)
        self.macro_align_proj = nn.Linear(config.macro_dim, 32)

    def _alignment(
        self, micro_emb: torch.Tensor, macro_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity between projected micro/macro → [-1, +1]."""
        m = F.normalize(self.micro_align_proj(micro_emb), dim=-1)
        M = F.normalize(self.macro_align_proj(macro_emb), dim=-1)
        return (m * M).sum(dim=-1)

    def forward(
        self,
        micro_embedding: torch.Tensor,
        macro_embedding: torch.Tensor,
    ) -> FusionOutput:
        """
        Parameters
        ----------
        micro_embedding : (B, micro_dim) — من LOB Transformer
        macro_embedding : (B, macro_dim) — من Price Cycle Model

        Returns
        -------
        FusionOutput
        """
        B = micro_embedding.size(0)

        # Fusion
        if self.config.fusion_method == "gating":
            fused, gate = self.fusion(micro_embedding, macro_embedding)
            micro_gate = gate
            macro_gate = 1.0 - gate
        else:
            fused = self.fusion(micro_embedding, macro_embedding)
            # symmetric methods: gates = 0.5
            micro_gate = torch.full((B,), 0.5, device=fused.device)
            macro_gate = torch.full((B,), 0.5, device=fused.device)

        # Decision
        decision_logits = self.decision_head(fused)

        # Alignment
        alignment = self._alignment(micro_embedding, macro_embedding)

        return FusionOutput(
            decision_logits=decision_logits,
            fused_embedding=fused,
            micro_gate=micro_gate,
            macro_gate=macro_gate,
            alignment_score=alignment,
        )

    def compute_loss(
        self, output: FusionOutput, target_direction: torch.Tensor,
        alignment_target: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """Decision loss + optional alignment auxiliary loss."""
        l_decision = F.cross_entropy(
            output.decision_logits, target_direction.long(), reduction=reduction,
        )
        result = {"decision": l_decision, "total": l_decision}

        if alignment_target is not None:
            l_align = F.mse_loss(output.alignment_score, alignment_target, reduction=reduction)
            result["alignment"] = l_align
            result["total"] = l_decision + 0.2 * l_align

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ════════════════════════════════════════════════════════════════════════════
# End-to-end orchestrator (model-agnostic)
# ════════════════════════════════════════════════════════════════════════════


class MultiScaleTradingSystem:
    """Orchestrates micro + macro models + fusion into a unified decision.

    Model-agnostic by design: accepts any objects providing the right
    embedding interface. No hard imports of specific model classes.

    Usage:
        system = MultiScaleTradingSystem(
            lob_model=hierarchical_lob_transformer,   # micro
            cycle_model=price_cycle_model,             # macro
            fusion=multi_scale_fusion,
        )
        decision = system.decide(lob_inputs, bar_features)
    """

    def __init__(self, lob_model, cycle_model, fusion: MultiScaleFusion):
        self.lob_model = lob_model
        self.cycle_model = cycle_model
        self.fusion = fusion

    def decide(
        self,
        lob_order_features: torch.Tensor,
        lob_order_masks: torch.Tensor,
        bar_features: torch.Tensor,
        lob_bar_mask: Optional[torch.Tensor] = None,
        lob_context: Optional[torch.Tensor] = None,
    ) -> dict:
        """Full multi-scale decision.

        Parameters
        ----------
        lob_order_features, lob_order_masks : inputs للـ LOB Transformer
        bar_features : (B, T, 16) — inputs للـ Price Cycle Model

        Returns
        -------
        dict with decision + diagnostics from both scales
        """
        # Micro: LOB Transformer
        self.lob_model.eval()
        with torch.no_grad():
            lob_out = self.lob_model(
                lob_order_features, lob_order_masks,
                lob_bar_mask, lob_context,
            )
            micro_emb = lob_out.shared_embedding

        # Macro: Price Cycle Model
        self.cycle_model.eval()
        with torch.no_grad():
            macro_emb = self.cycle_model.get_cycle_embedding(bar_features)

        # Fusion
        self.fusion.eval()
        with torch.no_grad():
            fusion_out = self.fusion(micro_emb, macro_emb)

        return {
            "decision_probs": fusion_out.decision_probs().cpu().numpy(),
            "micro_gate": fusion_out.micro_gate.cpu().numpy(),
            "macro_gate": fusion_out.macro_gate.cpu().numpy(),
            "alignment_score": fusion_out.alignment_score.cpu().numpy(),
            "micro_direction": lob_out.direction_probs().cpu().numpy(),
            "macro_phase": macro_emb,  # raw embedding
        }
