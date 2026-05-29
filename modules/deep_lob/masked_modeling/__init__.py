"""
modules.deep_lob.masked_modeling
═══════════════════════════════════════════════════════════════════════════════
Masked-reconstruction auxiliary task for the deep_lob SSL backbone.

Why this exists
───────────────
The existing pretrain_lob.py trains six FORECASTING tasks (next_price,
next_imbalance, next_volatility, next_regime, wall_persist,
time_to_event). These give the model a forward-looking signal but
never force it to learn "what is consistent with the surrounding
context". Masked reconstruction adds that signal: random positions in
the input are zeroed and the model must rebuild them from the rest of
the window.

The two paradigms are complementary, not exclusive — this module is
designed as an ADDITIONAL task, not a replacement.

Usage
─────
    from modules.deep_lob.masked_modeling import (
        MaskedModelingConfig, generate_mask, apply_mask,
        ReconstructionHead, compute_recon_loss,
    )

    cfg = MaskedModelingConfig(enabled=True, mask_ratio=0.15)
    head = ReconstructionHead(
        encoder_dim=hidden_dim,
        feature_dim=ORDER_FEATURE_DIM,
        hidden_dim=cfg.reconstruction_hidden_dim,
    )

    # In the training step (after the existing encoder forward):
    mask = generate_mask(order_masks, cfg, generator=rng)
    masked_input = apply_mask(order_features, mask,
                              provide_mask_indicator=cfg.provide_mask_indicator)
    encoded = backbone.encode(masked_input, order_masks, ...)
    recon = head(encoded)
    l_recon = compute_recon_loss(
        recon, order_features, mask, order_masks,
    )
    total_loss = existing_total + cfg.reconstruction_weight * l_recon

Engineering contract
────────────────────
- Pure mask generation (seedable) → reproducible debugging.
- Padding-aware: never asks the model to reconstruct padding.
- Per-sample mask floor → every batch sample gets gradient signal.
- Off by default (`enabled=False`) → backward-compatible with the
  existing six-task setup.
"""
from modules.deep_lob.masked_modeling.config import MaskedModelingConfig
from modules.deep_lob.masked_modeling.masking import (
    apply_mask,
    generate_mask,
)
from modules.deep_lob.masked_modeling.reconstruction import (
    ReconstructionHead,
    compute_recon_loss,
)

__all__ = [
    "MaskedModelingConfig",
    "ReconstructionHead",
    "apply_mask",
    "compute_recon_loss",
    "generate_mask",
]
