"""
modules.deep_lob.masked_modeling.config
═══════════════════════════════════════════════════════════════════════════════
Configuration for the masked-reconstruction auxiliary task.

Off by default for backward compatibility. When `enabled=True`, the
pretrain loop generates a random mask over the order-features tensor,
zeroes out the masked positions (optionally concatenating a mask
indicator), runs the encoder, and reconstructs the original features
from the encoded output. The MSE on the masked subset is added to the
total loss with `reconstruction_weight`.

Strategy
────────
random  Each valid order has independent Bernoulli(mask_ratio) chance
        of being masked. Standard BERT-style masking. Default.

patch   Contiguous patches of `patch_size` orders within each bar are
        masked together. Mirrors temporary tape outages — useful when
        the model needs to recover from gaps, not just isolated drops.

bar     Whole bars are masked (all N orders in the bar). The strongest
        test of the model's bar-level context understanding.

Trading-aware notes
───────────────────
1. mask_ratio between 0.10 and 0.25 is the documented sweet spot
   (BERT 0.15, MAE 0.75 for images; tabular financial data sits closer
   to BERT — too aggressive masking destroys causal microstructure).
2. provide_mask_indicator=True is the SimMIM/MAE convention for
   continuous data: the model needs to know which positions are
   genuinely zero vs masked-to-zero. Setting to False puts you in
   strict BERT mode (model must infer mask positions from context).
3. min_mask_per_sample guarantees at least one masked position per
   training sample so the reconstruction head always gets a gradient.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MaskedModelingConfig:
    """Auxiliary masked-reconstruction config for the deep_lob backbone."""

    # ── On/off ──────────────────────────────────────────────────────
    enabled: bool = False

    # ── Masking ─────────────────────────────────────────────────────
    mask_ratio: float = 0.15
    strategy: str = "random"            # "random" | "patch" | "bar"
    patch_size: int = 4                 # only used when strategy="patch"
    min_mask_per_sample: int = 1        # guarantees gradient signal

    # ── Input augmentation ──────────────────────────────────────────
    provide_mask_indicator: bool = True

    # ── Reconstruction head ─────────────────────────────────────────
    reconstruction_hidden_dim: int = 128

    # ── Loss weight ─────────────────────────────────────────────────
    reconstruction_weight: float = 1.0

    # ── RNG ─────────────────────────────────────────────────────────
    # When set, mask generation is deterministic (essential for
    # debugging the auxiliary loss while training the main objective).
    seed: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.mask_ratio < 1.0):
            raise ValueError(
                f"mask_ratio must be in (0, 1), got {self.mask_ratio}"
            )
        if self.strategy not in {"random", "patch", "bar"}:
            raise ValueError(
                f"strategy must be 'random' | 'patch' | 'bar', "
                f"got {self.strategy!r}"
            )
        if self.patch_size < 1:
            raise ValueError(
                f"patch_size must be ≥ 1, got {self.patch_size}"
            )
        if self.min_mask_per_sample < 0:
            raise ValueError(
                f"min_mask_per_sample must be ≥ 0, got "
                f"{self.min_mask_per_sample}"
            )
        if self.reconstruction_hidden_dim < 1:
            raise ValueError(
                f"reconstruction_hidden_dim must be ≥ 1, got "
                f"{self.reconstruction_hidden_dim}"
            )
        if self.reconstruction_weight < 0.0:
            raise ValueError(
                f"reconstruction_weight must be ≥ 0, got "
                f"{self.reconstruction_weight}"
            )
