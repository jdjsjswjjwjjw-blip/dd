"""
modules.deep_lob.masked_modeling.masking
═══════════════════════════════════════════════════════════════════════════════
Pure mask generation + application for the deep_lob order-features tensor.

Shapes throughout
─────────────────
    order_features    (B, T, N, D)   D=7 by default for deep_lob
    order_masks       (B, T, N) bool  True = real order, False = padding
    mask              (B, T, N) bool  True = position to reconstruct

Design contract
───────────────
generate_mask is PURE — given the same seed, it produces the same mask.
It respects the padding mask: only valid (non-pad) orders are eligible
for masking. This guarantees the reconstruction loss has actual targets
and the model isn't asked to "reconstruct padding".

apply_mask is PURE — zeroes the masked positions in order_features and
optionally concatenates a mask indicator as an extra feature dimension.
The encoder receives a tensor of shape (B, T, N, D) or (B, T, N, D+1).
"""
from __future__ import annotations

import torch

from modules.deep_lob.masked_modeling.config import MaskedModelingConfig


def _enforce_min_per_sample(
    mask: torch.Tensor, order_masks: torch.Tensor, min_count: int
) -> torch.Tensor:
    """For each sample in the batch, ensure at least `min_count` valid
    positions are masked. If a sample's mask sum < min_count, randomly
    pick additional valid positions until the floor is met.

    Pure: returns a new tensor, doesn't mutate the input.
    """
    if min_count <= 0:
        return mask

    B = mask.shape[0]
    out = mask.clone()
    for b in range(B):
        valid_positions = order_masks[b].reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if valid_positions.numel() == 0:
            continue
        current = out[b].reshape(-1)
        already_masked = current.nonzero(as_tuple=False).squeeze(-1).numel()
        need = min_count - already_masked
        if need <= 0:
            continue
        # Candidates: valid positions not already masked
        candidates = valid_positions[~current[valid_positions]]
        if candidates.numel() == 0:
            continue
        take = min(int(need), int(candidates.numel()))
        # Deterministic pick: first `take` candidates (already shuffled by
        # the caller's seed via the initial Bernoulli draw)
        idx = candidates[:take]
        current[idx] = True
        out[b] = current.reshape(out[b].shape)
    return out


def _random_mask(
    shape: tuple[int, int, int],
    ratio: float,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    """Bernoulli mask of shape (B, T, N) at probability `ratio`."""
    rand = torch.rand(shape, generator=generator, device=device)
    return rand < ratio


def _patch_mask(
    shape: tuple[int, int, int],
    ratio: float,
    patch_size: int,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    """Mask contiguous patches of `patch_size` orders along the order
    dim N. The number of patches per (bar, batch) is set so the
    expected mask ratio matches `ratio`."""
    B, T, N = shape
    mask = torch.zeros(B, T, N, dtype=torch.bool, device=device)
    # Number of patch starts per bar so coverage ≈ ratio
    n_patches = max(1, int(round((N * ratio) / max(patch_size, 1))))
    for b in range(B):
        for t in range(T):
            # Sample n_patches start positions
            max_start = max(N - patch_size + 1, 1)
            starts = torch.randint(
                0, max_start, (n_patches,),
                generator=generator, device=device,
            )
            for s in starts.tolist():
                mask[b, t, s : s + patch_size] = True
    return mask


def _bar_mask(
    shape: tuple[int, int, int],
    ratio: float,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    """Mask whole bars (all N orders in the chosen bars)."""
    B, T, N = shape
    bar_drop = torch.rand(B, T, generator=generator, device=device) < ratio
    return bar_drop.unsqueeze(-1).expand(B, T, N).contiguous()


def generate_mask(
    order_masks: torch.Tensor, config: MaskedModelingConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Produce a boolean mask of the same shape as `order_masks`.

    True positions are the ones the reconstruction head must predict.
    Padding positions (where `order_masks` is False) are always False
    in the output — we never ask the model to reconstruct padding.

    Pure: same `generator` state → same mask.
    """
    if order_masks.dtype != torch.bool:
        raise TypeError(
            f"order_masks must be bool, got {order_masks.dtype}"
        )
    B, T, N = order_masks.shape
    device = order_masks.device

    if config.strategy == "random":
        raw = _random_mask((B, T, N), config.mask_ratio, generator, device)
    elif config.strategy == "patch":
        raw = _patch_mask(
            (B, T, N), config.mask_ratio, config.patch_size,
            generator, device,
        )
    elif config.strategy == "bar":
        raw = _bar_mask((B, T, N), config.mask_ratio, generator, device)
    else:
        raise ValueError(f"unknown strategy: {config.strategy!r}")

    # Padding-aware: never mask padding positions
    mask = raw & order_masks

    # Guarantee a per-sample floor so the head gets gradient signal
    mask = _enforce_min_per_sample(
        mask, order_masks, config.min_mask_per_sample,
    )
    return mask


def apply_mask(
    order_features: torch.Tensor,
    mask: torch.Tensor,
    provide_mask_indicator: bool = True,
) -> torch.Tensor:
    """Zero-out masked positions and optionally concat a mask indicator.

    order_features : (B, T, N, D) float
    mask           : (B, T, N) bool — True = masked

    Returns
    ───────
    (B, T, N, D)    when provide_mask_indicator=False
    (B, T, N, D+1)  when provide_mask_indicator=True (extra dim is 1.0
                    where masked, 0.0 elsewhere)

    Pure: input tensors are not mutated.
    """
    if order_features.shape[:3] != mask.shape:
        raise ValueError(
            f"shape mismatch: order_features {tuple(order_features.shape)} "
            f"vs mask {tuple(mask.shape)}"
        )
    not_masked = (~mask).unsqueeze(-1).to(order_features.dtype)
    out = order_features * not_masked
    if provide_mask_indicator:
        indicator = mask.unsqueeze(-1).to(order_features.dtype)
        out = torch.cat([out, indicator], dim=-1)
    return out
