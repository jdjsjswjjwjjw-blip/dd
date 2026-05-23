"""
modules/price_cycle/config.py
──────────────────────────────
Configuration للـ Price Cycle Model (macro-scale deep learning).

النموذج يتعلّم البنية الكلّية للسوق على مستوى الـ bars/swings/cycles —
بخلاف الـ LOB Transformer الذي يعمل على مستوى الـ orders (micro).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SwingConfig:
    """Swing detection (ZigZag-based, causal).

    Swing = local extreme defined by a minimum reversal threshold.
    Classified as HH/HL/LH/LL relative to previous swings.
    """

    # Reversal threshold: a swing forms when price reverses by this much
    reversal_pct: float = 0.0          # if > 0, percent-based
    reversal_atr_mult: float = 2.0     # if reversal_pct == 0, ATR-based
    atr_window: int = 14

    # Minimum bars between swings (de-noising)
    min_bars_between_swings: int = 3

    # Lookback for swing classification
    classification_lookback: int = 4   # compare against last N swings


@dataclass
class PhaseConfig:
    """Wyckoff phase classification config.

    Phases: accumulation, markup, distribution, markdown.
    """

    n_phases: int = 4
    # Window for phase feature computation
    feature_window: int = 50
    # Volume profile bins
    volume_profile_bins: int = 20
    # Smoothing
    smoothing_window: int = 10


@dataclass
class FractalConfig:
    """Multi-timeframe fractal feature config."""

    # Timeframe multipliers (relative to base bar)
    timeframe_multipliers: tuple[int, ...] = (1, 4, 16)
    # Hurst exponent window
    hurst_window: int = 100
    # Hurst lag range
    hurst_max_lag: int = 20


@dataclass
class CycleEncoderConfig:
    """Temporal Convolutional Network (TCN) encoder config.

    TCN advantages للـ price sequences (Bai, Kolter, Koltun 2018):
        - Causal (no look-ahead leakage)
        - Large receptive field via dilation
        - Parallelizable (faster than LSTM)
        - Stable gradients
    """

    # Input features per bar
    n_input_features: int = 16   # OHLCV + derived (returns, ATR, RSI, etc.)

    # TCN structure
    n_channels: int = 64         # hidden channels per layer
    kernel_size: int = 3
    n_levels: int = 6            # dilation 1,2,4,8,16,32 → receptive field ~127
    dropout: float = 0.1

    # Multi-resolution
    use_multi_resolution: bool = True
    resolution_factors: tuple[int, ...] = (1, 4, 16)

    # Output embedding dim
    embed_dim: int = 64

    def __post_init__(self):
        if self.n_levels < 1 or self.n_levels > 12:
            raise ValueError(f"n_levels must be in [1, 12], got {self.n_levels}")
        if self.kernel_size < 2:
            raise ValueError(f"kernel_size must be >= 2, got {self.kernel_size}")

    @property
    def receptive_field(self) -> int:
        """Receptive field = 1 + 2*(kernel_size-1)*(2^n_levels - 1)."""
        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.n_levels - 1)


@dataclass
class CycleHeadsConfig:
    """Multi-task heads للـ Price Cycle Model."""

    shared_dim: int = 64

    # Phase classification (Wyckoff)
    phase_n_classes: int = 4         # accumulation/markup/distribution/markdown
    phase_weight: float = 1.0

    # Trend maturity
    maturity_n_classes: int = 3      # young/mature/exhausted
    maturity_weight: float = 0.5

    # Swing direction prediction
    swing_n_classes: int = 3         # up/down/neutral
    swing_weight: float = 0.5

    # Reversal proximity (regression: bars until likely reversal)
    reversal_weight: float = 0.3

    # Cycle position (regression: [0,1] phase within full cycle)
    cycle_position_weight: float = 0.2

    dropout: float = 0.1


@dataclass
class MultiScaleFusionConfig:
    """Config للـ fusion بين LOB Transformer (micro) و Price Cycle (macro)."""

    micro_dim: int = 64       # LOB Transformer shared embedding
    macro_dim: int = 64       # Price Cycle embedding
    fusion_dim: int = 64      # output

    fusion_method: str = "cross_attention"  # cross_attention | gating | concat
    n_heads: int = 4
    dropout: float = 0.1

    # Decision heads (after fusion)
    decision_n_classes: int = 3   # LONG/SHORT/NEUTRAL

    def __post_init__(self):
        if self.fusion_method not in {"cross_attention", "gating", "concat"}:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method!r}")


@dataclass
class PriceCycleConfig:
    """Top-level config للـ Price Cycle Model."""

    swing: SwingConfig = field(default_factory=SwingConfig)
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    fractal: FractalConfig = field(default_factory=FractalConfig)
    encoder: CycleEncoderConfig = field(default_factory=CycleEncoderConfig)
    heads: CycleHeadsConfig = field(default_factory=CycleHeadsConfig)
    fusion: MultiScaleFusionConfig = field(default_factory=MultiScaleFusionConfig)

    model_name: str = "price_cycle_model_v1"
    version: int = 1

    def __post_init__(self):
        # Consistency: encoder embed_dim → heads shared_dim
        if self.encoder.embed_dim != self.heads.shared_dim:
            self.heads.shared_dim = self.encoder.embed_dim
        # Consistency: fusion macro_dim ← encoder embed_dim
        if self.fusion.macro_dim != self.encoder.embed_dim:
            self.fusion.macro_dim = self.encoder.embed_dim

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def small_dev(cls) -> "PriceCycleConfig":
        """Smaller config للـ rapid iteration."""
        return cls(
            encoder=CycleEncoderConfig(
                n_channels=16, n_levels=3, embed_dim=32,
                resolution_factors=(1, 4),
            ),
            heads=CycleHeadsConfig(shared_dim=32),
            fusion=MultiScaleFusionConfig(micro_dim=32, macro_dim=32, fusion_dim=32),
        )
