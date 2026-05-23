"""
modules/deep_lob/config.py
──────────────────────────
Configuration dataclasses للـ Hierarchical LOB Transformer.

All hyperparameters centralized + validated.
Defaults chosen based on production research at major quant firms
(scaled down for 6B FX intraday data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ════════════════════════════════════════════════════════════════════════════
# Component configs
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class OrderEmbedderConfig:
    """Order-level embedding configuration.

    Each order is embedded as a vector encoding:
        - side (B/S) as 1-bit
        - size (log-normalized)
        - price (distance from mid, scaled by tick)
        - timestamp (relative to bar start, milliseconds)
        - type (T=trade, A=add, C=cancel)
        - venue (optional, default 0)
    """

    # Discrete categorical embeddings
    side_vocab_size: int = 2       # B, S
    type_vocab_size: int = 4       # T, A, C, M (modify)
    venue_vocab_size: int = 8      # up to 8 venues

    # Continuous features
    size_log_scale: float = 1.0    # log1p(size) * scale
    price_tick_size: float = 0.0001  # FX 6B = 0.0001
    time_scale_ms: float = 1000.0  # normalize time to seconds

    # Embedding dim
    embed_dim: int = 32

    # Dropout على الـ embedding
    dropout: float = 0.1

    def __post_init__(self):
        if self.embed_dim < 8:
            raise ValueError(f"embed_dim must be >= 8, got {self.embed_dim}")
        if self.embed_dim % 4 != 0:
            raise ValueError(f"embed_dim must be divisible by 4, got {self.embed_dim}")


@dataclass
class TransformerConfig:
    """Transformer encoder configuration (order-level).

    Captures cross-order interactions via multi-head self-attention.
    """

    # Model dimensions
    embed_dim: int = 32
    n_heads: int = 4               # embed_dim must be divisible by n_heads
    n_layers: int = 4
    feedforward_dim: int = 128

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Activation
    activation: str = "gelu"  # gelu | relu

    # Positional encoding
    max_orders_per_bar: int = 500
    positional_encoding: str = "learned"  # learned | sinusoidal

    # Initialization
    init_std: float = 0.02

    def __post_init__(self):
        if self.embed_dim % self.n_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_layers < 1 or self.n_layers > 24:
            raise ValueError(f"n_layers must be in [1, 24], got {self.n_layers}")
        if self.activation not in {"gelu", "relu"}:
            raise ValueError(f"activation must be gelu|relu, got {self.activation!r}")
        if self.positional_encoding not in {"learned", "sinusoidal"}:
            raise ValueError(
                f"positional_encoding must be learned|sinusoidal, got {self.positional_encoding!r}"
            )


@dataclass
class EventAggregatorConfig:
    """Event-level aggregation (clusters orders into events).

    Discovered events (soft clustering):
        - sweep:        cascade of aggressive orders
        - absorb:       passive orders eating aggressive
        - spoof:        large orders then cancellation
        - iceberg:      small visible, large hidden
        - wall_build:   accumulating passive orders
        - wall_break:   walls being consumed
    """

    n_event_types: int = 6
    event_dim: int = 32
    pooling: str = "attention"  # attention | mean | max
    temperature: float = 1.0    # softmax temperature for soft clustering


@dataclass
class BarLSTMConfig:
    """Bar-level LSTM configuration.

    Sequences events across bars → narrative understanding.
    """

    input_dim: int = 32        # = EventAggregatorConfig.event_dim
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False  # causal للـ real-time
    sequence_length: int = 50    # نفس DeepLOB time_steps


@dataclass
class ContextEncoderConfig:
    """Encodes the 138 features الموجودة كـ context vector."""

    n_input_features: int = 138
    hidden_dim: int = 64
    output_dim: int = 32

    # Per-group encoders (regime, simulators, bell pairs, etc.)
    use_per_group_encoding: bool = True

    dropout: float = 0.1


@dataclass
class MultiTaskHeadsConfig:
    """Multi-task prediction heads.

    Auxiliary tasks force the backbone to learn 'fundamental' market structure:
        - direction (main, classification 3-way)
        - next_price (auxiliary, regression)
        - next_imbalance (auxiliary, regression)
        - next_volatility (auxiliary, regression)
        - next_regime (auxiliary, classification 4-way: matches regime_config.REGIMES)
        - wall_persist (auxiliary, regression - bars until wall consumed)
        - time_to_event (auxiliary, regression - bars until next significant event)
    """

    shared_dim: int = 64       # input dim from fusion layer

    # Direction (main task)
    direction_n_classes: int = 3     # LONG / SHORT / NEUTRAL
    direction_weight: float = 1.0     # primary loss weight

    # Auxiliary tasks (each contributes to backbone learning)
    next_price_weight: float = 0.3
    next_imbalance_weight: float = 0.2
    next_volatility_weight: float = 0.2
    next_regime_weight: float = 0.2
    wall_persist_weight: float = 0.15
    time_to_event_weight: float = 0.10

    # Regime config — matches regime_config.REGIMES order:
    #   0=trending, 1=ranging, 2=volatile, 3=low_liquidity
    regime_n_classes: int = 4

    # Dropout على الـ heads
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    Production-grade defaults (with sensible bounds).
    """

    # Optimizer
    optimizer: str = "adamw"  # adamw | adam | sgd
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8

    # Schedule
    scheduler: str = "cosine_warmup"  # cosine_warmup | constant | step
    warmup_steps: int = 1000
    max_steps: int = 100_000

    # Batch
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Mixed precision
    use_amp: bool = True       # automatic mixed precision (fp16)

    # Validation
    val_interval_steps: int = 500
    val_batches: int = 50

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"  # val_loss | val_direction_acc

    # Checkpointing
    checkpoint_interval_steps: int = 2000
    keep_top_k_checkpoints: int = 3

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self):
        if self.optimizer not in {"adamw", "adam", "sgd"}:
            raise ValueError(f"optimizer must be adamw|adam|sgd, got {self.optimizer!r}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


# ════════════════════════════════════════════════════════════════════════════
# Top-level config
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class DeepLOBConfig:
    """Top-level configuration for Hierarchical LOB Transformer + MTL."""

    order_embedder: OrderEmbedderConfig = field(default_factory=OrderEmbedderConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    event_aggregator: EventAggregatorConfig = field(default_factory=EventAggregatorConfig)
    bar_lstm: BarLSTMConfig = field(default_factory=BarLSTMConfig)
    context_encoder: ContextEncoderConfig = field(default_factory=ContextEncoderConfig)
    multi_task_heads: MultiTaskHeadsConfig = field(default_factory=MultiTaskHeadsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Model name (for checkpoint identification)
    model_name: str = "hierarchical_lob_transformer_mtl_v1"
    version: int = 1

    def __post_init__(self):
        # Cross-component validation
        if self.order_embedder.embed_dim != self.transformer.embed_dim:
            raise ValueError(
                f"order_embedder.embed_dim ({self.order_embedder.embed_dim}) "
                f"must == transformer.embed_dim ({self.transformer.embed_dim})"
            )
        if self.event_aggregator.event_dim != self.bar_lstm.input_dim:
            raise ValueError(
                f"event_aggregator.event_dim ({self.event_aggregator.event_dim}) "
                f"must == bar_lstm.input_dim ({self.bar_lstm.input_dim})"
            )
        # shared_dim must accommodate bar_lstm + context fusion
        expected_shared = self.bar_lstm.hidden_dim
        if self.multi_task_heads.shared_dim != expected_shared:
            # auto-adjust بدلاً من error
            self.multi_task_heads.shared_dim = expected_shared

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (للـ checkpoints + logging)."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def production_default(cls) -> "DeepLOBConfig":
        """Production-tuned defaults for FX intraday."""
        return cls()

    @classmethod
    def small_dev(cls) -> "DeepLOBConfig":
        """Smaller config للـ rapid iteration on dev."""
        return cls(
            transformer=TransformerConfig(
                embed_dim=16, n_heads=2, n_layers=2, feedforward_dim=32,
            ),
            order_embedder=OrderEmbedderConfig(embed_dim=16),
            event_aggregator=EventAggregatorConfig(event_dim=16),
            bar_lstm=BarLSTMConfig(input_dim=16, hidden_dim=32, n_layers=1),
            context_encoder=ContextEncoderConfig(output_dim=16, hidden_dim=32),
            multi_task_heads=MultiTaskHeadsConfig(shared_dim=32),
        )
