"""
modules/deep_lob/__init__.py
─────────────────────────────
Sprint 17: Hierarchical LOB Transformer + Multi-Task Heads.

Deep structural understanding of the limit order book through:
  1. Order-level embeddings (size, side, price, time, type)
  2. Self-attention across orders within bar (Transformer)
  3. Event-level aggregation (sweep / absorb / spoof / iceberg)
  4. Bar-level sequence modeling (LSTM)
  5. Multi-task heads (direction, target, stop, regime, volatility, wall)

Scientific references:
  - Zhang, Zohren, Roberts (2019): "DeepLOB" - foundation
  - Vaswani et al. (2017): "Attention is All You Need" - transformer
  - Caruana (1997): "Multitask Learning" - auxiliary tasks
  - Easley, López de Prado, O'Hara (2012): "VPIN" - flow toxicity
  - Xiong et al. (2020): "On Layer Normalization in the Transformer"
  - Lee et al. (2019): "Set Transformer" - event aggregation inspiration
  - Kendall et al. (2018): "Multi-Task Learning Using Uncertainty"
"""

from .config import (
    DeepLOBConfig,
    OrderEmbedderConfig,
    TransformerConfig,
    EventAggregatorConfig,
    BarLSTMConfig,
    ContextEncoderConfig,
    MultiTaskHeadsConfig,
    TrainingConfig,
)
from .data_structures import (
    Order,
    OrderType,
    OrderSide,
    OrderBatch,
    BarLOB,
)
from .order_embedder import OrderEmbedder, SinusoidalTimeEncoding
from .transformer_blocks import (
    LOBTransformerEncoder,
    LOBTransformerLayer,
    MultiHeadSelfAttention,
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from .event_aggregator import EventAggregator
from .bar_lstm import BarLevelLSTM
from .context_encoder import ContextEncoder, CrossAttentionFusion
from .multi_task_heads import (
    MultiTaskHeads,
    MultiTaskOutput,
    MultiTaskTargets,
)
from .hierarchical_model import HierarchicalLOBTransformer
from .pattern_discovery import (
    DiscoveredPattern,
    discover_critical_orders_from_attention,
    discover_event_assignments,
    discover_clusters_from_embeddings,
    counter_factual_importance,
)
from .anomaly_detection import (
    AnomalyResult,
    MahalanobisAnomalyDetector,
    KNearestAnomalyDetector,
    MultiTaskDiscrepancyDetector,
)
from .training_loop import (
    HierarchicalLOBTrainer,
    TrainingMetrics,
)
from .integration_adapter import (
    DeepLOBCNNAdapter,
    BridgeAdapter,
    compute_dynamic_tp_sl,
)
from .system_integration import (
    get_visual_embedder,
    EnhancedBridgeDecision,
    extend_alpha_library_with_discoveries,
    system_readiness_check,
)

__all__ = [
    # Config
    "DeepLOBConfig",
    "OrderEmbedderConfig",
    "TransformerConfig",
    "EventAggregatorConfig",
    "BarLSTMConfig",
    "ContextEncoderConfig",
    "MultiTaskHeadsConfig",
    "TrainingConfig",
    # Data
    "Order",
    "OrderType",
    "OrderSide",
    "OrderBatch",
    "BarLOB",
    # Components
    "OrderEmbedder",
    "SinusoidalTimeEncoding",
    "LOBTransformerEncoder",
    "LOBTransformerLayer",
    "MultiHeadSelfAttention",
    "LearnedPositionalEmbedding",
    "SinusoidalPositionalEmbedding",
    "EventAggregator",
    "BarLevelLSTM",
    "ContextEncoder",
    "CrossAttentionFusion",
    "MultiTaskHeads",
    "MultiTaskOutput",
    "MultiTaskTargets",
    # Full model
    "HierarchicalLOBTransformer",
    # Pattern discovery
    "DiscoveredPattern",
    "discover_critical_orders_from_attention",
    "discover_event_assignments",
    "discover_clusters_from_embeddings",
    "counter_factual_importance",
    # Anomaly detection
    "AnomalyResult",
    "MahalanobisAnomalyDetector",
    "KNearestAnomalyDetector",
    "MultiTaskDiscrepancyDetector",
    # Training
    "HierarchicalLOBTrainer",
    "TrainingMetrics",
    # Integration
    "DeepLOBCNNAdapter",
    "BridgeAdapter",
    "compute_dynamic_tp_sl",
    # System integration
    "get_visual_embedder",
    "EnhancedBridgeDecision",
    "extend_alpha_library_with_discoveries",
    "system_readiness_check",
]
