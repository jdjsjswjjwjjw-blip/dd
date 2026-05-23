"""
modules/price_cycle — Macro-scale deep learning للدورة السعرية.

النموذج الكلّي المُكمِّل للـ LOB Transformer (الميكروي):

    LOB Transformer (micro)  → ماذا يحدث الآن في دفتر الأوامر
    Price Cycle Model (macro) → أين نحن في الدورة الكبرى
    Multi-Scale Fusion        → قرار يحترم المقياسين

Components:
    - data_structures   : BarSequence, SwingPoint, enums
    - swing_analysis    : causal swing detection + HH/HL/LH/LL + trend maturity
    - phase_classifier  : Wyckoff phase (accumulation/markup/distribution/markdown)
    - fractal_features  : Hurst exponent + multi-timeframe alignment
    - cycle_encoder     : multi-resolution TCN backbone
    - cycle_model       : full Price Cycle Model + 5 heads
    - multi_scale_fusion: micro ⊗ macro fusion
    - feature_pipeline  : DataFrame → tensors + weak labels

Scientific references:
    - Wyckoff (1931): market cycle theory
    - Mandelbrot (1963): self-similarity of prices
    - Hurst (1951): rescaled-range analysis
    - Bai, Kolter, Koltun (2018): Temporal Convolutional Networks
    - Lo, Mamaysky, Wang (2000): Foundations of Technical Analysis
    - Lo (2004): Adaptive Markets Hypothesis
"""

from .config import (
    PriceCycleConfig,
    SwingConfig,
    PhaseConfig,
    FractalConfig,
    CycleEncoderConfig,
    CycleHeadsConfig,
    MultiScaleFusionConfig,
)
from .data_structures import (
    BarSequence,
    SwingPoint,
    SwingType,
    WyckoffPhase,
    TrendMaturity,
)
from .swing_analysis import (
    detect_swings,
    classify_swings,
    swing_structure_score,
    estimate_trend_maturity,
    TrendMaturityResult,
)
from .phase_classifier import (
    PhaseAssessment,
    compute_phase_features,
    classify_phase_rulesbased,
    classify_phase_sequence,
)
from .fractal_features import (
    hurst_exponent,
    rolling_hurst,
    fractal_dimension,
    multi_timeframe_alignment,
    compute_fractal_features,
)
from .cycle_encoder import (
    CycleEncoder,
    TCNEncoder,
    TemporalBlock,
    CausalConv1d,
)
from .cycle_model import (
    PriceCycleModel,
    CycleHeads,
    CycleOutput,
    CycleTargets,
)
from .multi_scale_fusion import (
    MultiScaleFusion,
    MultiScaleTradingSystem,
    FusionOutput,
    CrossAttentionFusion,
    GatingFusion,
    ConcatFusion,
)
from .feature_pipeline import (
    CycleFeatureSet,
    build_cycle_features,
    make_training_windows,
)

__all__ = [
    # Config
    "PriceCycleConfig", "SwingConfig", "PhaseConfig", "FractalConfig",
    "CycleEncoderConfig", "CycleHeadsConfig", "MultiScaleFusionConfig",
    # Data
    "BarSequence", "SwingPoint", "SwingType", "WyckoffPhase", "TrendMaturity",
    # Swing analysis
    "detect_swings", "classify_swings", "swing_structure_score",
    "estimate_trend_maturity", "TrendMaturityResult",
    # Phase
    "PhaseAssessment", "compute_phase_features",
    "classify_phase_rulesbased", "classify_phase_sequence",
    # Fractal
    "hurst_exponent", "rolling_hurst", "fractal_dimension",
    "multi_timeframe_alignment", "compute_fractal_features",
    # Encoder
    "CycleEncoder", "TCNEncoder", "TemporalBlock", "CausalConv1d",
    # Model
    "PriceCycleModel", "CycleHeads", "CycleOutput", "CycleTargets",
    # Fusion
    "MultiScaleFusion", "MultiScaleTradingSystem", "FusionOutput",
    "CrossAttentionFusion", "GatingFusion", "ConcatFusion",
    # Pipeline
    "CycleFeatureSet", "build_cycle_features", "make_training_windows",
]
