# Price Cycle Model — التعلّم العميق للدورة السعرية

> Sprint 18 — النموذج الكلّي (macro) المُكمِّل للـ LOB Transformer (micro).

## 🎯 الفكرة

السوق له بنيتان منفصلتان:

```
LOB Transformer (Sprint 17)  →  micro: دفتر الأوامر، ثوانٍ-دقائق
Price Cycle Model (Sprint 18) →  macro: الدورة السعرية، ساعات-أيام
Multi-Scale Fusion            →  يدمج المقياسين في قرار واحد
```

## 🏗 المعمارية

```
BarSequence (OHLCV)
      ↓
16-dim derived features/bar  +  structural features (swing/phase/fractal)
      ↓
Multi-Resolution TCN Encoder  (dilated causal conv، 1× 4× 16×)
      ↓
Cycle Embedding (64-dim)
      ↓
5 Multi-Task Heads:
  - phase            (Wyckoff: accumulation/markup/distribution/markdown)
  - trend_maturity   (young/mature/exhausted)
  - swing_direction  (up/down/neutral)
  - reversal_proximity (regression)
  - cycle_position   (regression [0,1])
```

## 📦 المكونات (9 modules)

| Module | الوظيفة |
|---|---|
| `config.py` | type-safe configs |
| `data_structures.py` | BarSequence, SwingPoint, enums |
| `swing_analysis.py` | causal ZigZag + HH/HL/LH/LL + trend maturity |
| `phase_classifier.py` | Wyckoff phase (rules-based + features) |
| `fractal_features.py` | Hurst exponent + multi-TF alignment |
| `cycle_encoder.py` | multi-resolution TCN backbone (PyTorch) |
| `cycle_model.py` | full model + 5 heads |
| `multi_scale_fusion.py` | micro ⊗ macro fusion + MultiScaleTradingSystem |
| `feature_pipeline.py` | DataFrame → tensors + weak labels |

## 🔬 الأساس العلمي

- **Wyckoff (1931)** — market cycle theory (accumulation → markup → distribution → markdown)
- **Mandelbrot (1963)** — self-similarity of price series
- **Hurst (1951)** — rescaled-range analysis
- **Bai, Kolter, Koltun (2018)** — Temporal Convolutional Networks
- **Lo, Mamaysky, Wang (2000)** — Foundations of Technical Analysis
- **Lo (2004)** — Adaptive Markets Hypothesis (multi-scale dynamics)

## 🧩 لماذا TCN وليس LSTM/Transformer للـ macro؟

| الخاصية | الفائدة |
|---|---|
| **Causal by construction** | dilated causal conv → صفر look-ahead |
| **Receptive field كبير** | 6 levels → ~253 bar في الماضي |
| **Parallelizable** | أسرع من LSTM في التدريب |
| **Stable gradients** | residual + weight norm |

## 🚀 الاستخدام

```python
import torch
from modules.price_cycle import (
    BarSequence, PriceCycleConfig, PriceCycleModel,
    build_cycle_features,
)

# 1. Build bar sequence from OHLCV DataFrame
bars = BarSequence.from_dataframe(ohlcv_df)

# 2. Feature pipeline (+ weak labels)
features = build_cycle_features(bars, PriceCycleConfig())

# 3. Model
model = PriceCycleModel(PriceCycleConfig())
bar_feats = torch.from_numpy(features.bar_features).unsqueeze(0)
output = model(bar_feats)
# output.phase_probs(), output.cycle_position, ...
```

### Multi-Scale Fusion (الدمج مع LOB Transformer)

```python
from modules.deep_lob import HierarchicalLOBTransformer, DeepLOBConfig
from modules.price_cycle import (
    PriceCycleModel, PriceCycleConfig,
    MultiScaleFusion, MultiScaleFusionConfig,
    MultiScaleTradingSystem,
)

lob = HierarchicalLOBTransformer(DeepLOBConfig())       # micro
cycle = PriceCycleModel(PriceCycleConfig())              # macro
fusion = MultiScaleFusion(MultiScaleFusionConfig())

system = MultiScaleTradingSystem(lob, cycle, fusion)
decision = system.decide(
    lob_order_features, lob_order_masks,   # micro inputs
    bar_features,                           # macro inputs
)
# decision = {
#     "decision_probs": [...],         ← القرار الموحّد
#     "micro_gate": ...,               ← كم اعتمد على micro
#     "macro_gate": ...,               ← كم اعتمد على macro
#     "alignment_score": ...,          ← توافق micro/macro
# }
```

## 🧪 Tests

```bash
python3 -m unittest discover tests/price_cycle -v
# 50 tests:
#   - swing analysis (causal detection + classification + maturity)
#   - phase classifier (Wyckoff)
#   - fractal features (Hurst, multi-TF)
#   - models (encoder, full model, fusion)
#   - pipeline + config + causality verification
```

## ✅ ضمانات

| الضمان | كيف |
|---|---|
| **Causality** | كل feature + كل layer causal — مُختبَر بـ `test_causality_no_lookahead` |
| **Model-agnostic fusion** | `MultiScaleFusion` يقبل embeddings فقط، لا يستورد models |
| **Backward compat** | 100% — module مستقل، opt-in |
| **التكامل مع Sprint 17** | branch متفرّع من Sprint 17 → LOB Transformer مرئي |

## 📋 الحالة

| البند | الحالة |
|---|---|
| Architecture (9 modules) | ✅ |
| Tests | ✅ 50/50 |
| Multi-Scale Fusion | ✅ يربط micro + macro |
| **Merge** | ⏸️ **بدون merge — ينتظر طلبك** |
| Real-data training | ⏸️ ينتظر بياناتك |
