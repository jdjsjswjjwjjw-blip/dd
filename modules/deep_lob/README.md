# Hierarchical LOB Transformer + Multi-Task Heads

> Sprint 17 — Deep structural understanding of the limit order book.

## 🎯 الفكرة المركزية

النموذج يتعلم **بنية دفتر الأوامر** على 3 مستويات هرمية:

```
Order Level  →  Event Level  →  Bar Level  →  Decision
   (orders)     (sweeps, walls,    (sequence    (multi-task
                 icebergs, ...)      of bars)     predictions)
```

بدلاً من تعلّم patterns سطحياً، النموذج يفهم:
- **من** يضع الـ orders (HFT vs institution vs retail)
- **لماذا** (genuine vs spoof vs probe)
- **كيف** تتفاعل (causal chains، wall dynamics)
- **متى** (multi-scale temporal patterns)

---

## 🏗 المعمارية

### Stage 1: Order Embedder
كل order → 32-dim vector عبر:
- Discrete embeddings (side, type, venue)
- Continuous projections (size, price, has_id)
- Sinusoidal time encoding (multi-scale)

### Stage 2: Order-Level Transformer
- 4 layers, 4 heads (default)
- Pre-LayerNorm + GELU (Xiong et al. 2020)
- Self-attention يكشف العلاقات بين الـ orders
- Attention weights قابلة للاستخراج للـ pattern discovery

### Stage 3: Event Aggregator
- 6 learnable event queries (sweep, absorb, spoof, iceberg, wall_build, wall_break)
- Cross-attention: events ← orders (Set Transformer style)
- Soft clustering مع temperature-controlled sharpness

### Stage 4: Bar-Level LSTM
- Causal LSTM على event sequence
- Event pooling per bar (attention-based)
- Final hidden state للـ next stages

### Stage 5: Context Encoder
- يدمج الـ 138 features الموجودة (regime, simulators, Bell pairs)
- MLP مع LayerNorm + GELU
- Output: 32-dim context embedding

### Stage 6: Cross-Attention Fusion
- Book ↔ Context bidirectional attention
- Fused representation: 64-dim shared embedding

### Stage 7: Multi-Task Heads (7 heads)
| Head | Output | Purpose |
|---|---|---|
| direction | (3,) softmax | LONG/SHORT/NEUTRAL (main) |
| next_price | scalar | Direct target prediction |
| next_imbalance | scalar [-1, 1] | Order flow forecast |
| next_volatility | scalar > 0 | ATR-relative forecast |
| next_regime | (3,) softmax | trending/ranging/volatile |
| wall_persist | scalar > 0 | Bars until wall consumed |
| time_to_event | scalar > 0 | Bars until next event |

---

## 📐 الـ Math

### Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### Event Aggregation
```
For each event type e:
    α(e, i) = softmax_i(<Q_e, K_i>)
    Event_e = Σ_i α(e, i) · V_i
```

### Multi-Task Loss
```
L_total = w_main · L_direction + Σ w_aux · L_aux
        = 1.0 · CE(dir)
        + 0.3 · MSE(next_price)
        + 0.2 · MSE(next_imbalance)
        + 0.2 · MSE(log next_vol)
        + 0.2 · CE(next_regime)
        + 0.15 · Huber(wall_persist)
        + 0.10 · Huber(time_to_event)
```

### Dynamic TP/SL
```python
tp_mult = base_tp × vol_factor × wall_factor × time_factor
sl_mult = base_sl × vol_factor

where:
    vol_factor = clip(next_vol / (next_vol + 0.5), 0.5, 2.0)
    wall_factor = 1 - 0.5*0.5 if wall_persist > 5 else 1
    time_factor = 0.8 if time_to_event < 3 else 1
```

---

## 🚀 الاستخدام

### Quick Start

```python
import torch
from modules.deep_lob import (
    DeepLOBConfig,
    HierarchicalLOBTransformer,
    MultiTaskTargets,
    HierarchicalLOBTrainer,
    TrainingConfig,
)

# Configure
config = DeepLOBConfig()  # production defaults
# Or for rapid iteration:
# config = DeepLOBConfig.small_dev()

# Build
model = HierarchicalLOBTransformer(config)
print(model.count_parameters())  # ~5-10M params (default), ~40K (small_dev)

# Train (with your DataLoader)
trainer = HierarchicalLOBTrainer(
    model=model,
    config=TrainingConfig(learning_rate=1e-4, max_steps=10000),
    output_dir="outputs/transformer",
    device="cuda",
)
for step, (batch, targets) in enumerate(my_dataloader):
    metric = trainer.train_step_batch(batch, targets, step=step)
```

### Drop-in replacement for DeepLOB CNN

```python
# Before:
from modules.deeplob_cnn import DeepLOBCNN
embedder = DeepLOBCNN(channels=7)

# After (zero changes to Stage 2 / train_v19.py):
from modules.deep_lob import DeepLOBCNNAdapter
embedder = DeepLOBCNNAdapter(
    channels=7,
    brain_file="outputs/hierarchical_lob_transformer.pt",
)

# Same .predict(lob_tensor) → (N, 8) embedding
```

### Bridge integration (paper/live)

```python
from modules.deep_lob import (
    HierarchicalLOBTransformer,
    EnhancedBridgeDecision,
)

model = HierarchicalLOBTransformer.from_checkpoint("ckpt.pt")
enhancer = EnhancedBridgeDecision(
    transformer_model=model,
    bridge=integration_bridge,
    regime_tp_sl=REGIME_TP_SL,
)

decision = enhancer.evaluate(bar)
# decision = {
#     "action": "OPEN_LONG",
#     "tp_mult_dynamic": 1.85,  # ← adaptive
#     "sl_mult_dynamic": 0.95,
#     "transformer_predictions": {...},
# }
```

### Pattern Discovery

```python
from modules.deep_lob import (
    discover_clusters_from_embeddings,
    extend_alpha_library_with_discoveries,
)

# After training:
all_embeddings = collect_embeddings(model, all_bars)  # (N, 64)
outcomes = collect_outcomes(all_bars)                  # (N,)

patterns = discover_clusters_from_embeddings(
    all_embeddings, outcomes,
    min_cluster_size=20, n_clusters=30,
)

# Auto-add high-quality patterns to alpha library
result = extend_alpha_library_with_discoveries(
    "artifacts/alphas_library.json",
    patterns, min_score=1.0,
)
print(f"Added {result['n_added']} discovered patterns")
```

### Anomaly Detection

```python
from modules.deep_lob import MahalanobisAnomalyDetector

# After training, capture training embeddings
training_embs = collect_embeddings(model, train_bars)

detector = MahalanobisAnomalyDetector(threshold_sigma=3.0)
detector.fit(training_embs)

# Live: check each bar
for bar in live_bars:
    emb = model.get_visual_embedding(bar_tensor)
    result = detector.detect(emb.numpy()[0])
    if result.is_anomaly:
        alert("Novel market state detected", score=result.score)
        # → reduce position size OR halt
```

---

## 🔬 Tests

```bash
python3 -m unittest discover tests/deep_lob -v
# 78 tests covering:
#   - Config validation (4)
#   - Data structures (9)
#   - Components: embedder, transformer, aggregator, LSTM, ... (16)
#   - Full model: forward, backward, predict, checkpoints (10)
#   - Pattern discovery + anomaly detection (10)
#   - Training loop (10)
#   - Integration adapters (8)
#   - System readiness check (2)
```

---

## 📚 References

- **Zhang, Zohren, Roberts (2019)** — "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books". *IEEE TSP*.
- **Vaswani et al. (2017)** — "Attention is All You Need". *NeurIPS*.
- **Xiong et al. (2020)** — "On Layer Normalization in the Transformer Architecture". *ICML*.
- **Lee et al. (2019)** — "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks". *ICML*.
- **Caruana (1997)** — "Multitask Learning". *Machine Learning*.
- **Kendall et al. (2018)** — "Multi-Task Learning Using Uncertainty to Weigh Losses". *CVPR*.
- **Easley, López de Prado, O'Hara (2012)** — "Flow Toxicity and Liquidity in a High-Frequency World". *RFS*.
- **Hendrycks & Gimpel (2016)** — "Gaussian Error Linear Units (GELUs)". arXiv:1606.08415.

---

## 📦 Status

| الجانب | الحالة |
|---|---|
| Architecture implementation | ✅ كامل (9 modules) |
| PyTorch backend | ✅ |
| Training infrastructure | ✅ (AMP, schedulers, checkpointing) |
| Pattern discovery (3 methods) | ✅ |
| Anomaly detection (3 detectors) | ✅ |
| Integration adapters | ✅ (CNN drop-in + Bridge enhancer) |
| System readiness hooks | ✅ |
| Tests | ✅ 78/78 |
| Documentation | ✅ |
| **Merge to trunk** | ⏸️ **awaiting user request** |
| Real-data training | ⏸️ awaiting data |
