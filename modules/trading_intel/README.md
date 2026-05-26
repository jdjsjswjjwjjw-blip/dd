# `trading_intel/` — Trading Intelligence Layer

All enhancements added on top of the legacy `day_trade` rule system
and the `deep_lob` + `price_cycle` SSL backbones. Single package, clear
boundaries, no scattered files.

```
trading_intel/
├── __init__.py              # Public API
├── README.md                # This file
│
├── lob/                     # Enhanced LOB reading
│   ├── features.py          # 13-channel LOB tensor builder (pure numpy)
│   └── cnn.py               # Human-style 4-layer CNN, 32-dim embedding
│
├── ssl_heads/               # Trading-aware SSL heads
│   ├── short_term.py        # 5 heads: wall_break, imbalance_shift,
│   │                        # micro_target, liquidity_sweep, gap_fill
│   ├── short_term_labels.py # Causal label builders for above
│   ├── adaptive.py          # 3 heads: max_R_reached, target_bucket,
│   │                        # regime_risk
│   └── adaptive_labels.py   # Causal label builders for above
│
├── hybrid/                  # Fusion + decision
│   ├── model.py             # MLP fusing day_trade + SSL + CNN
│   └── policy.py            # Auditable rule-based decision layer
│
└── training/
    └── train_hybrid.py      # End-to-end training script
```

## Data flow (the only diagram you need)

```
                  RAW MBO + MBP-10 DATA
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
       ▼                  ▼                  ▼
   prepare_           build_order        lob/features.py
   day_trading.py     batches.py         (13-channel LOB)
   (existing)         (existing)         (new)
       │                  │                  │
   135 features       order tensors     LOB tensor v2
   + rule labels      (B, T, N, 7)       (B, T, P, 13)
       │                  │                  │
       │                  │                  ▼
       │                  │         lob/cnn.py
       │                  │       (HumanLOBCNN)
       │                  │             │
       │                  │             ▼
       │                  │       32-dim embed
       │                  │             │
       └──────┬───────────┴─────────────┘
              │
              ▼
       Existing SSL backbone (deep_lob/, price_cycle/)
       + ssl_heads/short_term + ssl_heads/adaptive
              │
              ▼
       161-dim SSL embedding
              │
              ▼
       hybrid/model.py — HybridModel
       (135 + 161 + 32 = 328 → 256 hidden → heads)
              │
       ┌──────┼──────────┐
       ▼      ▼          ▼
    event   direction   confidence
       │      │          │
       └──────┴──────────┘
              │
              ▼
       hybrid/policy.py — decide()
       (combines with day_trade rules + adaptive head outputs)
              │
              ▼
       TradeDecision
       (take_trade, direction, size, tp_mult, sl_mult, reason)
```

## Public API contract

The **top-level** `from modules.trading_intel import X` is the supported
entry point. Sub-imports like `from modules.trading_intel.lob.features
import ...` work but are subject to refactor.

```python
from modules.trading_intel import (
    # LOB feature layer
    build_lob_tensor_v2_for_bar, N_LOB_CHANNELS_V2, HumanLOBCNN,
    # SSL trading heads
    ShortTermHeads, AdaptiveTargetHeads,
    # Fusion + decision
    HybridModel, HybridConfig, decide, TradeDecisionPolicy,
)
```

## Boundaries / what does NOT live here

| Concern | Lives in | Why not here |
|---|---|---|
| Day-trade rule labels (`event_flag`, `event_direction`) | `prepare_day_trading.py` (root) | The proven 78.3% baseline — touched only by Stage 1's 6-year re-run |
| SSL backbone (HierarchicalLOBTransformer) | `modules/deep_lob/` | Pretrained checkpoint compatibility |
| Price cycle model | `modules/price_cycle/` | Same — pretrained |
| Backtest engine | `self_supervised/backtest_day_trade.py` | Backtests both legacy and hybrid outputs |
| MBO order-tensor builder | `self_supervised/build_order_batches.py` | Pre-existing — `trading_intel` only consumes it |

This separation means `trading_intel/` can be **deleted entirely** and
the legacy pipeline still works at 78.3% hit rate. Trading intel is a
**pure addition**, not a replacement.

## How to use

### Inference (one decision)

```python
from modules.trading_intel import HybridModel, HybridConfig, decide

# Load trained models
model = HybridModel(HybridConfig(...))
model.load_state_dict(torch.load("best_hybrid.pt")["state_dict"])

# Forward pass (you supply the features)
out = model(daytrade_features, ssl_embedding, cnn_embedding)

# Combine with day_trade rule outputs + adaptive head outputs → decision
decision = decide(
    rule_event_flag=1, rule_event_direction=+1, rule_signal_quality=2,
    hybrid_event_prob=out.event_prob().item(),
    hybrid_p_long=out.direction_probs()[0, 0].item(),
    hybrid_p_short=out.direction_probs()[0, 1].item(),
    hybrid_p_neutral=out.direction_probs()[0, 2].item(),
    hybrid_confidence=out.confidence().item(),
    adaptive_max_r_pred=2.5, adaptive_bucket=2, adaptive_regime_risk=0,
)

if decision.take_trade:
    # decision.direction, decision.position_size_scale, decision.tp_mult, ...
    ...
```

### Training the hybrid (once embeddings exist)

```bash
python -m modules.trading_intel.training.train_hybrid \
    --features    combined_6y/day_trading_features.parquet \
    --ssl-embeddings  checkpoints/ssl_6y/embeddings/embeddings.npy \
    --output      checkpoints/hybrid_6y \
    --use-dataset-slice
```

## Tests

All tests live in `/tests/` at the repo root. Filenames map 1:1 to
submodules:

| Test file | Covers |
|---|---|
| `tests/test_lob_features_v2.py` | `lob.features` + `lob.cnn` |
| `tests/test_short_term_ssl.py` | `ssl_heads.short_term*` |
| `tests/test_adaptive_targets.py` | `ssl_heads.adaptive*` + `hybrid.policy` |
| `tests/test_hybrid_model.py` | `hybrid.model` |

Run them all:
```bash
python -m pytest tests/test_lob_features_v2.py tests/test_short_term_ssl.py \
                 tests/test_hybrid_model.py tests/test_adaptive_targets.py -v
```

84 tests passing as of the last refactor.
