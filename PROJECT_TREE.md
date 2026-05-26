# 📂 Project Structure

> **Last refactor:** `trading_intel/` consolidation
> **Baseline tag:** `v1-baseline-pre-cleanup` (commit `c7b88dd`)

## 🎯 ACTIVE PIPELINE

Two root-level entry points + cleanly-bounded packages.

### Root entry points
| File | Purpose |
|---|---|
| `prepare_day_trading.py` | Main pipeline — bars + features + rule labels |
| `regime_config.py` | Hand-tuned constants |

### Active packages
| Directory | Purpose |
|---|---|
| `modules/deep_lob/` | SSL backbone (HierarchicalLOBTransformer + heads) — pretrained, untouched |
| `modules/price_cycle/` | Price cycle SSL model — pretrained, untouched |
| **`modules/trading_intel/`** | **All new enhancements (this refactor)** — see `modules/trading_intel/README.md` |
| `modules/` (other) | Shared helpers: context_features, intrabar_microstructure, etc. |
| `self_supervised/` | SSL training + extraction + backtest scripts |
| `tests/` | Pytest suite (84 trading_intel tests + others) |
| `tools/` | Paper-trade utilities |
| `docs/` | Documentation |

### `modules/trading_intel/` — Single-package architecture
```
trading_intel/
├── __init__.py              ← Public API
├── README.md                ← Architecture doc + usage
├── lob/                     ← Enhanced LOB reading
│   ├── features.py          ← 13-channel LOB tensor
│   └── cnn.py               ← Human-style 4-layer CNN
├── ssl_heads/               ← Trading-aware SSL heads
│   ├── short_term.py        ← 5 micro-event heads
│   ├── short_term_labels.py
│   ├── adaptive.py          ← 3 adaptive-target heads
│   └── adaptive_labels.py
├── hybrid/                  ← Fusion + decision
│   ├── model.py             ← HybridModel MLP
│   └── policy.py            ← TradeDecisionPolicy
└── training/
    └── train_hybrid.py      ← End-to-end training
```

**Public API (the only import callers need):**
```python
from modules.trading_intel import (
    build_lob_tensor_v2_for_bar, HumanLOBCNN,
    ShortTermHeads, AdaptiveTargetHeads,
    HybridModel, HybridConfig, decide, TradeDecisionPolicy,
)
```

## 🗄️ ARCHIVED (`_archive/`)

87 .py files moved with `git mv` (history preserved). Nothing deleted.

| Subdir | What's there | Why archived |
|---|---|---|
| `_archive/legacy_v19/` | V19 backtest/train/predict/walkforward + stage1/2/3 + V19 tests | Old V19 pipeline |
| `_archive/diagnostics/` | 21 one-shot analysis scripts | Diagnostic, ran once |
| `_archive/unused_empty_dirs/` | discovery/, dl_pipeline/, simulators/, deployment/ | Only had `__init__.py` |
| `_archive/v19_2_dir/` | 20 .py older fork | Historical |
| `_archive/v20_dir/` | 2 .py aborted prototype | Aborted |
| `_archive/variants/` | prepare_day_trading_enriched.py | Experimental |
| `_archive/old_artifacts/` | Logs, samples, experiments | Generated artifacts |

## 🔁 Restoring an archived file
```bash
git mv _archive/legacy_v19/<file>.py ./<file>.py
git commit -m "restore: <file> for <reason>"
```

## ✅ Verified
- All `from modules.trading_intel import *` works
- All `modules.deep_lob`, `modules.price_cycle` imports work
- `prepare_day_trading.py` runs unchanged
- 84/84 tests passing
