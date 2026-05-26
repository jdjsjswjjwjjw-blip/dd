# 🏛️ Subsystem Architecture

The repository contains **3 independent subsystems** that communicate ONLY
through data files (parquet / npy), NEVER through code imports. Each can
fail or be rewritten without breaking the others.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       3 INDEPENDENT SUBSYSTEMS                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌──────────────────────┐
│ A. day_trade    │───▶│ B. SSL pipeline │───▶│ C. trading_intel     │
│ (rule-based)    │    │ (representation │    │ (hybrid fusion +     │
│                 │    │  learning)      │    │  decision policy)    │
│ 78.3% hit rate  │    │ embeddings      │    │ adaptive TP/risk     │
│ PROVEN          │    │ EXPERIMENTAL    │    │ EXPERIMENTAL         │
└─────────────────┘    └─────────────────┘    └──────────────────────┘
        ▲                       ▲                        ▲
        │                       │                        │
        │  parquet + npy        │  embeddings.npy        │
        │  outputs              │  + checkpoints         │
        └─────────── DATA FILES ONLY ─────────────────────┘
                  (no code imports between A, B, C)
```

## SUBSYSTEM A — day_trade (rule-based, 78.3% hit rate)

### Purpose
Convert raw MBO + MBP-10 ticks into 15-min bars + 135 hand-crafted features
+ rule-based labels (event_flag, event_direction, signal_quality,
path_outcome). This is the **proven baseline**.

### Entry point
```bash
python prepare_day_trading.py --mbo <path> --mbp <path> --output <dir>
```

### Code locations
| File | Role |
|---|---|
| `prepare_day_trading.py` | Main script (~5000 lines, monolithic by design) |
| `regime_config.py` | Hand-tuned constants (REGIME_TP_SL, REGIME_MAX_BARS) |
| `modules/context_features.py` | Helper |
| `modules/intrabar_mbp_microstructure.py` | Helper |
| `modules/tick_intrabar_slices.py` | Helper |
| `modules/manifest_v19.py` | Helper |

### Outputs (the data contract)
```
<output_dir>/
├── day_trading_features.parquet     # 135 cols + labels + dataset_slice
├── lob_tensors.npy                   # (N, T=50, P=20, C=9)
├── lob_tensor_timestamps.npy         # (N,)
├── day_trading_manifest.json         # metadata
└── artifact_manifest.json
```

### What it does NOT depend on
- ❌ No imports from `modules/deep_lob/`
- ❌ No imports from `modules/price_cycle/`
- ❌ No imports from `modules/trading_intel/`
- ❌ No imports from `self_supervised/`

### Failure mode
If `prepare_day_trading.py` is broken, **subsystem A fails alone**. B and C
cannot start (they need its output), but the failure is contained and the
fix is local to the rule pipeline.

---

## SUBSYSTEM B — SSL pipeline (representation learning)

### Purpose
Take subsystem A's outputs and learn a 161-dim embedding per bar via
self-supervised pretext tasks (next_price, next_volatility, masked LOB,
etc.). Adds the new trading-aware heads (short_term, adaptive) from
`trading_intel/ssl_heads/`.

### Entry points
```bash
# Pretrain
python self_supervised/pretrain_lob.py --features <...> --lob-tensors <...>
python self_supervised/pretrain_cycle.py --features <...> --lob-tensors <...>

# Extract embeddings
python self_supervised/extract_embeddings.py --features <...> \
    --lob-checkpoint <...> --cycle-checkpoint <...>

# OR: end-to-end pipeline
./self_supervised/run_ssl_only.sh <combined_dir> <output_dir>
```

### Code locations
| Directory | Role |
|---|---|
| `self_supervised/*.py` | Training scripts (pretrain, extract, etc.) |
| `modules/deep_lob/` | HierarchicalLOBTransformer + existing heads |
| `modules/price_cycle/` | Price cycle model |
| `modules/trading_intel/ssl_heads/` | Trading-aware heads (NEW) |

### Outputs (the data contract)
```
<ssl_output>/
├── lob/best_ssl_lob.pt                 # LOB transformer checkpoint
├── cycle/best_ssl_cycle.pt             # Price cycle checkpoint
├── embeddings/embeddings.npy           # (N, 161)
├── embeddings/embedding_valid.npy      # (N,) bool mask
└── validation_report.json
```

### What it depends on
- ✅ Subsystem A's outputs (parquet + npy)
- ✅ `modules/deep_lob/` (own architecture)
- ✅ `modules/price_cycle/` (own architecture)
- ✅ `modules/trading_intel/ssl_heads/` for the new heads

### What it does NOT depend on
- ❌ No imports from `prepare_day_trading.py`
- ❌ No imports from `modules/trading_intel/hybrid/`
- ❌ No imports from `modules/trading_intel/lob/` (uses raw lob_tensors)
- ❌ `regime_config.py` referenced only for label names (no logic shared)

### Failure mode
If pretraining fails or embeddings are unusable, **subsystem C cannot
produce hybrid predictions**, but **subsystem A continues to work alone**
(rule-based trading at 78.3% hit rate).

---

## SUBSYSTEM C — trading_intel (hybrid fusion + decision policy)

### Purpose
Fuse subsystem A's hand-crafted features + subsystem B's learned
embeddings (+ optional LOB CNN) into a single trading decision with
adaptive TP / regime-aware sizing.

### Entry point
```bash
python -m modules.trading_intel.training.train_hybrid \
    --features <A's parquet> \
    --ssl-embeddings <B's embeddings.npy> \
    --output <hybrid_output>
```

### Code locations
```
modules/trading_intel/
├── lob/         # Enhanced 13-channel LOB + CNN
├── ssl_heads/   # Trading-aware SSL heads (used by subsystem B during retrain)
├── hybrid/      # Fusion model + decision policy
└── training/    # train_hybrid.py
```

See `modules/trading_intel/README.md` for full details.

### Outputs (the data contract)
```
<hybrid_output>/
├── best_hybrid.pt                  # HybridModel checkpoint + norms
├── predictions.parquet             # per-bar event_prob, direction_probs,
│                                   # confidence
└── training_report.json
```

### What it depends on
- ✅ Subsystem A's outputs (`day_trading_features.parquet`)
- ✅ Subsystem B's outputs (`embeddings.npy`, `embedding_valid.npy`)
- ✅ Its own helper modules (`trading_intel/lob/`, `trading_intel/hybrid/`)

### What it does NOT depend on
- ❌ No imports from `prepare_day_trading.py`
- ❌ Reads parquet + npy only, never rule-engine code

### Failure mode
If hybrid fails or degrades: **switch off trading_intel and run subsystem A
alone** at proven 78.3% hit rate. No data corruption — A and B outputs are
untouched. Trading continues without ML enhancement.

---

## Data Contract — the formal interface between subsystems

### A → B
```python
# day_trading_features.parquet schema (key columns)
{
    'ts_event': datetime64[us],
    'open', 'high', 'low', 'close': float,
    'volume': int,
    'atr_14': float,
    'event_flag': int8,         # 0/1
    'event_direction': int8,     # -1/0/+1
    'event_score': float32,
    'signal_quality': int8,      # 0/1/2
    'path_outcome': int8,        # 0=long_tp, 1=short_tp, 2=long_sl, ...
    'regime_label': str,         # trending/ranging/volatile
    'dataset_slice': str,        # train/holdout
    # ... +120 hand-crafted features
}

# lob_tensors.npy shape: (N_bars, T=50, P=20, C=9)
# lob_tensor_timestamps.npy shape: (N_bars,) datetime64[us]
```

### B → C
```python
# embeddings.npy shape: (N_bars, 161)  float32
# embedding_valid.npy shape: (N_bars,)  bool
# Aligned 1:1 with A's day_trading_features.parquet
```

### C → trading_execution (future)
```python
# predictions.parquet schema
{
    'ts_event': datetime64[us],
    'emb_valid': bool,
    'event_prob': float32,
    'p_long', 'p_short', 'p_neutral': float32,
    'confidence': float32,
}
```

---

## Deployment scenarios — proves the separation

### Scenario 1: Subsystem A alone (production today)
```bash
# Pure rule-based trading at 78.3% hit rate
python prepare_day_trading.py --mbo ... --mbp ... --output features/
python self_supervised/backtest_day_trade.py --features features/... --strict
# → No SSL, no trading_intel, no hybrid. Works standalone.
```

### Scenario 2: A + B (SSL embeddings as features, no hybrid)
```bash
python prepare_day_trading.py --output features/
./self_supervised/run_ssl_only.sh features/ ssl_checkpoints/
# → A's labels + B's embeddings, but no fusion model. Still works.
```

### Scenario 3: Full stack (A + B + C)
```bash
python prepare_day_trading.py --output features/
./self_supervised/run_ssl_only.sh features/ ssl_checkpoints/
python -m modules.trading_intel.training.train_hybrid \
    --features features/day_trading_features.parquet \
    --ssl-embeddings ssl_checkpoints/embeddings/embeddings.npy \
    --output hybrid_checkpoints/
# → Full enhanced trading decisions
```

### Scenario 4: A + C without B (NOT supported)
This is intentionally NOT supported. trading_intel REQUIRES SSL embeddings
as input. If you want pure rules + ML on hand-features, use a different
classifier (xgboost, etc.) directly on A's output.

---

## Boundaries enforcement

### Automatic lint check
```bash
python tools/check_subsystem_boundaries.py
```
Returns exit code 0 if clean, 1 if violations found. Run before every commit
(or wire into pre-commit). Also exercised by `tests/test_subsystem_boundaries.py`.

### Forbidden import rules

```python
# ❌ FORBIDDEN — caught by lint
prepare_day_trading.py imports modules.deep_lob.*  (except shared helpers, see below)
prepare_day_trading.py imports modules.trading_intel.*
modules/deep_lob/ imports prepare_day_trading
modules/deep_lob/ imports modules.trading_intel
modules/trading_intel/ imports prepare_day_trading
self_supervised/ imports prepare_day_trading
```

### Shared-helper allowlist (legitimate cross-subsystem imports)

Some modules contain PURE feature-engineering utilities that don't belong
exclusively to any subsystem. They're allowlisted in
`tools/check_subsystem_boundaries.py`:

| Allowed import | Used by | Why it's a helper, not coupling |
|---|---|---|
| `modules.price_cycle.data_structures` (BarSequence) | day_trade, SSL | Pure numpy container — no SSL state |
| `modules.price_cycle.feature_pipeline` (build_cycle_features) | day_trade, SSL | Pure feature engineering (Wyckoff + swing + fractal), causal, no training code |

These give day_trade 12 structural columns (HH/HL/LH/LL pattern, swing
position, fractal score, cycle position, cycle_hurst) without crossing
into SSL training territory.

### Adding a new allowed cross-import

If a new shared helper is needed:
1. Identify the importer and importee
2. Verify the importee has NO training/ML state (just pure functions or
   data classes)
3. Add the importee to `SHARED_HELPER_ALLOWLIST` in
   `tools/check_subsystem_boundaries.py`
4. Document why in the inline comment + this README

If the importee DOES carry training state, it doesn't belong in the
allowlist. Either factor out the pure helpers, or rethink the design.

---

## When to extend / when to fork

| Want to... | Extend in... |
|---|---|
| Add a new rule feature for day_trade | `prepare_day_trading.py` + helpers |
| Add a new SSL pretext task | `modules/trading_intel/ssl_heads/` |
| Add a new LOB channel | `modules/trading_intel/lob/features.py` |
| Add a new trade decision rule | `modules/trading_intel/hybrid/policy.py` |
| Try a totally new approach | Create `modules/<new_subsystem>/` and define its own data contract |

The boundary is the data contract. Code never crosses subsystems — only
data does.
