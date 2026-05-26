# QuantSystem — 6B Day-Trading Pipeline

A quantitative trading system for **6B (GBP/USD futures)** built around **three
independent subsystems**: a hand-crafted rule pipeline that already delivers
**78.3 % strict-holdout hit rate**, an experimental SSL representation
layer, and a hybrid fusion layer that combines both with adaptive TP / regime-
aware sizing.

> **Branch:** `claude/task-d-RcDhu`
> **Baseline tag:** `v1-baseline-pre-cleanup` (commit `c7b88dd`)
> **Architecture authority:** [`SUBSYSTEMS.md`](SUBSYSTEMS.md)
> **Tests:** 88 passing (84 trading_intel + 4 boundary lint)

---

## 📐 The three subsystems

The repo enforces clear separation. Each subsystem can fail or be rewritten
without breaking the others — they only communicate through data files.

```
┌─────────────────┐    ┌─────────────────┐    ┌──────────────────────┐
│ A. day_trade    │───▶│ B. SSL pipeline │───▶│ C. trading_intel     │
│ (rule-based)    │    │ (representation │    │ (hybrid fusion +     │
│                 │    │  learning)      │    │  decision policy)    │
│ 78.3% hit rate  │    │ embeddings      │    │ adaptive TP/risk     │
│ PROVEN          │    │ EXPERIMENTAL    │    │ EXPERIMENTAL         │
└─────────────────┘    └─────────────────┘    └──────────────────────┘
        │ parquet+npy           │ embeddings.npy        │
        └────── DATA FILES ONLY (no code imports) ──────┘
```

| Subsystem | Status | Entry point | What it does |
|---|---|---|---|
| **A. day_trade** | ✅ Production (78.3% hit rate on holdout) | `prepare_day_trading.py` | Raw MBO/MBP-10 → bars + 135 features + rule labels |
| **B. SSL pipeline** | ⚠️ Experimental (works at 6-mo scale, expected to shine at 6-yr scale) | `self_supervised/pretrain_lob.py` etc. | Pretrains transformer + cycle models → 161-dim embeddings |
| **C. trading_intel** | ⚠️ Experimental (built, awaiting full-scale training) | `python -m modules.trading_intel.training.train_hybrid` | Fuses A + B + adaptive heads → trade decisions |

Boundary enforcement: `python tools/check_subsystem_boundaries.py` (also
run as a test in `tests/test_subsystem_boundaries.py`).

---

## 🚀 Quick start

### Scenario 1 — Run the proven baseline (no ML)

```bash
# 1. Build features + labels from raw MBO/MBP
python prepare_day_trading.py \
    --mbo /path/to/6BH5.mbo.parquet \
    --mbp /path/to/6BH5.mbp10.parquet \
    --output q1_6BH5 \
    --freq 15min

# 2. Backtest with strict anti-leakage mode (holdout only)
python self_supervised/backtest_day_trade.py \
    --features q1_6BH5/day_trading_features.parquet \
    --output backtests/baseline_strict \
    --strict
```

Expected output: ~78 % hit rate, ~2-3 % annualized return on the holdout slice
with 1-bp RT cost. See [`SUBSYSTEMS.md § Subsystem A`](SUBSYSTEMS.md) for the
data contract.

### Scenario 2 — Add the SSL embedding layer

```bash
# After running prepare_day_trading.py for ALL quarters you have:
# 3. Build order tensors (MBO → per-bar order batches) per quarter
python self_supervised/build_order_batches.py \
    --mbo Q1.mbo --features q1_6BH5/day_trading_features.parquet \
    --output q1_6BH5 --freq 15min --lookback-bars 50 --n-orders 200

# 4. Combine quarters into one chronological dataset (handles rolls)
python self_supervised/combine_quarter_artifacts.py \
    --quarter-dirs q1_6BH5 q2_6BM5 \
    --output-dir combined_6m \
    --overlap-mode auto

# 5. Run the full SSL pipeline (pretrain LOB + cycle + extract embeddings)
./self_supervised/run_ssl_only.sh combined_6m checkpoints/ssl_6m 0.75
```

Outputs land in `checkpoints/ssl_6m/embeddings/embeddings.npy` (161-dim per
bar) + `embedding_valid.npy` mask.

### Scenario 3 — Train the hybrid model

```bash
# 6. Train HybridModel on (day_trade features + SSL embeddings)
python -m modules.trading_intel.training.train_hybrid \
    --features    combined_6m/day_trading_features.parquet \
    --ssl-embeddings  checkpoints/ssl_6m/embeddings/embeddings.npy \
    --embedding-valid checkpoints/ssl_6m/embeddings/embedding_valid.npy \
    --output      checkpoints/hybrid_6m \
    --use-dataset-slice \
    --epochs 80 --hidden-dim 256 --n-layers 3
```

Then route the predictions through `modules.trading_intel.hybrid.policy.decide()`
to get the final `TradeDecision` (adaptive TP, regime-aware size, reason).

---

## 🗂️ Repository structure

```
/dd
├── prepare_day_trading.py          ← Subsystem A entry point (rule pipeline)
├── regime_config.py                ← Hand-tuned constants (REGIME_TP_SL, etc.)
│
├── SUBSYSTEMS.md                   ← 🏛️ Architecture authority — READ FIRST
├── PROJECT_TREE.md                 ← Layout + restore instructions
├── README.md                       ← This file
├── CHANGELOG.md                    ← Release notes
├── AGENTS.md                       ← Agent runbook
│
├── modules/
│   ├── trading_intel/              ← Subsystem C: hybrid + new SSL heads
│   │   ├── __init__.py             ← Public API
│   │   ├── README.md               ← Detailed architecture + data flow
│   │   ├── lob/                    ← 13-channel LOB features + CNN
│   │   ├── ssl_heads/              ← Short-term + adaptive trading heads
│   │   ├── hybrid/                 ← Fusion model + decision policy
│   │   └── training/               ← train_hybrid.py
│   │
│   ├── deep_lob/                   ← Subsystem B: LOB transformer backbone
│   ├── price_cycle/                ← Subsystem B: price cycle model
│   ├── context_features.py         ← Shared helpers used by A
│   ├── intrabar_mbp_microstructure.py
│   ├── tick_intrabar_slices.py
│   └── ... (~100 helper modules)
│
├── self_supervised/                ← Subsystem B + backtest scripts
│   ├── pretrain_lob.py
│   ├── pretrain_cycle.py
│   ├── extract_embeddings.py
│   ├── combine_quarter_artifacts.py
│   ├── build_order_batches.py
│   ├── backtest_day_trade.py       ← --strict backtest (works on A's output)
│   ├── integrate_with_day_trade.py ← SSL × rule integration analysis
│   ├── run_ssl_only.sh             ← End-to-end SSL pipeline
│   └── ... (other helpers)
│
├── tools/
│   ├── check_subsystem_boundaries.py  ← Boundary lint
│   ├── paper_dry_run.py
│   └── backtest_smoke.py
│
├── tests/                           ← 88+ tests
│   ├── test_lob_features_v2.py     ← LOB layer (23 tests)
│   ├── test_short_term_ssl.py      ← Short-term heads (19 tests)
│   ├── test_hybrid_model.py        ← Hybrid fusion (16 tests)
│   ├── test_adaptive_targets.py    ← Adaptive heads + policy (26 tests)
│   ├── test_subsystem_boundaries.py ← Boundary lint (4 tests)
│   └── ... (other existing tests)
│
└── _archive/                        ← 87 archived files (legacy V19, diagnostics)
                                       Nothing deleted — restore with `git mv`
```

---

## 🧪 Tests

### Run the new subsystem tests
```bash
python -m pytest tests/test_lob_features_v2.py tests/test_short_term_ssl.py \
                 tests/test_hybrid_model.py tests/test_adaptive_targets.py \
                 tests/test_subsystem_boundaries.py -v
```

| Suite | Tests | Covers |
|---|---|---|
| `test_lob_features_v2.py` | 23 | 13-channel LOB tensor + HumanLOBCNN |
| `test_short_term_ssl.py` | 19 | 5 short-term SSL heads + causal label builders |
| `test_hybrid_model.py` | 16 | HybridModel fusion + per-head losses |
| `test_adaptive_targets.py` | 26 | Adaptive heads + decision policy + regression fixes |
| `test_subsystem_boundaries.py` | 4 | Architecture lint (positive + negative tests) |

### Boundary check
```bash
python tools/check_subsystem_boundaries.py
```
Returns exit code 0 (clean) or 1 (violations). Verifies that no active code
crosses subsystem lines (e.g., day_trade importing SSL training code).

---

## 🏗️ Development phases (the journey to current state)

The codebase went through 5 development phases. Every phase is a documented
commit on `claude/task-d-RcDhu`:

| Commit | Phase | What changed |
|---|---|---|
| `c7b88dd` | **v1 baseline** (tagged) | Working day_trade + SSL pipeline, 78.3% strict hit rate |
| `88e2f29` | **Phase 0: Cleanup** | 87 legacy files moved to `_archive/`, all imports verified |
| `a6eb872` | **Phase 2: LOB v2** | 13-channel LOB tensor + Human-style CNN (118k params) |
| `b9f763d` | **Phase 3: SSL heads** | 5 short-term SSL heads + causal label builders |
| `baf3b12` | **Phase 4: Hybrid** | Fusion MLP (day_trade + SSL + optional CNN) |
| `e91b5a8` | **Phase 4b: Adaptive** | TP-adaptive heads + regime-aware decision policy |
| `3d5cabb` | **Review fixes** | 6 HIGH/MEDIUM findings from code review addressed |
| `fb1ad5d` | **Refactor** | Consolidate new code into `modules/trading_intel/` package |
| `64abe04` | **Subsystem docs + lint** | SUBSYSTEMS.md + boundary lint + 4 enforcement tests |

> **Phase 1** (proposal's "isolate baseline + 6-year evaluation") is the next
> data-dependent step. It's blocked on 6-year MBO/MBP-10 from Databento.

---

## 📊 Honest performance summary

### Subsystem A (rule pipeline) on 6-month holdout
| Metric | Value |
|---|---|
| Hit rate (strict, holdout-only) | **78.3 %** |
| Annualized return @ 1-bp RT | ~+2.9 % |
| Annualized return @ 8-bp RT (conservative) | ~+1.5 % |
| Max drawdown | -0.14 % |
| Trades / month | ~13 |
| Holdout sample size | 46 trades — **statistically thin** (CI ±12 %) |

### Subsystem B (SSL pipeline) on 6-month data
| Metric | Value |
|---|---|
| Pretraining val loss | LOB 9.8 → 0.65, Cycle 3.6 → 2.4 |
| Embedding coverage | 82.6 % of bars valid |
| Direction head | ❌ Collapsed (predicted UP for 100 % of val) |
| Magnitude head rank-IC | **+0.25** (real signal, but mostly riding day_trade events) |

### Subsystem C (hybrid)
Built and tested but **not yet trained on 6-year data**. With 6-month data the
SSL bottleneck propagates and the hybrid offers marginal lift. The proposal's
Stage 1 (6-year baseline) is the path to validate whether the hybrid actually
adds value at scale.

---

## ⚠️ Known limitations

1. **6-month dataset is statistically thin** for SSL — 5,478 training bars
   for a 290k-parameter transformer is ~19 samples/parameter (10-100x is the
   convention).
2. **Hit-rate metrics on 46 holdout trades have wide CIs** (±12 % at 95 %).
   The 78.3 % is real signal, but the Sharpe needs hundreds of trades to be
   meaningful.
3. **SSL embeddings encode the SAME patterns** day_trade rules detect, not
   independent signal — verified empirically (SSL-only on non-event bars =
   52.4 % hit ≈ random).
4. **The hybrid model overlap** is partially redundant with day_trade rules.
   Real lift expected only with 6-year data.
5. **trading_intel.lob.features (v2)** is built but NOT yet wired into
   `prepare_day_trading.py` — this is intentional (preserves the 78.3 %
   baseline). It plugs in during Stage 1 re-run.

---

## 🗄️ Archive policy

Nothing in `_archive/` is deleted. Any file can be restored with:

```bash
git mv _archive/legacy_v19/<file>.py ./<file>.py
git commit -m "restore: <file> for <reason>"
```

Git history is preserved on every archived file. The `v1-baseline-pre-cleanup`
tag points to the exact state before any cleanup.

---

## 📚 Documentation index

| File | Purpose |
|---|---|
| [`README.md`](README.md) | This file — high-level overview + quick start |
| [`SUBSYSTEMS.md`](SUBSYSTEMS.md) | **Authoritative** subsystem architecture, data contracts, boundaries |
| [`PROJECT_TREE.md`](PROJECT_TREE.md) | Repository layout + archive restore instructions |
| [`modules/trading_intel/README.md`](modules/trading_intel/README.md) | trading_intel package detail + data-flow diagram |
| [`CHANGELOG.md`](CHANGELOG.md) | Release notes |
| [`AGENTS.md`](AGENTS.md) | Agent runbook (for automation users) |

When in doubt: **SUBSYSTEMS.md is the source of truth** for what depends on
what. Read it before adding a new module.

---

## 🤝 Contribution rules

1. **Don't touch the 78.3 % baseline** without explicit approval.
   `prepare_day_trading.py` and its direct helpers are frozen.
2. **New code goes in `modules/trading_intel/`** in the appropriate subpackage.
   The `__init__.py` files re-export the public API — keep them up to date.
3. **Run `python tools/check_subsystem_boundaries.py`** before committing.
   The lint is exit-code 0 or your CI breaks.
4. **Add a test** for every new feature. The current pattern: one test file
   per top-level subpackage of `trading_intel`.
5. **Document** new shared helpers in `SUBSYSTEMS.md` if they cross
   subsystems.

---

## 🛣️ What's next

The proposal's Stage 1-4 plan is implemented as code but data-dependent. To
move forward:

1. **Stage 1**: Re-run `prepare_day_trading.py` on 6 years (2019-2024) of
   6B data; combine 24 quarters with `combine_quarter_artifacts.py`.
2. **Stage 2**: Wire `modules/trading_intel/lob/features.py` (13-channel)
   into the LOB tensor builder of `prepare_day_trading.py` for the 6-year
   re-run.
3. **Stage 3**: Pretrain SSL on the 6-year combined dataset with the new
   short-term heads enabled.
4. **Stage 4**: Train HybridModel + run walk-forward backtest across years.

Each stage is independent — partial completion still produces useful
artifacts (Stage 1 alone re-validates the 78.3 % baseline; Stage 3 gives
better embeddings even without the hybrid).
