# QuantSystem — 6B Day-Trading Pipeline

A quantitative trading system for **6B (GBP/USD futures)** built around **three
independent subsystems**: a hand-crafted rule pipeline that already delivers
**78.3 % strict-holdout hit rate**, an experimental SSL representation
layer, and a hybrid fusion layer that combines both with adaptive TP / regime-
aware sizing.

> **Branch:** `claude/task-d-RcDhu`
> **Baseline tag:** `v1-baseline-pre-cleanup` (commit `c7b88dd`)
> **Architecture authority:** [`SUBSYSTEMS.md`](SUBSYSTEMS.md)
> **Tests:** 320 passing + 2 skipped, 0 failures (88 trading_intel + 4 boundary + 230 legacy day_trade/SSL backbone)

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

## 🧠 SSL Subsystem — Architecture & Anti-Collapse

SSL (Subsystem B) is the experimental representation-learning layer that
sits between day_trade's rule labels and trading_intel's hybrid decisions.
This section documents what's inside it and the four research-driven
counter-measures we've added to address the failures observed in our
6-month run.

### Encoder backbones (~743k params total)

```
                  bars + LOB + order tensors          context features
                              │                                │
                              ▼                                ▼
    ┌──────────────────────────────────┐    ┌──────────────────────────────┐
    │ HierarchicalLOBTransformer       │    │ PriceCycleModel              │
    │ (modules/deep_lob/)              │    │ (modules/price_cycle/)       │
    │                                   │    │                              │
    │  • OrderEmbedder    (per-order)  │    │  • CycleEncoder              │
    │  • OrderTransformer (per-bar)    │    │  • FractalFeatures           │
    │  • EventAggregator               │    │  • SwingAnalysis             │
    │  • BarLSTM          (temporal)   │    │  • PhaseClassifier           │
    │  • LOBImageEncoder  (optional)   │    │  • MultiScaleFusion          │
    │  • ContextEncoder                │    │                              │
    │  • Fusion + MultiTaskHeads       │    │  ~450k parameters            │
    │                                   │    │                              │
    │  ~293k parameters                 │    │                              │
    └──────────────────────────────────┘    └──────────────────────────────┘
                              │                                │
                              ▼                                ▼
                        shared_emb (64-dim)             cycle_emb (64-dim)
                              │                                │
                              └────────────┬───────────────────┘
                                           ▼
                            embeddings.npy  (N, 161)
                            = LOB(64) + Cycle(64) + seasonal/cycle (33)
```

### SSL heads (training-time only)

| Family | Module | Heads | What they predict |
|---|---|---|---|
| **Macro** (existing) | `deep_lob.multi_task_heads` | 6 (direction, next_price, next_imbalance, next_volatility, next_regime, wall_persist, time_to_event) | Long-horizon dynamics |
| **Short-term** (new) | `trading_intel.ssl_heads.short_term` | 5 (wall_break, imbalance_shift, micro_target, liquidity_sweep, gap_fill) | 1-4 bar trading events |
| **Adaptive** (new) | `trading_intel.ssl_heads.adaptive` | 3 (max_R_reached, target_bucket, regime_risk) | Trade sizing + TP scaling |

All heads attach to the shared backbone embedding and produce causal
labels (no lookahead). See `modules/trading_intel/ssl_heads/__init__.py`
for the public API.

### Anti-collapse modules (`modules/trading_intel/anti_collapse/`)

Each addresses a specific failure mechanism observed in the 6-month run.
All four are opt-in via CLI flags or environment variables — default
training behavior is unchanged.

| Module | Mechanism solved | Theory source | Wired via |
|---|---|---|---|
| `SimplexETFClassifier` + `dot_regression_loss` | #2 Class collapse (direction head → 100% UP) | Papyan-Han-Donoho 2020 "Neural Collapse"; Yang et al. 2022 "AllNC" | `--use-simplex-etf` |
| `OrthogonalRepresentationModule` | #5 Rule overlap (SSL on non-event bars = 52.4%, riding day_trade events) | Chinese OMoE; Stiefel-manifold projection | `--ortho-weight W` |
| `DBMTLBalancer` (LSB + GNB) | #7 Multi-task tradeoff (v1: magnitude wins; v2: direction wins; never both) | Lin et al. 2023 "DB-MTL"; Chen et al. 2018 "GradNorm" | `--use-dbmtl` |
| `SharpeRegularizer` + `differentiable_sharpe_loss` | #1 Loss-landscape mismatch (surrogate ≠ trading goal) | Moody-Saffell 2001; Donti et al. 2017 "Decision-Focused Learning" | `--sharpe-weight W` |

### Coverage of the seven failure mechanisms

| # | Mechanism | Direct evidence (6-mo run) | Code-level solution | Status |
|---|---|---|---|---|
| 1 | Loss-landscape mismatch | — (structural) | `SharpeRegularizer` | ✅ wired |
| 2 | Class collapse | direction head: 100% UP | `SimplexETFClassifier` + `dot_regression_loss` | ✅ wired |
| 3 | Horizon mismatch | — (structural) | `ShortTermHeads` (5 micro-horizon heads) | ✅ wired |
| 4 | Direction vs magnitude | v1 rank-IC=0.25 mag, dir collapsed | `AdaptiveTargetHeads` + Sharpe | ✅ wired |
| 5 | Rule overlap | SSL on non-event: 52.4% (random) | `OrthogonalRepresentationModule` | ✅ wired |
| 6 | Sample bottleneck | 5,478 train samples, 290k params | 5-year data scale (2021-2025) | ⏳ structural |
| 7 | MTL tradeoff | β=5 magnitude wins; α=5 direction wins | `DBMTLBalancer` (LSB + GNB) | ✅ wired |

**6 of 7 mechanisms have operational code-level solutions.** Mechanism #6
resolves at data scale, not in code.

### Running SSL with the anti-collapse modules

#### Default (no anti-collapse — baseline behavior):
```bash
./scripts/walk_forward.sh /raw/data /out 2021-01 2025-12 6B
```

#### With all four counter-measures enabled:
```bash
WF_USE_SIMPLEX_ETF=1 \
WF_ORTHO_WEIGHT=0.5 \
WF_USE_DBMTL=1 \
WF_SHARPE_WEIGHT=0.3 \
  ./scripts/walk_forward.sh /raw/data /out 2021-01 2025-12 6B
```

The script auto-detects these env vars and switches to the enhanced fold
runner (`tools/run_walk_forward_fold_enhanced.py`). Any subset works:
e.g., `WF_USE_SIMPLEX_ETF=1 WF_SHARPE_WEIGHT=0.3 ...` enables just ETF + Sharpe.

#### As a library (in your own training script):
```python
from modules.trading_intel.anti_collapse import (
    SimplexETFClassifier, dot_regression_loss,
    OrthogonalRepresentationModule,
    DBMTLBalancer, DBMTLConfig,
    SharpeRegularizer, SharpeLossConfig,
)
from modules.trading_intel.hybrid.model import HybridModel, HybridConfig

# Replace the direction head with a Simplex ETF (0 trainable params)
etf_head = SimplexETFClassifier(SimplexETFConfig(
    feature_dim=128, num_classes=3,
))
model = HybridModel(HybridConfig(hidden_dim=128), direction_head=etf_head)

# Anti-overlap penalty against rule features
ortho = OrthogonalRepresentationModule(weight=0.5)

# DB-MTL gradient balancer across event/direction/confidence/sharpe
balancer = DBMTLBalancer(["event", "direction", "confidence", "sharpe"])

# Sharpe regularizer (Western Decision-Focused Learning)
sharpe = SharpeRegularizer(weight=0.3)
```

### Empirical guarantees (from the test suite)

Each anti-collapse module ships with empirical proofs in
`tests/test_anti_collapse.py`, `tests/test_dbmtl_and_sharpe.py`, and
`tests/test_enhanced_fold_integration.py` (39 tests total):

- **Simplex ETF** — `test_resists_majority_collapse`: 200 gradient steps
  on 180/15/5 imbalanced 3-class data, **minority-class accuracy > 50%**
  (would be 0 under standard CE).
- **Orthogonal Rep** — `test_perfectly_correlated_high_penalty`: penalty
  on duplicated-feature embedding is **>5× higher** than on truly
  uncorrelated; `test_gradient_flows_into_embedding_only`: rule features
  receive zero gradient (detached).
- **DB-MTL** — `test_gradient_norm_balancing_lifts_quiet_tasks`: a head
  initialized with 0.001× weights gets a **larger task weight** than the
  loud head (the quiet task is "lifted", not muted).
- **Sharpe** — `test_optimization_improves_sharpe`: 300 gradient steps
  on a synthetic problem **improves empirical Sharpe by > 0.3** vs the
  random initialization.

### Masked-reconstruction auxiliary task (Path B)

The existing six forecasting heads (next_price, next_imbalance,
next_volatility, next_regime, wall_persist, time_to_event) give the
model a forward-looking signal but never force it to learn what's
*consistent with the surrounding context*. `modules/deep_lob/masked_modeling/`
adds that signal: random positions in the order-features tensor are
zeroed and a small head reconstructs them from the encoder output. The
two paradigms are complementary, not exclusive — masked reconstruction
is an *additional* auxiliary task, not a replacement.

| Piece | What it does |
|---|---|
| `MaskedModelingConfig` | All knobs in one frozen dataclass; off by default |
| `generate_mask(order_masks, cfg)` | Seedable, padding-aware, per-sample mask floor |
| `apply_mask(features, mask)` | Zero-out + optional mask-indicator channel |
| `ReconstructionHead` | Two-layer MLP (encoder_dim → hidden → feature_dim) |
| `compute_recon_loss` | Padding-aware MSE on masked-AND-valid positions |

Three masking strategies cover the documented failure modes:

```
random  Bernoulli(mask_ratio) per order — BERT-style baseline
patch   Contiguous patches of `patch_size` orders — mirrors tape outages
bar     Whole bars masked — strongest bar-level context test
```

Integration recipe for `pretrain_lob.py`:

```python
from modules.deep_lob.masked_modeling import (
    MaskedModelingConfig, generate_mask, apply_mask,
    ReconstructionHead, compute_recon_loss,
)

cfg = MaskedModelingConfig(enabled=True, mask_ratio=0.15)
head = ReconstructionHead(
    encoder_dim=hidden_dim, feature_dim=ORDER_FEATURE_DIM,
    hidden_dim=cfg.reconstruction_hidden_dim,
)
# In the training step:
mask = generate_mask(order_masks, cfg)
masked = apply_mask(order_features, mask, cfg.provide_mask_indicator)
encoded = backbone.encode(masked, order_masks, ...)
recon = head(encoded)
l_recon = compute_recon_loss(recon, order_features, mask, order_masks)
total_loss = existing_total + cfg.reconstruction_weight * l_recon
```

### Operational hardening for SSL

The anti-collapse modules above are necessary but not sufficient. Four
additional layers were added so the SSL pipeline survives contact with
reality (live execution, noisy feature inputs, market regime shifts):

| Layer | Module / tool | What it solves | Tests |
|---|---|---|---|
| **Normalization drift** | `trading_intel/hybrid/inference.py` — `FrozenScaler`, `load_hybrid_for_inference`, `predict_one` | The "transform with train stats — never `fit_transform`" rule, enforced in code. Loading a trained hybrid for live inference cannot accidentally re-fit scalers. | `test_inference_helpers.py` (10) |
| **Input redundancy** | `tools/audit_feature_redundancy.py` + `modules/trading_intel/training/feature_selection.py` | Empirical correlation-cluster connected-components auditor; in our codebase it found 14 CVD / 12 ATR / 7 imbalance variants. Output `drop_list` is consumed by both fold runners via `--drop-features-from-audit`. | `test_feature_redundancy_audit.py` (14) + `test_feature_selection.py` (14) |
| **Execution stress** | `tools/diagnostics/stress_test_backtest.py` | Re-runs the strict backtest under 5 progressively worse latency + slippage scenarios (baseline → mild → moderate → severe → extreme). Classifies the strategy as ROBUST / ACCEPTABLE / FRAGILE / BROKEN. | `test_stress_test.py` (13) |
| **Regime sensitivity** | `tools/diagnostics/regime_parity_test.py` | Splits `trades.csv` by categorical regime AND volatility quartile, computes per-subset Sharpe/PF/DD, verdicts ROBUST / ACCEPTABLE / UNSTABLE / REGIME_BIAS / INSUFFICIENT_DATA. Catches "one-regime trick" strategies that look profitable in aggregate. | `test_regime_parity.py` (19) |
| **Label-gate health** | `tools/diagnostics/audit_event_gate.py` + auto-hook in `prepare_day_trading.py` | Audits the upstream event gate that decides which bars get labeled (everything else is silently stamped NEUTRAL). Reports overall rate vs the 20–30 % design target, per-regime + per-session rates, component failure breakdown, `cvd_direction_pct` `fillna(0.5)` contamination, and warm-up zero-bias from `_zscore` min_periods. Verdicts HEALTHY / LOW_RATE / STARVED / WARMUP_HEAVY / COMPONENT_DOMINATED / REGIME_STARVED. **The pipeline runs this automatically after writing the features parquet** and surfaces the verdict inline — no separate command needed. | `test_event_gate_audit.py` (23) + `test_event_gate_auto_hook.py` (6) |

Quick usage:

```bash
# 1. Audit feature redundancy (writes drop_list to summary.json)
python tools/audit_feature_redundancy.py \
    --features combined/day_trading_features.parquet \
    --output   audits/redundancy --threshold 0.9

# 2. Walk-forward fold with the auditor's drop list applied
python tools/run_walk_forward_fold_enhanced.py ... \
    --drop-features-from-audit audits/redundancy/redundancy_summary.json

# 3. Stress-test the trained strategy under degrading execution
python tools/diagnostics/stress_test_backtest.py \
    --features combined/day_trading_features.parquet \
    --output   diagnostics/stress

# 4. Verify the edge is not regime-conditional
python tools/diagnostics/regime_parity_test.py \
    --trades-csv backtests/baseline_strict/trades.csv \
    --output     diagnostics/regime

# 5. Audit the upstream event gate (catches "80 % NEUTRAL" pathologies)
#    NOTE: prepare_day_trading.py runs this automatically as a post-write
#    step and prints the verdict inline. Run manually only when re-auditing
#    an already-built parquet.
python tools/diagnostics/audit_event_gate.py \
    --features combined/day_trading_features.parquet \
    --output   diagnostics/event_gate
```

Pipeline-issues coverage status (from `docs/PIPELINE_ISSUES_AUDIT.md`):

| Pipeline risk | Status | Addressed by |
|---|---|---|
| Normalization drift (live ↔ train) | ✅ Mitigated | `FrozenScaler` + `load_hybrid_for_inference` |
| Feature bloat / redundancy | ✅ Mitigated | `audit_feature_redundancy` + `--drop-features-from-audit` |
| Slippage realism | ✅ Diagnosed | Stress test (severe / extreme scenarios) |
| Latency degradation | ✅ Diagnosed | Stress test (`extra_latency_bars`) |
| Regime drift | ✅ Diagnosed | Regime parity test |
| Live state management / event buffer | ⏳ Deferred | No live layer exists yet — documented as the first thing to build |

### Execution realism — Replay Engine (separate stack)

The four-layer hardening above protects the *training* and *bar-level
backtest* paths. Bar-level backtest still assumes constant slippage
(`0.5 pip/side` in `self_supervised/backtest_day_trade.py`) and integer
"latency bars". For execution realism we ship a **stand-alone**
event-by-event simulator in `modules/replay_engine/`:

| Piece | What it models |
|---|---|
| `LOBSnapshot` + `LOBState` | Immutable 10-level book view + sequential MBP-10 ingestion |
| `fill_market_order` | Walks the book level-by-level — returns VWAP fill, residual, levels consumed |
| `AdaptiveSlippage` | Slippage = f(level-1 depth ratio); blows up when L1 is thin |
| `LatencyModel` | Lognormal-ish per-order latency with seedable RNG |
| `ReplayBacktest` | Orchestrator: signal → snapshot @ decision → +latency → snapshot @ arrival → fill → log |

Run it as a separate CLI — independent of the bar-level backtest:

```bash
python tools/run_replay_backtest.py \
    --signals  backtests/baseline/signals.csv \
    --mbp      raw/mbp_10.parquet \
    --output   replay_results/baseline \
    --tick-size 0.0001 \
    --latency-mean-us 5000 --latency-jitter-us 2000 \
    --slip-base-ticks 0.5 --slip-mid-ticks 1.5
```

Outputs `execution_log.jsonl` (one fill per line, snapshot included) and
`replay_summary.json` (aggregate slippage / latency / fill-quality stats).
The log is what you diff against the model's *assumed* slippage to
quantify backtest overfitting.

**Slippage calibration** — `tools/calibrate_slippage.py` closes the
model-vs-realised loop. It reads `execution_log.jsonl`, splits events
by region (sub-L1 / linear ramp / walking), refits each
`AdaptiveSlippage` constant via region-specific OLS-through-origin, and
applies safety clamps (non-negativity + base ≤ mid). On our mock data
it correctly refused to suggest a negative `mid_ticks` value when
region B was dominated by L0-fits, and surfaced a meaningful
`extra_per_level` drop (1.0 → 0.144) that closes the 3.23× over-shoot.
Verdicts: `WELL_CALIBRATED / OVER_CALIBRATED / UNDER_CALIBRATED / INSUFFICIENT_DATA`.

```bash
python tools/calibrate_slippage.py \
    --log    replay_results/baseline/execution_log.jsonl \
    --output replay_results/baseline/calibration \
    --tick-size 0.0001 \
    --current-base 0.5 --current-mid 1.5 --current-extra 1.0
```

### See also

- [`SUBSYSTEMS.md`](SUBSYSTEMS.md) — full architectural contract between
  day_trade, SSL, and trading_intel
- [`docs/RESEARCH_SYNTHESIS.md`](docs/RESEARCH_SYNTHESIS.md) — mapping of
  the seven failure mechanisms to fourteen proposed solutions across the
  Western / Chinese / Russian schools
- [`docs/PIPELINE_ISSUES_AUDIT.md`](docs/PIPELINE_ISSUES_AUDIT.md) — six
  pipeline failure modes (state, normalization, pandas-in-loop, lookahead,
  feature bloat, parity) and the code-level mitigation for each
- [`docs/PIPELINE_GUIDE.pdf`](docs/PIPELINE_GUIDE.pdf) — bilingual step-by-step
  operational guide (17 pages, EN + AR)

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
│   │   ├── anti_collapse/          ← Simplex ETF, Orthogonal Rep,
│   │   │                              DB-MTL balancer, Sharpe loss
│   │   ├── hybrid/                 ← Fusion model + policy + inference.py
│   │   │                              (FrozenScaler, load_hybrid_for_inference)
│   │   └── training/               ← train_hybrid.py + feature_selection.py
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
│   ├── extract_continuous_contract.py ← Smart contract rollover
│   ├── build_walk_forward_folds.py    ← Fold definition generator
│   ├── run_walk_forward_fold.py       ← Baseline fold runner
│   ├── run_walk_forward_fold_enhanced.py  ← Anti-collapse fold runner
│   │                                    (+--drop-features-from-audit)
│   ├── aggregate_walk_forward.py      ← Per-fold metrics aggregator
│   ├── audit_feature_redundancy.py    ← Correlation-cluster auditor
│   ├── dead_features_audit.py         ← Zero-variance / mostly-NaN cols
│   ├── backtest_smoke.py
│   ├── run_replay_backtest.py        ← Event-by-event LOB replay CLI
│   ├── calibrate_slippage.py         ← Refits AdaptiveSlippage from replay log
│   ├── generate_mock_data.py         ← Synthetic MBP-10 + signals fixtures
│   └── diagnostics/
│       ├── stress_test_backtest.py    ← Latency + slippage stress test
│       ├── regime_parity_test.py      ← Per-regime / per-vol-quartile parity
│       └── audit_event_gate.py        ← Upstream event-gate health audit
│
├── modules/replay_engine/             ← Event-by-event LOB simulator (separate stack)
│   ├── book.py                       ← LOBSnapshot + fill_market_order + AdaptiveSlippage
│   └── engine.py                     ← LatencyModel + ExecutionEvent + ReplayBacktest
│
├── modules/deep_lob/masked_modeling/  ← Masked-reconstruction auxiliary task (Path B)
│   ├── config.py                     ← MaskedModelingConfig dataclass
│   ├── masking.py                    ← generate_mask + apply_mask (3 strategies)
│   └── reconstruction.py             ← ReconstructionHead + compute_recon_loss
│
├── tests/                           ← 581 tests (2 skipped)
│   ├── test_lob_features_v2.py     ← LOB layer (23 tests)
│   ├── test_short_term_ssl.py      ← Short-term heads (19 tests)
│   ├── test_hybrid_model.py        ← Hybrid fusion (16 tests)
│   ├── test_adaptive_targets.py    ← Adaptive heads + policy (26 tests)
│   ├── test_subsystem_boundaries.py ← Boundary lint (4 tests)
│   └── ... (other existing tests)
│
└── _archive/                        ← 126 archived files (legacy V19,
                                       diagnostics, dead quantum_* modules,
                                       broken feature_enrichment, broken
                                       tests). Nothing deleted — restore
                                       with `git mv`. See PROJECT_TREE.md.
```

---

## 🧪 Tests

### Run the full active test suite
```bash
python -m pytest tests/ -q
# Expect: 457 passed, 2 skipped, 0 failed
```

### Run just the SSL + hybrid + diagnostics sweep
```bash
python -m pytest tests/test_lob_features_v2.py tests/test_short_term_ssl.py \
                 tests/test_hybrid_model.py tests/test_adaptive_targets.py \
                 tests/test_subsystem_boundaries.py \
                 tests/test_anti_collapse.py tests/test_dbmtl_and_sharpe.py \
                 tests/test_enhanced_fold_integration.py \
                 tests/test_inference_helpers.py \
                 tests/test_feature_redundancy_audit.py \
                 tests/test_feature_selection.py \
                 tests/test_stress_test.py tests/test_regime_parity.py \
                 tests/test_event_gate_audit.py -v
```

| Suite | Tests | Covers |
|---|---|---|
| `test_lob_features_v2.py` | 23 | 13-channel LOB tensor + HumanLOBCNN |
| `test_short_term_ssl.py` | 19 | 5 short-term SSL heads + causal label builders |
| `test_hybrid_model.py` | 16 | HybridModel fusion + per-head losses |
| `test_adaptive_targets.py` | 26 | Adaptive heads + decision policy + regression fixes |
| `test_subsystem_boundaries.py` | 4 | Architecture lint (positive + negative tests) |
| `test_anti_collapse.py` | * | Simplex ETF + Orthogonal Representation guarantees |
| `test_dbmtl_and_sharpe.py` | * | DB-MTL balancer + differentiable Sharpe loss |
| `test_enhanced_fold_integration.py` | 8 | Walk-forward fold runner with all anti-collapse flags |
| `test_inference_helpers.py` | 10 | `FrozenScaler` + live-safe hybrid loader |
| `test_feature_redundancy_audit.py` | 14 | Correlation-cluster auditor |
| `test_feature_selection.py` | 14 | `--drop-features-from-audit` integration |
| `test_stress_test.py` | 13 | Latency + slippage degradation scenarios |
| `test_regime_parity.py` | 19 | Per-regime / per-vol-quartile parity diagnostic |
| `test_event_gate_audit.py` | 23 | Upstream event-gate health + ghost-hunt diagnostics |
| `test_event_gate_auto_hook.py` | 6 | Pipeline-integrated post-write audit hook |
| `test_replay_engine.py` | 25 | LOB snapshot + market-order fill + adaptive slippage + latency + replay backtest |
| `test_calibrate_slippage.py` | 38 | Region-by-region OLS refit + safety clamps + verdict cascade |
| `test_masked_modeling.py` | 32 | Masked-reconstruction config + 3 mask strategies + recon head + loss |
| `test_walk_forward.py` | 11 | Fold generation + aggregator + end-to-end smoke |

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
| [`docs/RESEARCH_SYNTHESIS.md`](docs/RESEARCH_SYNTHESIS.md) | Seven SSL failure mechanisms → fourteen proposed solutions (Western / Chinese / Russian schools) |
| [`docs/PIPELINE_ISSUES_AUDIT.md`](docs/PIPELINE_ISSUES_AUDIT.md) | Six pipeline failure modes + the code-level mitigation for each |
| [`docs/PIPELINE_GUIDE.pdf`](docs/PIPELINE_GUIDE.pdf) | Bilingual operational guide (17 pages, EN + AR) |
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
