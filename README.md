# QuantSystem — Quantum-Inspired Regime-Aware Trading System

نظام تداول كمي مدفوع بـ **الإحصاء + التعلم العميق + الفكر الكمي + Regime-Aware Discovery**.

> **Status (May 2026)**: 13/13 phases من تقرير الدمج الأول +
> 3/4 تعديلات من تقرير Regime Analysis = **17 PR** على branch
> [`claude/task-d-RcDhu`](https://github.com/jdjsjswjjwjjw-blip/dd/tree/claude/task-d-RcDhu).

---

## 📋 جدول التقدم

### تقرير الدمج (Integration Report v2 Quantum)

| Phase | الوصف | Sprint | PR |
|---|---|---|---|
| **Phase 1** | Import V19.2 (15 ملف) | initial | — |
| **Phase 2** | Label Fix (Triple Barrier + MFE/MAE) | initial | — |
| **Phase 3** | Enriched Features (87 → 138) | initial + Sprint 5 | [#5](https://github.com/jdjsjswjjwjjw-blip/dd/pull/5) |
| **Phase 4** | Statistical Validation (FDR + permutation) | initial | — |
| **Phase 5** | DeepLOB 7-channel tensors | initial | — |
| **Phase 6** | Integration Bridge | initial + Sprint 6 | [#6](https://github.com/jdjsjswjjwjjw-blip/dd/pull/6) |
| **Phase 7** | Cleanup + Tests | initial + Sprint 1 | [#1](https://github.com/jdjsjswjjwjjw-blip/dd/pull/1) |
| **Phase A** كمي | Vectorization (33-182×) | initial | — |
| **Phase B** كمي | Bell Pairs (9 entangled) | initial | — |
| **Phase C** كمي | Grover-amplified discovery | Sprint 3 | [#3](https://github.com/jdjsjswjjwjjw-blip/dd/pull/3) |
| **Phase D** كمي | Quantum DL (QLSTM, QCNN, VQE) | Sprint 4 | [#4](https://github.com/jdjsjswjjwjjw-blip/dd/pull/4) |
| **التقرير 6.4** | quantum_core + quantum_features | Sprint 2 | [#2](https://github.com/jdjsjswjjwjjw-blip/dd/pull/2) |
| **التقرير 3.2** | Hierarchy reorganization | Sprint 8 | [#8](https://github.com/jdjsjswjjwjjw-blip/dd/pull/8) |
| **المشكلة #5** | train_v19 splitting (facade) | Sprint 7 | [#7](https://github.com/jdjsjswjjwjjw-blip/dd/pull/7) |
| **المشكلة #7** | dead features audit | Sprint 9 | [#9](https://github.com/jdjsjswjjwjjw-blip/dd/pull/9) |
| **Operational infra** | Backtest/Paper smoke harness | Sprint 9 | [#9](https://github.com/jdjsjswjjwjjw-blip/dd/pull/9) |
| **PDF reproducibility** | Report builder | Sprint 10 | [#10](https://github.com/jdjsjswjjwjjw-blip/dd/pull/10) |
| **README** | Comprehensive rewrite | Sprint 11 | [#11](https://github.com/jdjsjswjjwjjw-blip/dd/pull/11) |

### تقرير Regime Analysis (الـ regime-aware enhancements)

| # | الوصف | Sprint | PR |
|---|---|---|---|
| **②** | Regime × Session interaction (chi-square + Cramér's V) | Sprint 12 | [#12](https://github.com/jdjsjswjjwjjw-blip/dd/pull/12) |
| **③** | Per-Regime Alpha Discovery + Library + Bridge | Sprint 13 | [#13](https://github.com/jdjsjswjjwjjw-blip/dd/pull/13) |
| **④** | Soft Regime Probabilities + weighted TP/SL | Sprint 14 | [#14](https://github.com/jdjsjswjjwjjw-blip/dd/pull/14) |
| ① | `stride: 50 → 12` (config-only) | — | — |

### Operational items (خارج scope الكود)

| البند | يحتاج |
|---|---|
| Backtest 6 سنوات | بيانات MBO/MBP حقيقية + GPU |
| Paper trading شهر | live data feed |
| Live trading | broker + capital |

---

## 🏗 الهيكل الموحد (التقرير 3.2)

```
QuantSystem/
├── core/                     ⚛ V19.2 core (market_specs, statistics_module, fix_ohlc)
├── simulators/               ⚛ V19.2 microstructure (feature_simulators, wall_depth, iceberg)
├── context/                  ⚛ V19.2 context (session_mapper, liquidity_topology)
├── discovery/                ⚛ V19.2 statistical + Phase C quantum (Grover, QAOA, VQE)
├── dl_pipeline/              ⚛ tensors + labels + enrichment + deeplob_v7ch
├── training/                 ⚛ Sprint 7 facade (data + 3 stages + orchestrator)
├── deployment/               ⚛ lazy loaders (backtest, paper, live, bridge)
│
├── modules/                  Production ML core + Quantum + Regime-aware
│   ├── label_engine_v2.py            Phase 2 (Triple Barrier + MFE/MAE)
│   ├── feature_enrichment.py         Phase 3 + Sprint 12 regime_session integration
│   ├── statistical_validation_layer.py Phase 4 + Sprint 13 per-regime discovery
│   ├── deeplob_v7ch.py               Phase 5 (7-channel CNN)
│   ├── integration_bridge.py         Phase 6 + Sprint 13 regime-aware filtering
│   ├── regime_session_interaction.py Sprint 12 (chi-square + Cramér's V)
│   ├── regime_alpha_library.py       Sprint 13 (Per-regime alpha JSON library)
│   ├── regime_probabilities.py       Sprint 14 (Soft P(regime) + weighted TP/SL)
│   ├── quantum_core/                 ⚛ التقرير 6.4 - 5 ملفات
│   ├── quantum_features/             ⚛ التقرير 6.4 - 3 ملفات + Bell pairs
│   ├── quantum_discovery/            ⚛ Phase C - 4 ملفات
│   ├── quantum_dl/                   ⚛ Phase D - 4 ملفات
│   └── ... (V19 ML core)
│
├── v19_2/                    V19.2 Discovery Layer (15 ملف، self-contained)
│   ├── edge_scanner_v2.py    Phase A (33-182× speedup)
│   ├── quantum/bell_pairs.py Phase B (9 Bell pairs)
│   └── tests/                8/8 simulators + 22/22 bell_pairs + 5/5 edge_scanner_v2
│
├── prepare_day_trading.py    + V19.2 enrichment hook (--enrich-v19-2)
├── train_v19.py              3-stage pipeline
├── backtest_v19.py / walkforward_v19.py / paper_v19.py / live_predictor.py
│   (paper + live: + Phase 6 bridge wiring)
├── predict_v19.py
│
├── tests/                    300+ tests (~270 + 22 V19.2 + 5 edge_scanner_v2)
├── tools/
│   ├── diagnostics/          13 diagnose scripts
│   ├── dead_features_audit.py    Sprint 9
│   ├── backtest_smoke.py         Sprint 9
│   ├── paper_dry_run.py          Sprint 9
│   ├── demo_integration_bridge.py
│   └── docs/
│       └── build_integration_report.py  Sprint 10 (PDF generator)
│
└── docs/
    ├── OPERATIONAL_GUIDE.md                     end-to-end usage
    └── QuantSystem_Integration_Report_v2_Quantum.pdf  التقرير الأول (31 ص)
```

---

## 🧠 Regime-Aware Stack (Sprints 12, 13, 14)

تنفيذ تقرير `Regime_Analysis_Report_v2.pdf` بشكل علمي وكمي:

### ② Regime × Session Interaction — Sprint 12

يميّز "trending في London" عن "trending في Asia" — chi-square + Cramér's V.

```python
from modules.regime_session_interaction import enrich_with_regime_session

df_enriched, diag = enrich_with_regime_session(df)
# diag["statistics"]["cramers_v"] = 0.234 ("moderate")
# diag["statistics"]["chi_square"] = 187.3, p_value < 0.001
```

**Math**:
- H₀: `P(target|regime) = P(target|regime, session)`
- H₁: session adds info beyond regime alone
- Test: Chi-square independence (Wilson-Hilferty p-value)
- Effect size: Cramér's V بـ Cohen 1988 benchmarks
- Cardinality: 3 × 16 = 48 combinations مع rare bucketing

### ③ Per-Regime Alpha Discovery — Sprint 13 (الفكرة الذهبية)

alpha في trending ≠ alpha في ranging. يحلّ مشكلة "alphas ذهبية تختبئ تحت المتوسط".

```python
from modules.statistical_validation_layer import discover_alphas_by_regime
from modules.regime_alpha_library import RegimeAlphaLibrary
from modules.integration_bridge import IntegrationBridge, BridgeConfig

# Discovery منفصل لكل regime
library = discover_alphas_by_regime(df, verbose=True)
# library.summary() → {'trending': 8, 'ranging': 4, 'volatile': 0}

# Save / Load
library.save('alphas_library.json')
library = RegimeAlphaLibrary.load('alphas_library.json')

# Regime-aware filtering في الـ Bridge
bridge = IntegrationBridge(
    alpha_set=AlphaSet(),       # legacy fallback
    config=BridgeConfig(),
    regime_library=library,      # ← per-regime filtering
)
decision = bridge.evaluate_row(row, dl_proba)
# يفلتر alphas حسب row['regime_label'] تلقائياً
```

**Math** (التقرير ص 12):
```
WR_combined = (10K_trending × 70%) + (2K_ranging × 30%) / 12K = 63%
              → alpha تُرفض إحصائياً
بينما WR_trending_only = 70% (لو فلترنا)
```

### ④ Soft Regime Probabilities — Sprint 14

استبدال hard labels بـ `P(regime|features)` → weighted TP/SL، يحل قفزات TP الحادة في live.

```python
from modules.regime_probabilities import (
    compute_soft_regime_pipeline,
    weighted_tp_sl_batch,
)

out = compute_soft_regime_pipeline(df, method='from_labels', smoothing_alpha=0.3)
tp_w = out.tp_weighted     # smooth TP per bar
sl_w = out.sl_weighted     # smooth SL per bar
```

**Math**:
1. Softmax: `P_i = exp(z_i/T) / Σ exp(z_j/T)`
2. Membership: `trending = σ((ADX-25)/10)`
3. EMA smoothing: `P_smooth_t = α·P_raw_t + (1-α)·P_smooth_{t-1}`
4. Weighted TP: `TP_w = Σ_r P(r) × TP_r`

**Properties verified (P1-P6)**:
- P1: Normalization (Σ P = 1)
- P2: Non-negativity (P ≥ 0)
- P3: Convergence to hard label
- P4: Convergence to uniform
- P5: Smoothness bound (|P_t - P_{t-1}|_∞ ≤ 2α)
- P6: TP monotonicity in P(trending)

---

## ⚛ Quantum Stack (التقرير الفصل 6)

```python
# Core primitives (Sprint 2)
from modules.quantum_core import (
    QuantumState, hadamard_transform,       # 6.1 Superposition
    bell_state, ghz_state, cnot_gate,       # 6.2 Entanglement
    grover_iteration, optimal_iterations,    # 6.3 Interference
    measure_greedy, fidelity,               # measurement
    DecoherenceHandler,                     # noise
)

# Entangled features (Sprint 2)
from modules.quantum_features import (
    compute_ghz_features, build_feature_tensor,
    compute_quantum_walk_features,
)

# Discovery (Sprint 3 — Phase C)
from modules.quantum_discovery import (
    grover_alpha_search,         # O(√N) discovery
    qaoa_select_subset,
    vqe_find_alpha_mode,
    cluster_alphas_by_returns,
)

# Quantum DL (Sprint 4 — Phase D)
from modules.quantum_dl import (
    QNNCircuit, QLSTM,
    vqe_portfolio_optimize,
    fidelity_loss, trace_distance,
)
```

---

## 🚀 Quick Start

### 1. التثبيت

```bash
# الـ base dependencies
pip install -r requirements.txt

# للـ docs/PDF generation (اختياري)
pip install -r requirements-docs.txt
# Linux: apt-get install fonts-dejavu
# macOS: brew install --cask font-dejavu
```

### 2. Build dataset (مع V19.2 enrichment)

```bash
python prepare_day_trading.py \
    --mbo path/to/mbo.parquet \
    --mbp path/to/mbp.parquet \
    --output pipeline_day_trading/features \
    --enrich-v19-2     # +21 sim/bp/zone features (Sprint 5)
                       # + regime_session interaction (Sprint 12)
```

### 3. Audit الـ features

```bash
python tools/dead_features_audit.py \
    --input pipeline_day_trading/features/day_trading_features.parquet \
    --output artifacts/audit.json
```

### 4. Discovery الكامل (3 طبقات)

```python
# الطبقة 1: Classical vectorized (33-182× أسرع)
from v19_2.edge_scanner_v2 import scan_edges_v2
classical = scan_edges_v2(df, horizons=[3, 6, 12])

# الطبقة 2: Per-Regime (الفكرة الذهبية)
from modules.statistical_validation_layer import discover_alphas_by_regime
library = discover_alphas_by_regime(df, verbose=True)

# الطبقة 3: Grover-amplified
from modules.quantum_discovery import grover_alpha_search, GroverDiscoveryConfig
quantum = grover_alpha_search(classical_candidates, GroverDiscoveryConfig(top_k=15))
```

### 5. Training (3 stages عبر facade)

```python
from training import run_training_pipeline
run_training_pipeline(
    csv_path='pipeline_day_trading/features/day_trading_features.parquet',
    output_dir='outputs_v19',
)
```

### 6. Paper trading مع Phase 6 Bridge + Regime-Aware

```bash
python paper_v19.py \
    --csv recent_features.parquet \
    --models outputs_v19 \
    --output outputs_paper \
    --use-bridge \
    --alpha-set artifacts/alphas_library.json
```

→ `outputs_paper/paper_bridge.jsonl` يحوي bridge decisions لكل bar
(مفلترة حسب regime لو `alpha_set` مبني بـ `discover_alphas_by_regime`).

### 7. Smoke harness (synthetic — للتحقق بدون بيانات)

```bash
python tools/backtest_smoke.py --n-rows 10000
python tools/paper_dry_run.py --n-bars 500
```

التشغيل الكامل end-to-end: [`docs/OPERATIONAL_GUIDE.md`](docs/OPERATIONAL_GUIDE.md).

---

## 🧪 Tests

```bash
# كل الـ tests
python3 -m unittest discover tests

# المجموعات الأساسية
python3 -m unittest tests.test_label_engine_v2          # Phase 2
python3 -m unittest tests.test_feature_enrichment       # Phase 3
python3 -m unittest tests.test_statistical_validation_layer  # Phase 4
python3 -m unittest tests.test_deeplob_v7ch             # Phase 5
python3 -m unittest tests.test_integration_bridge       # Phase 6

# Quantum stack
python3 -m unittest tests.test_quantum_core             # التقرير 6.4
python3 -m unittest tests.test_quantum_features         # التقرير 6.4
python3 -m unittest tests.test_quantum_discovery        # Phase C
python3 -m unittest tests.test_quantum_dl               # Phase D

# Regime-aware (Sprints 12, 13, 14)
python3 -m unittest tests.test_regime_session_interaction  # ② chi-square + Cramér's V
python3 -m unittest tests.test_per_regime_discovery        # ③ Per-regime library + bridge
python3 -m unittest tests.test_regime_probabilities        # ④ Soft P + weighted TP/SL

# V19.2 (في v19_2/tests/)
python3 v19_2/tests/test_simulators.py        # 8/8
python3 v19_2/tests/test_bell_pairs.py        # 22/22
python3 v19_2/tests/test_edge_scanner_v2.py   # 5/5
```

**Total**: ~300 tests، كلها passing عبر الـ branches.

---

## 🌳 الـ Branches

| Branch | محتوى |
|---|---|
| `claude/task-d-RcDhu` | **الـ trunk** — Phases 1-7 + A + B (الـ baseline) |
| `claude/sprint-1-cleanup-wiring` | Cleanup + Production wiring tests |
| `claude/sprint-2-quantum-core` | quantum_core/ + quantum_features/ |
| `claude/sprint-3-grover-discovery` | Phase C — Grover Discovery |
| `claude/sprint-4-quantum-dl` | Phase D — Quantum DL |
| `claude/sprint-5-prepare-day-trading-merge` | prepare_day_trading enrichment hook |
| `claude/sprint-6-paper-live-bridge` | paper_v19 + live_predictor ⇄ Bridge |
| `claude/sprint-7-train-split` | training/ facade |
| `claude/sprint-8-hierarchy-reorg` | Hierarchy (3.2) |
| `claude/sprint-9-audit-ops-infra` | Dead features audit + ops harness |
| `claude/sprint-10-docs-pdf-builder` | PDF builder للتقرير |
| `claude/sprint-11-readme-update` | README rewrite (الـ integration الأول) |
| `claude/sprint-12-regime-session-interaction` | ② Regime × Session |
| `claude/sprint-13-per-regime-discovery` | ③ Per-Regime Discovery (الذهبية) |
| `claude/sprint-14-soft-regime-probabilities` | ④ Soft Probabilities |
| `claude/sprint-15-readme-final-update` | README النهائي (هذا) |

كل sprint = PR منفصل قابل للـ review منفرداً.

---

## 📊 المقاييس المتوقعة (من التقريرين)

### Integration Report (الأول)
| المقياس | قبل | بعد |
|---|---|---|
| Class Distribution | 98% NEUTRAL | 40/30/30 |
| Precision LONG | 5-15% | 55-65% |
| Sharpe (backtest) | -0.3 ~ +0.5 | 1.0-2.0 |
| Features الحية | 51 | 138 (2.67×) |
| CNN channels | 4 | 7 (1.75×) |

### Regime Analysis Report (الثاني)
| المقياس | بدون regime-aware | مع regime-aware |
|---|---|---|
| WR | 55% | 65% |
| **Sharpe** | **0.6** | **1.7** (+183%) |
| Profit Factor | 1.2 | 2.1 |
| Max Drawdown | 18% | 11% |
| False exits في transitions | 15% | 5% |

---

## 📜 التوثيق

| المصدر | المحتوى |
|---|---|
| [`docs/OPERATIONAL_GUIDE.md`](docs/OPERATIONAL_GUIDE.md) | الـ end-to-end usage |
| [`docs/QuantSystem_Integration_Report_v2_Quantum.pdf`](docs/QuantSystem_Integration_Report_v2_Quantum.pdf) | التقرير التشخيصي الأول (31 ص) |
| `Regime_Analysis_Report_v2.pdf` (uploaded) | تقرير تحليل الـ regime (20 ص) |
| [`CHANGELOG.md`](CHANGELOG.md) | تاريخ كل الـ commits + sprints |
| [`tools/docs/README.md`](tools/docs/README.md) | كيفية إعادة بناء التقرير |

---

## 🔬 الـ Architecture الكاملة (post-integration)

### Data flow

```
MBO/MBP الخام
    ↓
prepare_day_trading.py (--enrich-v19-2)
    ├─ V19.2 simulators + Bell pairs (Phase 3)
    └─ Regime × Session interaction (Sprint 12)
    ↓
~115 feature parquet + regime_session col
    ↓
┌────────────────────────────────┐                ┌──────────────────────┐
│  Discovery Layer (3 طبقات)     │                │   DL Training        │
│                                │                │   (3 stages)         │
│  1. Classical vectorized       │                │                      │
│     (edge_scanner_v2)          │                │   Stage 1: CatBoost  │
│  2. Per-Regime (Sprint 13)     │  Phase 4       │   Stage 2: DeepLOB   │
│     ✦ Golden insight ✦         │  ─────────→    │   Stage 3: LSTM      │
│  3. Grover-amplified (Phase C) │                │                      │
│                                │                │                      │
│  → RegimeAlphaLibrary          │                │                      │
└─────────┬──────────────────────┘                └──────────┬───────────┘
          │                                                  │
          │            Phase 6 + Sprint 13                   │
          │       ┌──────────────────────┐                   │
          └──────→│ IntegrationBridge    │←──────────────────┘
                  │  + regime_library    │
                  │  + per-regime alphas │
                  │  + Sprint 14 soft TP/SL
                  │                      │
                  └─────────┬────────────┘
                            ↓
                    ┌───────────────┐
                    │  paper_v19    │ ← Sprint 6 wiring
                    │  live_predictor│
                    └───────────────┘
```

### الـ Quantum extensions

- **Superposition** (6.1): `prepare_state_from_dataframe` على Alpha space
- **Entanglement** (6.2): Bell pairs (9) + GHZ (3-particle)
- **Interference** (6.3): Grover amplification — O(√N) في discovery
- **Quantum DL** (6.5): QNN + QLSTM + VQE portfolio + fidelity loss

### الـ Regime-Aware extensions (Sprints 12-14)

- **Interaction** (②): Chi-square test + Cramér's V للـ regime × session
- **Discovery** (③): per-regime alpha library (الفكرة الذهبية)
- **Probabilities** (④): Soft P(regime) + weighted TP/SL/horizon

---

## 📞 Help / Issues

- Issues: [GitHub Issues](https://github.com/jdjsjswjjwjjw-blip/dd/issues)
- PRs (17): [Pull Requests](https://github.com/jdjsjswjjwjjw-blip/dd/pulls)

---

## 📑 Legacy QuantSystem V19 (الـ baseline)

النظام الأصلي قبل الدمج (يبقى backward compat 100%):

### Pipeline التدريب (3 stages)
1. `prepare_training_data.py` / `prepare_day_trading.py` — MBO/MBP → dataset
2. `train_v19.py` — 3 stages: CatBoost → DeepLOB → MetaLearner LSTM
3. `backtest_v19.py` / `walkforward_v19.py` — التقييم
4. `paper_v19.py` / `shadow_v19.py` — التشغيل غير الحي
5. `live_predictor.py` — التداول الحي

### الـ Stage stubs
- `stage1_refinery.py` (الجوهر بقي في train_v19.py، Sprint 7 وفّر facade)
- `stage2_catboost.py`
- `stage3_train.py`

### Recommended Environment
- Python 3.10 / 3.11 / 3.12 — `venv` كافية ومُفضّلة

### Base Dependencies
- `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `plotly`, `openpyxl`, `pyyaml`, `pyarrow`, `catboost`, `tensorflow`, `tqdm`

### Optional
- `jupyterlab`, `ipykernel`, `hmmlearn`

```bash
pip install -r requirements.txt
# optional:
pip install jupyterlab ipykernel hmmlearn
```

### Regime Defaults (legacy)
- `stage1` يستخدم `regime_mode=rules` بدل Wasserstein افتراضياً
- `regime_stride=50` + `deterministic_stage1=true`
- لـ research mode: شغّل Wasserstein يدوياً
- (Sprint 14 يضيف soft probabilities كـ optional layer فوق هذا)
