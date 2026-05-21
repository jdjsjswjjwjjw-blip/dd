# QuantSystem — Quantum-Inspired Trading System

نظام تداول كمي مدفوع بـ **الإحصاء + التعلم العميق + الفكر الكمي**.

> **Status (May 2026)**: 13/13 phases من تقرير الدمج
> [`QuantSystem_Integration_Report_v2_Quantum.pdf`](docs/QuantSystem_Integration_Report_v2_Quantum.pdf)
> منفّذة في **10 PRs** على branch [`claude/task-d-RcDhu`](https://github.com/jdjsjswjjwjjw-blip/dd/tree/claude/task-d-RcDhu).

---

## 📋 جدول التقدم (التقرير ⇄ الـ Sprints)

| Phase التقرير | الوصف | الحالة | Sprint |
|---|---|---|---|
| **Phase 1** | Import V19.2 (15 ملف) | ✅ | initial |
| **Phase 2** | Label Fix (Triple Barrier + MFE/MAE) | ✅ | initial |
| **Phase 3** | Enriched Features (87 → 138) | ✅ | initial + [#5](https://github.com/jdjsjswjjwjjw-blip/dd/pull/5) |
| **Phase 4** | Statistical Validation (FDR + permutation + purge) | ✅ | initial |
| **Phase 5** | DeepLOB 7-channel tensors | ✅ | initial |
| **Phase 6** | Integration Bridge | ✅ | initial + [#6](https://github.com/jdjsjswjjwjjw-blip/dd/pull/6) |
| **Phase 7** | Cleanup + Tests | ✅ | initial + [#1](https://github.com/jdjsjswjjwjjw-blip/dd/pull/1) |
| **Phase A** كمي | Vectorization (33-182×) | ✅ | initial |
| **Phase B** كمي | Bell Pairs (9 entangled) | ✅ | initial |
| **Phase C** كمي | Grover-amplified discovery | ✅ | [#3](https://github.com/jdjsjswjjwjjw-blip/dd/pull/3) |
| **Phase D** كمي | Quantum DL (QLSTM, QCNN, VQE) | ✅ | [#4](https://github.com/jdjsjswjjwjjw-blip/dd/pull/4) |
| **التقرير 6.4** | quantum_core + quantum_features | ✅ | [#2](https://github.com/jdjsjswjjwjjw-blip/dd/pull/2) |
| **التقرير 3.2** | Hierarchy reorganization | ✅ | [#8](https://github.com/jdjsjswjjwjjw-blip/dd/pull/8) |
| **المشكلة #5** | train_v19 splitting (facade) | ✅ | [#7](https://github.com/jdjsjswjjwjjw-blip/dd/pull/7) |
| **المشكلة #7** | dead features audit | ✅ | [#9](https://github.com/jdjsjswjjwjjw-blip/dd/pull/9) |
| **Operational infra** | Backtest/Paper smoke harness | ✅ | [#9](https://github.com/jdjsjswjjwjjw-blip/dd/pull/9) |
| **PDF Reproducibility** | Report builder | ✅ | [#10](https://github.com/jdjsjswjjwjjw-blip/dd/pull/10) |

| البند | الحالة | السبب |
|---|---|---|
| Backtest 6 سنوات | ⏸️ operational | يحتاج بيانات MBO/MBP حقيقية + GPU |
| Paper trading شهر | ⏸️ operational | يحتاج live data feed |
| Live trading | ⏸️ operational | يحتاج broker + capital |

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
├── modules/                  Production ML core
│   ├── label_engine_v2.py            Phase 2 (Triple Barrier + MFE/MAE)
│   ├── feature_enrichment.py         Phase 3 (Combine V19.2 features)
│   ├── statistical_validation_layer.py Phase 4 (FDR + permutation)
│   ├── deeplob_v7ch.py               Phase 5 (7-channel CNN)
│   ├── integration_bridge.py         Phase 6 (alphas + DL + walls)
│   ├── quantum_core/                 ⚛ التقرير 6.4 - 5 ملفات
│   ├── quantum_features/             ⚛ التقرير 6.4 - 3 ملفات + Bell pairs
│   ├── quantum_discovery/            ⚛ Phase C - 4 ملفات
│   ├── quantum_dl/                   ⚛ Phase D - 4 ملفات
│   └── ... (V19 ML core: catboost_brain, deeplob_cnn, lstm_brain, ...)
│
├── v19_2/                    V19.2 Discovery Layer (15 ملف، self-contained)
│   ├── edge_scanner_v2.py    Phase A (33-182× speedup)
│   ├── quantum/bell_pairs.py Phase B (9 Bell pairs)
│   └── tests/                8/8 simulators + 22/22 bell_pairs + 5/5 edge_scanner_v2
│
├── prepare_day_trading.py    + V19.2 enrichment hook (--enrich-v19-2)
├── train_v19.py              3-stage pipeline (Stage 1/2/3)
├── backtest_v19.py / walkforward_v19.py / paper_v19.py / live_predictor.py
│   (paper + live: + Phase 6 bridge wiring)
├── predict_v19.py
│
├── tests/                    ~150 tests (root suite)
├── tools/
│   ├── diagnostics/          13 diagnose scripts (نُقلت من الجذر)
│   ├── dead_features_audit.py    Sprint 9 (المشكلة #7)
│   ├── backtest_smoke.py         Sprint 9 (Phase 5 infra)
│   ├── paper_dry_run.py          Sprint 9 (Phase 6 infra)
│   ├── demo_integration_bridge.py
│   └── docs/
│       └── build_integration_report.py  Sprint 10 (PDF generator)
│
└── docs/
    ├── OPERATIONAL_GUIDE.md                     end-to-end usage
    └── QuantSystem_Integration_Report_v2_Quantum.pdf  التقرير الكامل (31 ص)
```

---

## ⚛ Quantum Stack (التقرير الفصل 6)

### Core primitives ([`modules/quantum_core/`](modules/quantum_core/))

```python
from modules.quantum_core import (
    QuantumState, hadamard_transform,       # 6.1 Superposition
    bell_state, ghz_state, cnot_gate,       # 6.2 Entanglement
    grover_iteration, optimal_iterations,    # 6.3 Interference
    measure_greedy, fidelity,               # measurement
    DecoherenceHandler,                     # noise
)
```

### Entangled features ([`modules/quantum_features/`](modules/quantum_features/))

```python
from modules.quantum_features import (
    compute_ghz_features,            # 3-particle entanglement
    build_feature_tensor,             # MPS-like outer product
    compute_quantum_walk_features,    # CT quantum walk على kNN graph
)
```

### Discovery — Phase C ([`modules/quantum_discovery/`](modules/quantum_discovery/))

```python
from modules.quantum_discovery import (
    grover_alpha_search,         # O(√N) discovery (بدل sequential stages)
    qaoa_select_subset,           # subset optimization
    vqe_find_alpha_mode,          # eigenvalue-based mode detection
    cluster_alphas_by_returns,    # spectral clustering (بدل K-means)
)
```

### Quantum DL — Phase D ([`modules/quantum_dl/`](modules/quantum_dl/))

```python
from modules.quantum_dl import (
    QNNCircuit, QNNLayer,              # parametrized rotation circuits
    QLSTM,                              # quantum LSTM (QNN gates)
    vqe_portfolio_optimize,             # Markowitz بـ variational ansatz
    fidelity_loss, trace_distance,      # quantum-inspired losses
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

### 2. Build dataset (Phase 3 — مع V19.2 enrichment)

```bash
python prepare_day_trading.py \
    --mbo path/to/mbo.parquet \
    --mbp path/to/mbp.parquet \
    --output pipeline_day_trading/features \
    --enrich-v19-2     # +21 sim/bp/zone features (Sprint 5)
```

### 3. Audit الـ features

```bash
python tools/dead_features_audit.py \
    --input pipeline_day_trading/features/day_trading_features.parquet \
    --output artifacts/audit.json
```

### 4. Discovery (الـ classical + Grover)

```python
# Classical (vectorized — 33-182× أسرع من V19 الأصلي)
from v19_2.edge_scanner_v2 import scan_edges_v2
classical = scan_edges_v2(df, horizons=[3, 6, 12])

# Grover-amplified (Phase C — O(√N))
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

### 6. Paper trading مع Phase 6 Bridge

```bash
python paper_v19.py \
    --csv recent_features.parquet \
    --models outputs_v19 \
    --output outputs_paper \
    --use-bridge \
    --alpha-set artifacts/alphas.json
```

→ `outputs_paper/paper_bridge.jsonl` يحوي bridge decisions لكل bar.

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

# المجموعات
python3 -m unittest tests.test_label_engine_v2          # Phase 2
python3 -m unittest tests.test_feature_enrichment       # Phase 3
python3 -m unittest tests.test_statistical_validation_layer  # Phase 4
python3 -m unittest tests.test_deeplob_v7ch             # Phase 5
python3 -m unittest tests.test_integration_bridge       # Phase 6
python3 -m unittest tests.test_quantum_core             # التقرير 6.4
python3 -m unittest tests.test_quantum_features         # التقرير 6.4
python3 -m unittest tests.test_quantum_discovery        # Phase C
python3 -m unittest tests.test_quantum_dl               # Phase D

# V19.2 (في v19_2/tests/)
python3 v19_2/tests/test_simulators.py
python3 v19_2/tests/test_bell_pairs.py
python3 v19_2/tests/test_edge_scanner_v2.py
```

كل الـ tests passing عبر كل branches الـ sprints.

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
| `claude/sprint-8-hierarchy-reorg` | Hierarchy (3.2): core/, simulators/, ... |
| `claude/sprint-9-audit-ops-infra` | Dead features audit + ops harness |
| `claude/sprint-10-docs-pdf-builder` | PDF builder للتقرير |

كل sprint = PR منفصل قابل للـ review منفرداً.

---

## 📑 Legacy QuantSystem V19 (الـ trunk)

النظام الأصلي قبل الدمج (لازال شغّال — backward compat 100%):

### Pipeline التدريب (3 stages)

1. `prepare_training_data.py` / `prepare_day_trading.py`
   - يحوّل MBO/MBP خام إلى dataset
   - يبني features (87) + labels + LOB tensors
2. `train_v19.py` — pipeline ثلاثي:
   - **Stage 1**: CatBoost + Regime meta-features
   - **Stage 2**: OOF DeepLOB visual embeddings
   - **Stage 3**: MetaLearner LSTM
3. `backtest_v19.py` / `walkforward_v19.py` — التقييم
4. `paper_v19.py` / `shadow_v19.py` — التشغيل غير الحي
5. `live_predictor.py` — التداول الحي

### الـ Stage stubs (الموجودة في الجذر)
- `stage1_refinery.py` — sharded refinery (resumable)
- `stage2_catboost.py` — CatBoost + Regime
- `stage3_train.py` — MetaLearner LSTM

> الجوهر بقي في `train_v19.py` (4707 سطر). الـ Sprint 7 وفّر `training/` facade
> كـ public API نظيف بدون نقل خطر.

### Architecture الأصلية

**Raw Data** (MBO/MBP) → **Features** (microstructure + book + context + AE + LOB tensors)
→ **Models** (CatBoost + Regime + DeepLOB + LSTM) → **Eval** (causal backtest + walk-forward + drift)

### Recommended Environment
- Python 3.10 / 3.11 / 3.12
- `venv` كافية ومُفضّلة

### Base Dependencies
- `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `plotly`, `openpyxl`, `pyyaml`, `pyarrow`, `catboost`, `tensorflow`, `tqdm`

### Optional
- `jupyterlab`, `ipykernel`, `hmmlearn`

```bash
pip install -r requirements.txt
# optional:
pip install jupyterlab ipykernel hmmlearn
```

### Regime Defaults
- `stage1` يستخدم `regime_mode=rules` بدل Wasserstein افتراضياً
- `regime_stride=50` + `deterministic_stage1=true`
- لـ research mode: شغّل Wasserstein يدوياً

---

## 📜 التوثيق

| المصدر | المحتوى |
|---|---|
| [`docs/OPERATIONAL_GUIDE.md`](docs/OPERATIONAL_GUIDE.md) | الـ end-to-end usage (Sprint 9) |
| [`docs/QuantSystem_Integration_Report_v2_Quantum.pdf`](docs/QuantSystem_Integration_Report_v2_Quantum.pdf) | التقرير التشخيصي (31 ص) |
| [`CHANGELOG.md`](CHANGELOG.md) | تاريخ كل الـ commits + sprints |
| [`tools/docs/README.md`](tools/docs/README.md) | كيفية إعادة بناء التقرير |

---

## 🔬 الـ Architecture الكاملة (post-integration)

### Data flow

```
MBO/MBP الخام
    ↓
prepare_day_trading.py (--enrich-v19-2) ← V19.2 simulators + Bell pairs
    ↓
~115 feature parquet ← Phase 3
    ↓
┌────────────────────┐                ┌──────────────────────┐
│  Discovery Layer   │                │   DL Training        │
│  (V19.2 + Phase C) │                │   (3 stages)         │
│                    │                │                      │
│  edge_scanner_v2   │   Phase 4      │   Stage 1: CatBoost  │
│  + Grover (Phase C)│  ─────────→    │   Stage 2: DeepLOB   │
│                    │                │   Stage 3: LSTM      │
│  → AlphaSet        │                │                      │
└─────────┬──────────┘                └──────────┬───────────┘
          │                                      │
          │            Phase 6                   │
          │       ┌──────────────────┐           │
          └──────→│ IntegrationBridge│←──────────┘
                  │  (alphas + DL +  │
                  │   wall exits)    │
                  └─────────┬────────┘
                            ↓
                    ┌───────────────┐
                    │  paper_v19    │ ← Sprint 6 wiring
                    │  live_predictor│
                    └───────────────┘
```

### الـ Quantum extensions

- **Superposition** (6.1): `prepare_state_from_dataframe` على Alpha space
- **Entanglement** (6.2): Bell pairs بين features (9 pairs) + GHZ (3-particle)
- **Interference** (6.3): Grover amplification — O(√N) بدل O(N) في discovery
- **Quantum DL** (6.5): QNN + QLSTM + VQE portfolio + fidelity loss

---

## 📞 Help/Issues

- Issues على GitHub: [Issues tab](https://github.com/jdjsjswjjwjjw-blip/dd/issues)
- PRs المفتوحة: [Pull Requests](https://github.com/jdjsjswjjwjjw-blip/dd/pulls)
