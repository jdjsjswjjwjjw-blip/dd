# Operational Guide — QuantSystem

دليل التشغيل بعد إكمال جميع الـ implementation work من التقرير
(`QuantSystem_Integration_Report_v2_Quantum.pdf`).

## المتطلبات

- Python 3.11+
- numpy, pandas (إلزامي)
- pyarrow (لـ parquet)
- اختياري للـ training: sklearn, catboost, tensorflow, torch
- بيانات MBO/MBP (Databento أو ما شابه)
- broker connection (للـ live trading)

---

## 1. Build dataset (Phase 3)

### بدون V19.2 features (legacy)

```bash
python prepare_day_trading.py \
    --mbo path/to/mbo.parquet \
    --mbp path/to/mbp.parquet \
    --output pipeline_day_trading/features
```

### مع V19.2 enrichment (موصى به — Sprint 5)

```bash
python prepare_day_trading.py \
    --mbo path/to/mbo.parquet \
    --mbp path/to/mbp.parquet \
    --output pipeline_day_trading/features \
    --enrich-v19-2
```

الـ output يحوي:
- 87 feature أصلية
- +18 sim_* (V19.2 simulators)
- +9 bp_* (Bell pairs)
- +zone_full, level_distance

= **~115 feature**.

---

## 2. Audit dead features (Sprint 9)

```bash
python tools/dead_features_audit.py \
    --input pipeline_day_trading/features/day_trading_features.parquet \
    --output artifacts/audit/dead_features.json
```

الـ output:
- dead (std=0)
- weak (std<1e-3)
- null-heavy (>50% null)
- low unique (<3 values)
- near-constant (>95% same value)

طبقاً للتقرير: نسبة ميتة > 10% = مشكلة في الـ pipeline.

---

## 3. Discovery (V19.2 + Phase C)

### Classical (sequential)

```bash
python v19_2/run_pipeline.py --features pipeline_day_trading/features
```

### Vectorized (33-182× أسرع — Sprint Phase A)

```python
from v19_2.edge_scanner_v2 import scan_edges_v2
result = scan_edges_v2(df, horizons=[3, 6, 12], run_permutation=True)
```

### Grover-amplified (Sprint 3 — Phase C)

```python
from modules.quantum_discovery import GroverDiscoveryConfig, grover_alpha_search

config = GroverDiscoveryConfig(
    perm_p_max=0.05, win_rate_min=0.55, sharpe_min=1.0, n_days_min=5, top_k=15,
)
result = grover_alpha_search(candidates_df, config)
# result.selected_alphas: top-K with grover_amplitude
```

---

## 4. Train (3 stages)

### Direct (legacy)

```bash
python train_v19.py --csv pipeline_day_trading/features/day_trading_features.parquet
```

### عبر training/ facade (Sprint 7)

```python
from training import run_training_pipeline

run_training_pipeline(
    csv_path='pipeline_day_trading/features/day_trading_features.parquet',
    output_dir='outputs_v19',
)
```

أو stage-by-stage:

```python
from training import run_stage1, run_stage2, run_stage3
artifacts = run_stage1(csv_path=...)
visual = run_stage2(csv_path=..., stage1_artifacts=artifacts)
final = run_stage3(csv_path=..., visual_embeddings=visual)
```

---

## 5. Backtest (Phase 5 من timeline)

### Smoke test (synthetic — Sprint 9)

```bash
python tools/backtest_smoke.py --n-rows 10000
```

### Walk-forward (real data)

```bash
python walkforward_v19.py \
    --csv pipeline_day_trading/features/day_trading_features.parquet \
    --models outputs_v19 \
    --output outputs_v19_walkforward
```

---

## 6. Paper trading (Phase 6)

### Dry run (synthetic — Sprint 9)

```bash
python tools/paper_dry_run.py --n-bars 500
```

### Real data + IntegrationBridge (Sprint 6)

```bash
python paper_v19.py \
    --csv path/to/recent_features.parquet \
    --models outputs_v19 \
    --output outputs_paper \
    --use-bridge \
    --alpha-set artifacts/alphas.json
```

Output: `outputs_paper/paper_bridge.jsonl` يحتوي bridge decisions لكل bar.

---

## 7. Live trading (Phase 7)

```python
from live_predictor import predict_live
from modules.integration_bridge import IntegrationBridge, BridgeConfig
from modules.statistical_validation_layer import AlphaSet

# Setup
alpha_set = AlphaSet(candidates=[...])  # من Phase C
bridge = IntegrationBridge(alpha_set, BridgeConfig())
models = {...}  # CatBoost regime models

# Per bar:
signal, conf, debug = predict_live(
    bar_features=bar,
    current_regime='trending',
    event_score=0.75,
    models=models,
    bridge=bridge,  # ← Phase 6 wiring
)

# debug['bridge_decision'] = {action, confidence, reason}
```

---

## 8. الهيكل المرتّب (Sprint 8 — التقرير 3.2)

```python
# Core infrastructure
from core import market_specs, statistics_module, fix_ohlc

# Microstructure simulators
from simulators import feature_simulators, wall_depth_simulator, iceberg_simulator

# Session + Liquidity context
from context import session_mapper, liquidity_topology_engine

# Statistical + Quantum discovery
from discovery import (
    edge_scanner_v2,        # classical, vectorized
    grover_alpha_search,    # quantum-amplified
)

# DL pipeline
from dl_pipeline import label_engine_v2, deeplob_v7ch, feature_enrichment

# Training (3 stages)
from training import run_stage1, run_stage2, run_stage3, run_training_pipeline

# Deployment (lazy loaders)
from deployment import (
    get_paper_v19, get_live_predictor, get_integration_bridge,
)
```

---

## 9. Quantum stack (التقرير الفصل 6)

```python
# Core primitives
from modules.quantum_core import (
    QuantumState, hadamard_transform,           # 6.1 Superposition
    bell_state, ghz_state, cnot_gate,           # 6.2 Entanglement
    grover_iteration, optimal_iterations,        # 6.3 Interference
    measure_greedy, fidelity,                   # measurement
    DecoherenceHandler,                         # noise
)

# Entangled features
from modules.quantum_features import (
    compute_ghz_features, build_feature_tensor,
    compute_quantum_walk_features,
)

# Discovery (Phase C)
from modules.quantum_discovery import (
    grover_alpha_search,
    qaoa_select_subset,
    vqe_find_alpha_mode,
    quantum_cluster,
)

# Quantum DL (Phase D)
from modules.quantum_dl import (
    QNNCircuit, QLSTM,
    vqe_portfolio_optimize,
    fidelity_loss, trace_distance,
)
```

---

## Map: التقرير ⇄ الـ Sprints

| التقرير | Sprint | PR |
|---|---|---|
| Phase 1: Import V19.2 | task-d-RcDhu | (initial) |
| Phase 2: Label Fix | task-d-RcDhu | (initial) |
| Phase 3: Enriched Features | task-d-RcDhu + Sprint 5 | PR #5 |
| Phase 4: Statistical Validation | task-d-RcDhu | (initial) |
| Phase 5: DeepLOB 7ch | task-d-RcDhu | (initial) |
| Phase 6: Integration Bridge | task-d-RcDhu + Sprint 6 | PR #6 |
| Phase 7: Cleanup | task-d-RcDhu + Sprint 1 | PR #1 |
| Phase A: Vectorization | task-d-RcDhu | (initial) |
| Phase B: Bell Pairs | task-d-RcDhu | (initial) |
| Phase C: Grover Discovery | Sprint 3 | PR #3 |
| Phase D: Quantum DL | Sprint 4 | PR #4 |
| التقرير 6.4: Quantum Core/Features | Sprint 2 | PR #2 |
| التقرير 3.2: Hierarchy | Sprint 8 | PR #8 |
| المشكلة #5: train_v19 splitting | Sprint 7 | PR #7 |
| المشكلة #7: dead features audit | Sprint 9 | (هذا) |
| Operational infrastructure | Sprint 9 | (هذا) |
