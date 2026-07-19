# MBO Simulation Layer -> SSL Foundation Blueprint

This document translates the target architecture into an implementable, leakage-safe
contract for this repository.

It follows the project constraints:
- Features are computed from present/past only.
- Labels (or predictive targets) alone may use future path.
- Session boundaries are sourced from `modules/session_features.py`.
- MBP-10 remains the spatial truth for book-shape tensors; MBO adds order-level detail.

## 1) End-to-end pipeline (target)

```text
MBO Raw Data
  -> Order Book Reconstruction
  -> Simulation Layer
  -> Feature Store
  -> Self-Supervised Foundation Model
  -> Latent Market States
  -> Statistical Testing
  -> LLM Research Assistant
  -> Reports / Hypotheses / Structures / Signals
```

## 2) Simulation Layer contract

The Simulation Layer is a deterministic transformation from reconstructed order book
state into causal features and optional simulator-only diagnostics.

### Required invariants

1. Causality:
   - At event time `t`, simulator output uses only events `<= t`.
2. Timestamp integrity:
   - Strictly monotonic event index per instrument/session.
3. Cross-source consistency:
   - Reconstructed MBO book snapshots should be periodically checked against MBP-10.
4. Relative normalization:
   - "Strong / weak" is measured versus trailing historical baseline only.
5. No global-fit scaling:
   - No full-series mean/std/min/max inside simulator feature computation.

### Canonical simulator families

1. Footprint Simulator:
   - Bid/ask volume, delta, imbalance, absorption.
2. Volume Profile Simulator:
   - POC, VAH/VAL, HVN/LVN, value migration.
3. Order Flow Simulator:
   - Aggressive buying/selling, trade initiation, liquidity consumption.
4. Liquidity Simulator:
   - Resting liquidity, pulls/adds, iceberg behavior.
5. Auction Market Simulator:
   - Balance/imbalance, expansion, pullback defense.
6. Cross-Market Simulator:
   - NQ vs MNQ lead/lag, divergence, confirmation failure, trap patterns.

## 3) Mapping to current codebase

This repository already contains major building blocks:

- Absorption detector:
  - `modules/features_v2/absorption.py`
  - `modules/features_v2/absorption_simulator.py`
- Iceberg detector:
  - `modules/features_v2/iceberg.py`
  - `modules/features_v2/iceberg_simulator.py`
- LOB spatial tensor and microstructure:
  - `modules/trading_intel/lob/features.py`
  - `modules/intrabar_mbp_microstructure.py`
- Execution realism / liquidity consumption replay:
  - `modules/replay_engine/book.py`
  - `modules/replay_engine/engine.py`
- SSL pretraining and embedding extraction:
  - `self_supervised/pretrain_lob.py`
  - `self_supervised/pretrain_cycle.py`
  - `self_supervised/extract_embeddings.py`

### Coverage status (current -> target)

| Simulator family | Current status | Main files | Gap to close |
|---|---|---|---|
| Footprint | Partial | `features_v2/absorption.py`, microstructure modules | Consolidated footprint outputs under one stable schema |
| Volume Profile | Partial | day-trade feature builders | Explicit profile simulator API (POC/VAH/VAL/HVN/LVN migration) |
| Order Flow | Partial | microstructure + replay engine | Unified aggressive-flow metrics with per-horizon windows |
| Liquidity | Partial-strong | `features_v2/iceberg.py`, LOB tensor channels | Pull/add liquidity event attribution by level |
| Auction Market | Partial | session/structure features | Explicit auction-state simulator outputs |
| Cross-Market | Limited | no dedicated NQ<->MNQ simulator package yet | Add synchronized dual-instrument simulator and diagnostics |

## 4) Feature Store contract

The Feature Store receives simulator outputs and writes:

1. Causal simulator features (for day-trade or hybrid usage).
2. Raw/high-dimensional tensors (for SSL foundation pretraining).
3. Metadata sidecar with:
   - build timestamp and git hash,
   - schema version,
   - causality checks,
   - missing-feature report.

Minimum schema fields:
- `ts_event` (UTC),
- `instrument_id`,
- `session_id` (from `session_features.py`),
- `feature_version`,
- `is_warmup`,
- simulator feature columns,
- optional validity masks per feature family.

## 5) SSL Foundation Model blueprint

The SSL foundation consumes simulator-aware raw representations and learns robust
latent states with multi-objective pretraining.

### Core SSL objectives

1. Temporal representation learning:
   - Event sequence and queue evolution encoding.
2. Hierarchical learning:
   - Event -> level -> book -> auction -> session abstractions.
3. Multi-scale learning:
   - microseconds, milliseconds, seconds, minutes contexts.
4. Contrastive learning:
   - positive/negative sampling across local regime context.
5. Masked modeling:
   - masked event/order/queue reconstruction.
6. Predictive world modeling:
   - next-state and liquidity-evolution prediction.
7. Latent memory:
   - long context and session memory.
8. Cross-market representation:
   - NQ/MNQ aligned latent space.
9. Causal representation:
   - intervention-aware and structural dependence constraints.

### Practical implementation note

Do not train all objectives at equal weight from day one. Use phased objective
enablement with validation gates to prevent representation collapse.

## 6) Statistical testing gate (mandatory)

Every simulator family and every SSL representation release must pass:

1. Significance tests:
   - test whether signal differs from null under chronological splits.
2. Robustness tests:
   - sensitivity to spreads, slippage, latency, and sampling shifts.
3. Out-of-sample tests:
   - forward-chaining / walk-forward only.
4. Regime validation:
   - per-session and per-regime breakdown.
5. Hypothesis verification:
   - explicit pass/fail criteria per claimed pattern.

No module graduates to production pipeline wiring without this gate.

## 7) LLM Research Assistant role (strictly downstream)

The LLM layer must be consumer-only:
- It interprets latent states and diagnostics.
- It proposes hypotheses and experiment plans.
- It does not generate training labels directly.
- It cannot bypass statistical testing gates.

Output artifacts:
- research reports,
- candidate hypotheses,
- discovered structure notes,
- alpha ideas tagged as unverified until tested.

## 8) Suggested implementation phases

Phase A: Simulator unification
- Add stable schemas for all six simulator families.
- Keep outputs causal and versioned.

Phase B: Feature Store hardening
- Add simulator manifests and validation masks.
- Add MBP-vs-MBO reconstruction parity checks.

Phase C: SSL objective registry
- Register objectives with per-objective metrics and fail-fast checks.

Phase D: Statistical gate automation
- Add reusable validation CLI for significance/robustness/OOS/regime tests.

Phase E: LLM research interface
- Generate structured report templates fed by tested artifacts only.

## 9) Verification checklist

Before any experiment run:

1. Causality checks pass for all simulator features.
2. Session definitions match `session_features.py`.
3. MBO reconstruction parity against MBP-10 is within tolerance.
4. SSL objective metrics are stable (no collapse warnings).
5. Out-of-sample statistical gates pass on latest split.

If any check fails, block promotion and log failure reason.
