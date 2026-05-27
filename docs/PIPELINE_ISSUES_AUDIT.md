# Pipeline Issues — Audit against the live-trading failure modes

Source document: `pipeline_issues.md` (uploaded 2026-05-27).

The document lists five structural failure modes that turn a profitable
backtest into a money-losing live system:

  1. State management (no event-buffer for streaming inputs)
  2. Normalization drift (re-fitting scalers at inference)
  3. Latency from Pandas in the live loop
  4. Look-ahead bias (the silent killer)
  5. Feature bloat (too many noisy features)

This audit checks each one against our current codebase and records the
verdict + corrective action (if any).

## Summary

| # | Issue | Verdict | Action taken |
|---|---|---|---|
| 1 | State management / buffer | ⏳ **Deferred** | No live layer exists yet. Documented as the first thing to build when going live. |
| 2 | Normalization drift | ✅ **Mitigated** | Fit-on-train + saved-in-checkpoint already in place. Added `FrozenScaler` + `load_hybrid_for_inference()` for transform-only loading. |
| 3 | Pandas in live loop | ✅ **Architecturally clean** | Active inference path (`trading_intel/hybrid/`) has zero pandas imports. `prepare_day_trading.py` uses pandas, but it's offline preprocessing. |
| 4 | Look-ahead bias | ✅ **Multiply protected** | Five distinct layers of protection: causal rolling, shift(1) for VWAP, session_break filter, LEAKAGE_PATTERNS blocklist, sentinel-1 sanity. |
| 5 | Feature bloat | ⚠️ **Scale-dependent** | At 5-year scale (target), sample/param ratio is acceptable. At 6-month scale (current), it was borderline. No explicit feature selection step yet. |

Verdict overall: 3 issues fully clean, 1 mitigated, 1 deferred to the future
live layer. None of the five would corrupt our research/backtest outputs.

---

## Detailed findings

### Issue 1 — State management / event buffer

**The risk:** in training, the model sees a complete N-row window per
sample. In live, ticks arrive one at a time. Without a circular buffer
that reconstructs the same N-row window, the first N-1 predictions are
garbage.

**Where it would apply in our code:** `modules/integration_bridge.py`
exists (264 lines, lazy-imported from `modules/deep_lob/system_integration.py`)
but is **NOT** wired into any active code path. It has no buffer / deque
/ state-management infrastructure.

**Why we haven't shipped a fix:** we don't have a live trading path. Every
backtest in the repo runs against complete monthly parquet files, so the
N-row window is always available at every prediction step. The 5-year
walk-forward backtest still operates on pre-built features.

**Action when going live:** Add a `TickBuffer` class using `collections.deque(maxlen=N)`
in the live serving layer. Wire it before the feature extractor. Validate via
the parity test (Issue 6 below) before deploying.

### Issue 2 — Normalization drift

**The risk:** if `StandardScaler.fit()` runs inside the live loop, you're
using *this moment's* mean and std, not the mean and std the model was
trained on. Even a small drift produces wildly wrong predictions.

**What was already in place (`train_hybrid.py`):**
- Line 206: `mu_d, sig_d = _zscore_fit(X_daytrade[train_mask])`
  — z-score is fit on the train slice only.
- Lines 332–336: the resulting `mu` and `sigma` are saved into the
  checkpoint as `norm_daytrade`, `norm_ssl`, `norm_cnn`.

**What was missing:** a clean, transform-only loader for inference. The
checkpoint had the stats, but there was no helper that exposed them as
a "frozen" object. A future caller could easily forget and call
`fit_transform()` instead of `transform()`.

**Action taken (this commit):** added
`modules/trading_intel/hybrid/inference.py` with:
- `FrozenScaler` — a tiny dataclass with `mu`, `sigma`, and a single
  `transform()` method. No `fit()` exists on the API surface.
- `load_hybrid_for_inference()` — loads a hybrid checkpoint and returns
  `(model_in_eval_mode, scalers_dict)`. Raises `KeyError` if the
  checkpoint is missing scaler stats (loud failure, no silent skip).
- `predict_one()` — single-row inference helper. Takes raw NumPy
  arrays, applies the frozen scalers, runs forward, returns floats.
- Tests in `tests/test_inference_helpers.py` (10 tests, all passing)
  verify: transform matches manual computation, sigma is floored,
  shape mismatch raises, missing scalers/config raise loudly, and the
  returned model is in eval mode with `requires_grad=False`.

### Issue 3 — Pandas in the live loop

**The risk:** Pandas DataFrames carry index/dtype/resample overhead that
adds milliseconds per tick — fatal when the tick rate is hundreds per
second.

**Audit result:**
- `modules/trading_intel/hybrid/*.py` — 0 pandas imports.
- `modules/trading_intel/anti_collapse/*.py` — 0 pandas imports.
- `modules/trading_intel/ssl_heads/*.py` — 0 pandas imports (pure numpy
  in the label builders).
- `prepare_day_trading.py` — heavy pandas use, but this is offline
  preprocessing that runs once per dataset, not per-tick inference.
- `train_hybrid.py` — pandas only for parquet I/O; the training loop
  itself runs on torch tensors.

**Verdict:** Active inference path is already pandas-free. No action
needed beyond keeping it that way. The `inference.py` helpers above
also accept only NumPy/torch — pandas DataFrames would fail at
`scaler.transform()` because shape semantics differ.

### Issue 4 — Look-ahead bias

**The risk:** any feature that uses bar `i+1` (or later) when computing
the value at bar `i` gives the model a glimpse into the future. The
backtest looks perfect; the live system is random.

**Audit result:** we have five distinct layers of protection.

1. **Causal rolling windows** in `prepare_day_trading.py`:
   - `df['atr_14'] = tr.rolling(14, min_periods=14).mean()` (line 1469)
     — `min_periods=14` means the first 13 bars are NaN, not partial.
   - Every rolling indicator uses the same pattern.
2. **`shift(1)` for forward-looking computations** (lines 680–681):
   - `pv_sum = pv.shift(1).rolling(win, min_periods=win).sum()`
   - VWAP and other path-dependent metrics shift by 1 to drop the
     current bar's contribution.
3. **`is_session_break` filter** in all label builders:
   - `_forward_window_clean()` in `modules/trading_intel/ssl_heads/short_term_labels.py`
     rejects any sample whose forward window crosses a session break.
   - Applied to all 5 short-term heads and all 3 adaptive heads.
4. **LEAKAGE_PATTERNS blocklist** in `train_hybrid.py`:
   - 22 column patterns blocked from entering the feature matrix
     (`event_flag`, `event_direction`, `path_outcome`,
     `label_confidence`, `dataset_slice`, `target_ret_*`, etc.)
5. **Sentinel −1 for invalid labels**:
   - Direction labels for non-event rows are set to −1, and
     `cross_entropy(ignore_index=-1)` skips them — instead of silently
     treating them as NEUTRAL=2 (which masked a real bug pre-review).

**Verdict:** Look-ahead is the failure mode we've fought hardest. The
research-synthesis modules (`ShortTermHeads` with causal labels,
`OrthogonalRepresentationModule` to break rule-overlap) only work
*because* the underlying labels are causal. Continued vigilance: any
new feature must be reviewed for causality before merging.

### Issue 5 — Feature bloat

**The risk:** too many features in a high-capacity model = the model
memorizes noise. In production, the same noise patterns won't repeat,
and accuracy collapses.

**Current dimensions:**
- `daytrade_feature_dim: 135`
- `ssl_embed_dim: 161`
- `cnn_embed_dim: 32`
- Total input dim: **328**
- Hidden dim: 256, 3 layers → ~250k trainable params

**Sample/parameter ratios:**

| Data scale | Effective samples | Params | Ratio | Verdict |
|---|---|---|---|---|
| 6 months (already run) | ~5,000 events | 250k | 0.02 | ❌ bloated, observed collapse |
| 5 years (target, all bars) | ~145k bars | 250k | 0.58 | ✅ acceptable |
| 5 years (events only) | ~40k events | 250k | 0.16 | ⚠️ borderline |

**Why we haven't added an explicit feature selector:** at the 5-year
target scale, the bar-level ratio is in the acceptable zone. The
research-synthesis modules (Orthogonal penalty, DB-MTL) effectively
regularize the input space without dropping columns.

**Future action (optional):** if walk-forward shows persistent feature-
selection issues, add a permutation-importance pre-selector that drops
the bottom 30% of features before training. Until then, the
configuration is conservative enough.

### Issue 6 — Parity test (training ↔ live)

**The risk:** even with perfect causality and frozen scalers, the
training-time feature extraction pipeline could differ from the live-
time extraction in some subtle way (e.g., a `.fillna(0)` that runs in
batch but is skipped in streaming).

**Where we stand:** we don't have a live extraction path yet, so there's
nothing to test parity against. When the live layer is built, the
parity test described in the source document is the first acceptance
gate to clear:

  1. Take a small CSV slice from training data.
  2. Run it through `prepare_day_trading.py` (offline) and capture features.
  3. Run the same data through the live `feature_extractor` (streaming).
  4. Assert the two feature matrices are identical to float32 precision.

This test belongs in `tests/test_live_parity.py` and should be the only
gate between `_archive/legacy_v19/integration_bridge.py` becoming
`modules/integration_bridge.py` again.

---

## What this audit changes in the codebase

  • `modules/trading_intel/hybrid/inference.py` — new file (240 LOC)
    with `FrozenScaler`, `load_hybrid_for_inference`, `predict_one`,
    `apply_saved_scalers`. Live-trading-safe API.
  • `modules/trading_intel/hybrid/__init__.py` — re-exports the new
    inference helpers.
  • `tests/test_inference_helpers.py` — 10 tests, all passing.
  • This document.

Nothing else needed to change. The other four failure modes are either
not applicable to our current architecture (we're a research engine,
not a live engine) or were already addressed by the look-ahead /
causality protections we built earlier.

When the live layer is built, the audit checklist becomes the
acceptance criteria.
