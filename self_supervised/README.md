# SSL Training Infrastructure for QuantSystem V19

Complete Self-Supervised Learning pipeline that pretrains LOB Transformer (PR #17) and Price Cycle Model (PR #18) on raw data, then fine-tunes a small direction head on the few clean labels.

## Why SSL?

**Problem:** Traditional supervised approach produces only ~100-500 directional labels from a month of data (too few to train). Even with Sprint 19 fixes, labels remain sparse.

**Solution:** SSL pretrains the deep encoders on data-derived targets (no labels needed), then a small direction head learns from the available labels. Transfer learning makes 100-500 labels enough.

```
Traditional: 100 labels → train everything → overfit
SSL:         1,234 bars × 10 SSL tasks → pretrain encoders
             + 100 labels → train direction head only
             → 95% of model knowledge from unlabeled data
```

## Quick Start

```bash
# 1. Prerequisites: prepare_day_trading.py output
ls pipeline_clean/day_trading_features.parquet
ls pipeline_clean/lob_tensors.npy
ls mbo_data.parquet   # ← مهم لـ iceberg detection!

# 2. Run full SSL pipeline (~5-7 hours on GPU، with iceberg support)
chmod +x self_supervised/run_full_pipeline.sh
./self_supervised/run_full_pipeline.sh pipeline_clean mbo_data.parquet checkpoints/ssl_run 0.75

# 3. Check verdict
cat checkpoints/ssl_run/validation_report.json
```

## ⚠️ Important: Real Orders vs Pseudo-Orders

**Pass raw MBO** to enable iceberg detection. Without it:
- ✅ Pipeline still works
- ❌ LOB Transformer can't detect iceberg patterns
- ❌ Cancellation patterns lost
- ❌ Order lifecycle info gone

**With raw MBO:**
- ✅ Real orders preserved (order_id, action, full sequence)
- ✅ Iceberg detection via repeat_count + level_hits
- ✅ Cancellation patterns intact
- ✅ Latent event types (sweep/absorb/spoof/iceberg/wall_build/wall_break) discoverable

## Pipeline Phases

### Phase A0: Build OrderBatches من raw MBO (NEW — ~15-30 min CPU)
```bash
python self_supervised/build_order_batches.py \
    --mbo mbo_data.parquet \
    --features pipeline_clean/day_trading_features.parquet \
    --output checkpoints/ssl_run/order_batches \
    --lookback-bars 50 --n-orders 200
```

يبني `order_features.npy` و `order_masks.npy`:
- Shape: `(N_bars, T=50, N_orders=200, F=7)`
- 7 features per order: side, action_type, log_size, price_dist, time_offset, **log_repeat_count** (iceberg!), **log_level_hits** (wall!)
- لكل bar، آخر 200 order من الـ window (T=50 bars)

### Phase A: Pretrain LOB Transformer (~1-2 hours GPU)
```bash
python self_supervised/pretrain_lob.py \
    --features pipeline_clean/day_trading_features.parquet \
    --lob-tensors pipeline_clean/lob_tensors.npy \
    --output checkpoints/ssl_lob \
    --epochs 50 --batch-size 32
```

Trains 6 SSL heads (no direction labels):
- `next_price`: log return next bar (regression)
- `next_imbalance`: next bar OBI (regression)
- `next_volatility`: next bar ATR (regression)
- `next_regime`: next bar regime class (classification)
- `wall_persist`: bars until OBI flip (regression)
- `time_to_event`: bars until next event (regression)

### Phase B: Pretrain Price Cycle Model (~30-60 min GPU)
```bash
python self_supervised/pretrain_cycle.py \
    --features pipeline_clean/day_trading_features.parquet \
    --lob-tensors pipeline_clean/lob_tensors.npy \
    --output checkpoints/ssl_cycle \
    --epochs 40 --batch-size 32
```

Trains 4 SSL heads (Wyckoff weak labels from rules):
- `phase`: Wyckoff phase (Accumulation/Markup/Distribution/Markdown)
- `maturity`: trend maturity (young/mature/exhausted)
- `swing`: swing direction (up/down/neutral)
- `cycle_position`: position within cycle [0, 1]

### Phase C: Extract Embeddings (~5-10 min)
```bash
python self_supervised/extract_embeddings.py \
    --features pipeline_clean/day_trading_features.parquet \
    --lob-tensors pipeline_clean/lob_tensors.npy \
    --lob-checkpoint checkpoints/ssl_lob/best_ssl_lob.pt \
    --cycle-checkpoint checkpoints/ssl_cycle/best_ssl_cycle.pt \
    --output-dir checkpoints/embeddings
```

Produces `embeddings.npy` of shape `(N, micro_dim + macro_dim + seasonal + cycle_structural)` per bar.

### Phase D: Fine-Tune Direction Head (~5-15 min)
```bash
python self_supervised/fine_tune_direction.py \
    --features pipeline_clean/day_trading_features.parquet \
    --embeddings checkpoints/embeddings/embeddings.npy \
    --output checkpoints/direction_head \
    --epochs 100 --batch-size 32
```

Small MLP (3 layers, ~10K parameters) trained on directional labels only.

### Phase E: Validation
```bash
python self_supervised/validate_ssl.py \
    --features pipeline_clean/day_trading_features.parquet \
    --embeddings checkpoints/embeddings/embeddings.npy \
    --direction-head checkpoints/direction_head/best_direction_head.pt \
    --output checkpoints/validation_report.json
```

Produces verdict report with 6 acceptance criteria.

## Acceptance Criteria

| Metric | Threshold | Why |
|---|---|---|
| Holdout Accuracy | > 55% | Better than random (50%) |
| Sharpe (after costs) | > 0.5 | Profitable after spread+fees |
| Calibration AUC | > 0.60 | Confidence is meaningful |
| LONG/SHORT balance | 0.5-2.0 | Not catastrophically biased |
| Per-regime accuracy | trending+ranging > 50% | Works in main regimes |
| Max Drawdown | < 5% | No catastrophic losses |

### Verdict outcomes

- **✅ PASS** (5+/6 criteria): SSL works, expand to multi-timeframe
- **⚠️ MIXED** (3-4/6): Tune hyperparameters, more data
- **❌ FAIL** (<3/6): SSL not suitable for this data; reconsider approach

## File Structure

```
self_supervised/
├── README.md                  # this file
├── __init__.py
├── data_loader.py             # SSL Dataset + DataLoaders
├── pretrain_lob.py            # Phase A: LOB Transformer SSL
├── pretrain_cycle.py          # Phase B: Price Cycle SSL
├── extract_embeddings.py      # Phase C: embedding extraction
├── fine_tune_direction.py     # Phase D: direction head training
├── validate_ssl.py            # Phase E: validation report
└── run_full_pipeline.sh       # orchestration script
```

## Outputs

```
checkpoints/ssl_run/
├── lob/best_ssl_lob.pt           # pretrained LOB Transformer
├── cycle/best_ssl_cycle.pt       # pretrained Price Cycle
├── embeddings/embeddings.npy     # combined embeddings per bar
├── direction/best_direction_head.pt  # fine-tuned direction head
├── validation_report.json        # final verdict
└── phase_*.log                   # per-phase logs
```

## Architecture Overview

```
                    Raw Inputs
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   LOB Tensors     OHLCV Bars     ts_event
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐    ┌─────────┐    ┌────────┐
   │   LOB   │    │  Price  │    │Seasonal│
   │  Trans. │    │  Cycle  │    │  Map   │
   │  (SSL)  │    │  (SSL)  │    │ (rules)│
   └────┬────┘    └────┬────┘    └───┬────┘
        │              │             │
        ▼              ▼             ▼
   micro_emb       macro_emb     21 features
     (64)            (64)
        │              │             │
        └──────────────┼─────────────┘
                       │ + 12 cycle structural
                       ▼
              [Combined: ~161-dim]
                       │
                       ▼
             [Direction Head MLP]
                       │
                       ▼
                [LONG | SHORT]
```

## Dependencies

- PyTorch >= 2.0 (with CUDA for GPU training)
- NumPy, Pandas
- scikit-learn (for AUC calculation)
- PR #17 modules (`modules/deep_lob/`)
- PR #18 modules (`modules/price_cycle/`)

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
Install PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu124`

### Phase A loss not decreasing
- Check learning rate (`--lr 5e-5` instead of `1e-4`)
- Increase batch size if GPU memory allows
- Verify SSL targets aren't all zeros (rare data issue)

### Few directional labels (< 50)
- Re-run prepare_day_trading with `--ignore-event-direction-veto`
- Lower thresholds: `--timeout-mfe-min-move-atr 0.3 --timeout-mfe-mae-ratio 1.3`
- Switch to 15min freq instead of 1min/5min

### Verdict FAIL
- Try multi-scale (add 1H, 1D)
- Get more historical data (3+ months)
- Consider RL approach instead
