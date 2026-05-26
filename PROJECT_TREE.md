# 📂 Project Structure (post-cleanup)

> **Cleanup tag:** `v1-baseline-pre-cleanup` (commit `c7b88dd`)
> **Cleanup commit:** see git log for "phase 0: cleanup"

## 🎯 ACTIVE PIPELINE

Two root-level entry points + 4 active package directories.

### Root entry points
| File | Purpose |
|---|---|
| `prepare_day_trading.py` | Main pipeline — bars + features + labels from MBO/MBP. Produces `day_trading_features.parquet`, `lob_tensors.npy`, etc. Generates `event_flag`, `event_direction`, `path_outcome` labels. |
| `regime_config.py` | Hand-tuned constants: `REGIME_TP_SL`, `REGIME_MAX_BARS`, event thresholds. |

### Active packages
| Directory | Files | Purpose |
|---|---|---|
| `modules/` | 103 | Shared library: `deep_lob/` (HierarchicalLOBTransformer), `price_cycle/` (PriceCycleModel), `context_features`, `intrabar_mbp_microstructure`, `tick_intrabar_slices`, `manifest_v19`, regime classifier, etc. |
| `self_supervised/` | 22 | SSL training pipeline + backtest tools |
| `tests/` | 36 | Pytest suite (V19-only tests archived) |
| `tools/` | 20 | Utilities: `paper_dry_run.py`, `backtest_smoke.py`, day-trade diagnostics |
| `core/` | tiny | Shared core utilities |
| `context/` | — | Context features (referenced by modules) |
| `docs/` | — | Documentation |

### Key self_supervised/ scripts (post-cleanup workflow)
```
1. prepare_day_trading.py           → per-quarter bars + labels + LOB
2. self_supervised/build_order_batches.py → per-quarter order tensors
3. self_supervised/combine_quarter_artifacts.py → merge quarters with rolls
4. self_supervised/pretrain_lob.py + pretrain_cycle.py → SSL pretrain
5. self_supervised/extract_embeddings.py → 161-dim embeddings
6. self_supervised/fine_tune_regression.py → multi-horizon head
7. self_supervised/backtest_day_trade.py --strict → honest backtest
8. self_supervised/integrate_with_day_trade.py → SSL × day_trade merge
```

## 🗄️ ARCHIVED (`_archive/`)

Nothing is deleted. Everything can be restored with `git mv` if needed.

| Subdir | What's there | Why archived |
|---|---|---|
| `_archive/legacy_v19/` | 39 .py files — `backtest_v19`, `train_v19`, `predict_v19`, `walkforward_v19`, `paper_v19`, `shadow_v19`, `monitor_v19`, `patch_backtest_v19_daytrade`, `plot_v19_*`, `stage1_refinery`, `stage2_catboost`, `stage3_train`, `prepare_training_data`, `live_predictor`, `online_learning`, `alpha_validation`, `deep_validation`, `label_visualizer_5m`, + V19 tests + V19 tools + V19-only modules (`catboost_5m_report`, `raw_replay_v19`, `online_learning`, `live_predictor`) + `training/` dir | Old V19 pipeline — predecessor of day_trade. Not imported by active code. |
| `_archive/diagnostics/` | 21 .py files — one-off analysis scripts: `analyze_*`, `check_*`, `verify_*`, `validate_*`, `visualize_*`, `find_*`, `intraday_system_diagnostic`, etc. | Diagnostic scripts that ran once and produced reports. Not part of pipeline. |
| `_archive/unused_empty_dirs/` | `discovery/`, `dl_pipeline/`, `simulators/`, `deployment/` | Empty package directories (only had `__init__.py`). |
| `_archive/v19_2_dir/` | 20 .py files — older v19 fork | Historical version. |
| `_archive/v20_dir/` | 2 .py files — incomplete v20 prototype | Aborted. |
| `_archive/variants/` | `prepare_day_trading_enriched.py` | Experimental variant of main pipeline. |
| `_archive/old_artifacts/artifacts_v19/` | Old experiment outputs, logs, sample data | Generated artifacts. |
| `_archive/old_outputs/` | reserved for future moves | — |

## 🔁 Restoring an archived file
```bash
git mv _archive/legacy_v19/<file>.py ./<file>.py
git commit -m "restore: <file> for <reason>"
```
History is preserved.

## ✅ Verified after cleanup
- All `modules/*` imports work
- All `self_supervised/*` scripts pass syntax check (except pre-existing
  syntax error in `self_supervised/backtest_zone_strategies.py` which
  predates this cleanup — not used by active pipeline)
- No dangling references to archived files in active code
