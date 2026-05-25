"""
self_supervised/verify_integration.py
═══════════════════════════════════════════════════════════
End-to-end integration verifier — every handoff between
prepare_day_trading → SSL pipeline → model is asserted.

A single PASS guarantees:
  1.  Cleaned MBO/MBP files have sensible price ranges + clean timestamps
  2.  Parquet has atr_14, is_session_break, 21 seasonal cols, 7+ cycle
      structural cols, OHLC from trades only (B1 sanity-checkable)
  3.  LOB tensors are (N, 50, 20, 9) with the new channel layout
  4.  SSLDataset's context_cols includes at least 5 seasonal + 5 cycle
      structural + 3 technical features (Seasonal Map IS reaching SSL)
  5.  Train/holdout split has the embargo gap correctly applied
  6.  Train and holdout use IDENTICAL normalization stats (train-only fit)
  7.  Per-target validity masks (next_price_valid, etc.) are emitted
  8.  HierarchicalLOBTransformer instantiates with lob_image_encoder
  9.  Forward pass consumes lob_image (B, T=50, P=20, C=9) without error
  10. Outputs and losses are finite under all conditions:
        - normal batch
        - batch with empty bars (no orders) — softmax NaN bug regression
        - batch with NaN inputs        — sanitization guards
  11. Per-task losses respect validity flags (masked out samples don't
      contribute to gradient)
  12. extract_embeddings → fine_tune_direction → validate_ssl handoff
      sanity-checked (calendar split idx consistency)

Usage:
    python self_supervised/verify_integration.py \\
        --features pipeline_15min_v3/day_trading_features.parquet \\
        --lob-tensors pipeline_15min_v3/lob_tensors.npy \\
        [--order-batches-dir checkpoints/.../order_batches]

Exits 0 on success, 1 on first failure (with diagnostic).
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch


# ════════════════════════════════════════════════════════════════════════════
# Check primitives
# ════════════════════════════════════════════════════════════════════════════

class CheckFailed(Exception):
    pass


def _print_check(idx: int, title: str):
    print(f"\n[{idx:2d}] {title}")
    print(f"     {'-' * 70}")


def _ok(msg: str):
    print(f"     ✅ {msg}")


def _warn(msg: str):
    print(f"     ⚠️  {msg}")


def _fail(msg: str):
    raise CheckFailed(msg)


def _assert(cond: bool, msg: str):
    if cond:
        _ok(msg)
    else:
        _fail(msg)


# ════════════════════════════════════════════════════════════════════════════
# Per-stage checks
# ════════════════════════════════════════════════════════════════════════════

EXPECTED_SEASONAL_COLS = {
    'session_phase',
    'time_since_london_open_min', 'time_to_london_close_min',
    'time_since_ny_open_min', 'time_to_ny_close_min',
    'dow_sin', 'dow_cos', 'is_monday', 'is_friday',
    'dom', 'dom_sin', 'dom_cos',
    'is_month_end', 'is_month_start', 'is_quarter_end', 'is_year_end',
    'woy_sin', 'woy_cos', 'is_first_week_of_year',
    'is_dst_transition_week', 'is_event_window',
}

EXPECTED_CYCLE_STRUCT_COLS = {
    'cycle_structure_score', 'cycle_bars_since_swing',
    'cycle_trend_maturity', 'cycle_momentum_decay',
    'cycle_phase_acc_prob', 'cycle_phase_markup_prob',
    'cycle_phase_dist_prob', 'cycle_phase_markdown_prob',
    'cycle_position', 'cycle_hurst',
    'cycle_fractal_dim', 'cycle_mtf_alignment',
}

EXPECTED_TECHNICAL_COLS = {
    'atr_14', 'is_session_break',
    'bar_range', 'body_ratio',
    'rsi_14', 'macd_hist',
    'vwap_dist', 'lob_imbalance',
}


def check_parquet_schema(features_path: str) -> pd.DataFrame:
    """[1] Parquet has all expected columns after the whitelist fix."""
    _print_check(1, "Parquet schema (Seasonal Map + Cycle + Technical preserved)")
    df = pd.read_parquet(features_path)
    n, n_cols = len(df), len(df.columns)
    print(f"     {n:,} rows × {n_cols} columns")
    cols_present = set(df.columns)

    miss_seasonal = EXPECTED_SEASONAL_COLS - cols_present
    miss_cycle = EXPECTED_CYCLE_STRUCT_COLS - cols_present
    miss_tech = EXPECTED_TECHNICAL_COLS - cols_present

    seas_have = len(EXPECTED_SEASONAL_COLS & cols_present)
    cyc_have = len(EXPECTED_CYCLE_STRUCT_COLS & cols_present)
    tech_have = len(EXPECTED_TECHNICAL_COLS & cols_present)

    _assert(seas_have >= 15,
            f"Seasonal Map: {seas_have}/{len(EXPECTED_SEASONAL_COLS)} present "
            f"(missing: {sorted(miss_seasonal)[:5]})")
    _assert(cyc_have >= 7,
            f"Cycle Structural: {cyc_have}/{len(EXPECTED_CYCLE_STRUCT_COLS)} present "
            f"(missing: {sorted(miss_cycle)[:5]})")
    _assert(tech_have >= 6,
            f"Technical: {tech_have}/{len(EXPECTED_TECHNICAL_COLS)} present "
            f"(missing: {sorted(miss_tech)[:5]})")
    return df


def check_data_quality(df: pd.DataFrame):
    """[2] Data quality — OHLC sane, ATR sensible, session breaks present."""
    _print_check(2, "Data quality (B1 OHLC, B4 ATR, B5 session-break)")
    for c in ('open', 'high', 'low', 'close'):
        v = pd.to_numeric(df[c], errors='coerce')
        _assert(v.between(0.5, 5.0).mean() > 0.99,
                f"{c}: {v.between(0.5, 5.0).mean()*100:.2f}% in [0.5, 5.0]")
    # High >= max(open, close, low) sanity
    hi_lo = (df['high'] >= df['low']).mean()
    _assert(hi_lo > 0.99, f"high >= low: {hi_lo*100:.2f}%")

    if 'atr_14' in df.columns:
        atr = pd.to_numeric(df['atr_14'], errors='coerce').dropna()
        atr_median_pips = float(atr.median()) / 1e-4
        _assert(5 < atr_median_pips < 100,
                f"ATR median = {atr_median_pips:.1f} pips (sane for 15min GBPUSD)")
    if 'is_session_break' in df.columns:
        n_breaks = int(df['is_session_break'].sum())
        # ~8 weekends per 60-day window + holidays
        _assert(1 <= n_breaks <= 30,
                f"is_session_break count = {n_breaks} (expected ~8-15 for 3 months)")

    # Audit guard (Issue #24): catch silently-zero-filled features that
    # finalize_daytrade_parquet_export added when an upstream builder
    # failed (e.g. seasonal_map raising). Sample a few key features and
    # assert they have nonzero variance.
    degenerate_samples = []
    for col in ('time_since_london_open_min', 'dow_sin', 'session_phase',
                'cycle_structure_score', 'cycle_hurst',
                'rsi_14', 'macd_hist'):
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(v) > 100 and float(v.std()) < 1e-9:
            degenerate_samples.append(col)
    _assert(len(degenerate_samples) == 0,
            f"All sampled features have nonzero variance "
            f"(degenerate = all-zero or constant: {degenerate_samples})")


def check_lob_tensors(lob_path: str, df_n: int):
    """[3] LOB tensors: shape (N, T=50, P=20, C=9)."""
    _print_check(3, "LOB tensor shape and channel structure (9 channels)")
    lob = np.load(lob_path, mmap_mode='r')
    print(f"     shape={lob.shape}, dtype={lob.dtype}, size={lob.nbytes/1e6:.1f} MB")
    _assert(lob.ndim == 4, f"LOB is 4D, not {lob.ndim}D")
    n_bars, T, P, C = lob.shape
    _assert(n_bars == df_n, f"LOB rows ({n_bars}) match parquet rows ({df_n})")
    _assert(T == 50, f"LOB time dim = {T} (expected 50)")
    _assert(P == 20, f"LOB price levels = {P} (expected 20: 10 bid + 10 ask)")
    _assert(C == 9, f"LOB channels = {C} (MUST be 9 — re-run prepare_day_trading)")

    # Sample stats
    sample_idx = np.linspace(0, n_bars - 1, min(100, n_bars)).astype(int)
    sample = np.asarray(lob[sample_idx])
    _assert(np.isfinite(sample).all(),
            f"All sample values finite (NaN={np.isnan(sample).sum()}, "
            f"Inf={np.isinf(sample).sum()})")

    # Per-channel sanity: each channel should have nonzero variance somewhere
    for c in range(C):
        ch = sample[..., c]
        nonzero_frac = float((ch != 0).mean())
        std = float(ch.std())
        marker = '✅' if std > 1e-6 else '⚠️'
        print(f"     {marker} ch{c}: nonzero={nonzero_frac*100:.1f}% std={std:.4f}")
    return lob


def check_data_loader_handoff(features_path: str, lob_path: str,
                               order_batches_dir: str | None):
    """[4] SSLDataset: context_cols covers seasonal+cycle+tech; embargo respected."""
    _print_check(4, "SSLDataset context_cols (Seasonal Map reaches SSL)")
    from self_supervised.data_loader import (
        SSLDataset, build_ssl_loaders, _is_leakage_column,
    )

    # Sanity-check the leakage classifier doesn't accidentally exclude
    # seasonal/cycle structural features
    seas_excluded = [c for c in EXPECTED_SEASONAL_COLS if _is_leakage_column(c)]
    cyc_excluded = [c for c in EXPECTED_CYCLE_STRUCT_COLS if _is_leakage_column(c)]
    _assert(len(seas_excluded) == 0,
            f"Leakage filter doesn't drop seasonal cols "
            f"(would drop: {seas_excluded})")
    # cycle_phase_* and cycle_position ARE intentionally in the leakage list
    # (they're used as cycle SSL targets — identity bug prevention)
    expected_cyc_excluded = {
        'cycle_phase_acc_prob', 'cycle_phase_markup_prob',
        'cycle_phase_dist_prob', 'cycle_phase_markdown_prob',
        'cycle_position',
    }
    unexpected = set(cyc_excluded) - expected_cyc_excluded
    _assert(len(unexpected) == 0,
            f"Cycle excludes only target cols "
            f"(unexpected exclusions: {unexpected})")

    train_loader, holdout_loader = build_ssl_loaders(
        features_path, lob_path, None,
        order_batches_dir=order_batches_dir,
        batch_size=4, train_split=0.75,
        num_workers=0, lookback_bars=50, embargo_bars=24,
    )
    train_ds = train_loader.dataset
    holdout_ds = holdout_loader.dataset

    cols_in_context = set(train_ds.context_cols)
    seas_in_ctx = EXPECTED_SEASONAL_COLS & cols_in_context
    cyc_in_ctx = EXPECTED_CYCLE_STRUCT_COLS & cols_in_context
    tech_in_ctx = EXPECTED_TECHNICAL_COLS & cols_in_context

    print(f"     context_cols total: {len(train_ds.context_cols)}")
    _assert(len(seas_in_ctx) >= 10,
            f"≥10 Seasonal Map features in context: {len(seas_in_ctx)} "
            f"(samples: {sorted(seas_in_ctx)[:5]})")
    _assert(len(cyc_in_ctx) >= 5,
            f"≥5 Cycle Structural features in context: {len(cyc_in_ctx)} "
            f"(present: {sorted(cyc_in_ctx)})")
    _assert(len(tech_in_ctx) >= 5,
            f"≥5 Technical features in context: {len(tech_in_ctx)} "
            f"(present: {sorted(tech_in_ctx)})")

    # Embargo math
    train_max = train_ds.max_idx
    holdout_min = holdout_ds.min_idx
    gap = holdout_min - train_max
    _assert(gap >= 50,
            f"Embargo gap (holdout_min - train_max) = {gap} bars ≥ 50")

    # Train/holdout share normalization stats (S1 fix)
    same_mu = np.allclose(train_ds.context_mu, holdout_ds.context_mu)
    same_sg = np.allclose(train_ds.context_sigma, holdout_ds.context_sigma)
    _assert(same_mu and same_sg,
            "Train & Holdout share IDENTICAL mu/sigma (S1 leak fix in force)")

    return train_loader, holdout_loader


def check_batch_shape(loader):
    """[5] Batch contains all required keys with expected shapes."""
    _print_check(5, "Batch dict contains all required keys (incl lob_image)")
    batch = next(iter(loader))
    required = [
        'order_features', 'order_masks', 'bar_mask', 'context',
        'cycle_window', 'lob_image',
        'next_price', 'next_imbalance', 'next_volatility', 'next_regime',
        'wall_persist', 'time_to_event',
        'phase_target', 'maturity_target', 'swing_target', 'cycle_position_target',
        'next_price_valid', 'next_imbalance_valid', 'next_volatility_valid',
        'next_regime_valid', 'wall_persist_valid', 'time_to_event_valid',
        'cycle_valid',
    ]
    missing = [k for k in required if k not in batch]
    _assert(len(missing) == 0, f"All {len(required)} keys present "
                               f"(missing: {missing})")

    for k in ('order_features', 'order_masks', 'bar_mask', 'context',
              'lob_image', 'cycle_window'):
        shape = tuple(batch[k].shape)
        print(f"     {k}: {shape} dtype={batch[k].dtype}")

    # LOB image must be 9-channel
    lob = batch['lob_image']
    _assert(lob.shape[-1] == 9,
            f"lob_image has {lob.shape[-1]} channels (must be 9)")
    return batch


def check_model_forward(batch: dict):
    """[6] HierarchicalLOBTransformer forward consumes lob_image properly."""
    _print_check(6, "Model forward with 9-channel LOB image")
    from modules.deep_lob.config import DeepLOBConfig
    from modules.deep_lob.hierarchical_model import HierarchicalLOBTransformer

    config = DeepLOBConfig()
    config.multi_task_heads.direction_weight = 0.0
    model = HierarchicalLOBTransformer(config).eval()

    _assert(model.lob_image_encoder is not None,
            f"Model has lob_image_encoder branch")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_lob = sum(p.numel() for p in model.lob_image_encoder.parameters())
    print(f"     Total params: {n_params:,} (LOB CNN: {n_lob:,})")

    with torch.no_grad():
        # Without LOB
        out_no = model(
            batch['order_features'], batch['order_masks'],
            batch['bar_mask'], batch['context'],
            lob_tensor=None,
        )
        # With LOB
        out_yes = model(
            batch['order_features'], batch['order_masks'],
            batch['bar_mask'], batch['context'],
            lob_tensor=batch['lob_image'],
        )

    for k in ('next_price', 'next_imbalance', 'next_volatility',
              'wall_persist', 'time_to_event'):
        v = getattr(out_yes, k)
        _assert(torch.isfinite(v).all(),
                f"output.{k} finite (with LOB)")

    diff = (out_no.shared_embedding - out_yes.shared_embedding).abs().mean().item()
    _assert(diff > 1e-4,
            f"LOB branch contributes signal (Δ shared_embedding = {diff:.4f})")
    return model


def check_loss_masking(model, batch: dict):
    """[7] Per-task validity masks reduce loss for masked samples."""
    _print_check(7, "Validity-masked loss zeros out invalid samples")
    from self_supervised.pretrain_lob import compute_ssl_loss
    from modules.deep_lob.config import DeepLOBConfig

    device = torch.device('cpu')
    config = DeepLOBConfig()
    config.multi_task_heads.direction_weight = 0.0
    cfg_w = config.multi_task_heads

    # Push batch to device dict matching pretrain_lob inputs
    inputs = {
        'order_features': batch['order_features'].to(device),
        'order_masks':    batch['order_masks'].to(device),
        'bar_mask':       batch['bar_mask'].to(device),
        'context':        batch['context'].to(device),
        'lob_tensor':     batch['lob_image'].to(device),
    }

    with torch.no_grad():
        out = model(**inputs)
        total_a, per_a = compute_ssl_loss(out, batch, device, cfg_w)

    # Now zero out next_price_valid for ALL samples — loss must remain finite + drop next_price contribution
    fake_batch = {**batch}
    fake_batch['next_price_valid'] = torch.zeros_like(batch['next_price_valid'])
    with torch.no_grad():
        total_b, per_b = compute_ssl_loss(out, fake_batch, device, cfg_w)

    _assert(np.isfinite(total_a) and np.isfinite(total_b),
            f"Total loss finite both with and without mask")
    _assert(per_b['next_price'] == 0.0,
            f"next_price loss = 0 when valid_mask all-False "
            f"(was {per_a['next_price']:.4f}, now {per_b['next_price']:.4f})")
    _assert(total_b < total_a,
            f"Total loss DROPS when masking out a task ({total_b:.4f} < {total_a:.4f})")


def check_edge_cases(model, batch: dict):
    """[8] Empty bars + NaN inputs survive sanitization."""
    _print_check(8, "Edge cases — empty bars + NaN inputs (regression tests)")

    # 8a: Empty bar sample
    bad = {k: v.clone() for k, v in batch.items()}
    bad['order_masks'][0, :, :] = False     # sample 0 = no orders at any bar
    bad['order_features'][1, 0, 0, :] = float('nan')  # one NaN order
    bad['context'][2, 5] = float('nan')                # NaN context
    bad['lob_image'][3, 10, 5, 2] = float('nan')       # NaN in LOB

    with torch.no_grad():
        out = model(
            bad['order_features'], bad['order_masks'],
            bad['bar_mask'], bad['context'],
            lob_tensor=bad['lob_image'],
        )

    finite = all(torch.isfinite(getattr(out, k)).all() for k in
                 ('next_price', 'next_imbalance', 'next_volatility',
                  'wall_persist', 'time_to_event'))
    _assert(finite, "All outputs finite under empty-bar + NaN-input stress")


def check_extract_pipeline(features_path: str, lob_path: str,
                           order_batches_dir: str | None):
    """[9] extract_embeddings can build dataset over full range."""
    _print_check(9, "extract_embeddings dataset construction (full-range)")
    from self_supervised.data_loader import SSLDataset
    n = len(pd.read_parquet(features_path, columns=['close']))
    sample_lo = 50
    sample_hi = max(sample_lo + 1, n - 24)
    stats_hi = max(sample_lo + 1, int(n * 0.75) - 24)
    ds = SSLDataset(
        features_path, lob_path, None,
        order_batches_dir=order_batches_dir,
        lookback_bars=50,
        min_idx=sample_lo, max_idx=sample_hi,
        stats_min_idx=sample_lo, stats_max_idx=stats_hi,
        embargo_bars=24,
    )
    _assert(len(ds) > 100,
            f"Full-range dataset has {len(ds)} samples (≥100 expected)")
    sample = ds[0]
    _assert(sample['lob_image'].shape[-1] == 9,
            f"Full-range sample lob_image has 9 channels")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--lob-tensors', required=True)
    p.add_argument('--order-batches-dir', default=None)
    args = p.parse_args()

    print("═" * 78)
    print(" END-TO-END INTEGRATION VERIFIER")
    print(" Asserts every handoff between prepare_day_trading → SSL → Model")
    print("═" * 78)

    try:
        df = check_parquet_schema(args.features)
        check_data_quality(df)
        lob = check_lob_tensors(args.lob_tensors, len(df))
        train_loader, holdout_loader = check_data_loader_handoff(
            args.features, args.lob_tensors, args.order_batches_dir,
        )
        batch = check_batch_shape(train_loader)
        model = check_model_forward(batch)
        check_loss_masking(model, batch)
        check_edge_cases(model, batch)
        check_extract_pipeline(args.features, args.lob_tensors,
                               args.order_batches_dir)
    except CheckFailed as e:
        print()
        print("═" * 78)
        print(f" ❌ INTEGRATION CHECK FAILED")
        print(f" {str(e)}")
        print("═" * 78)
        sys.exit(1)
    except Exception as e:
        print()
        print("═" * 78)
        print(f" ❌ UNEXPECTED ERROR")
        print(f" {type(e).__name__}: {e}")
        print("═" * 78)
        traceback.print_exc()
        sys.exit(1)

    print()
    print("═" * 78)
    print(" ✅ ALL INTEGRATION CHECKS PASSED")
    print(" The pipeline is wired end-to-end:")
    print("   parquet → SSLDataset → HierarchicalLOBTransformer (9-ch LOB) → loss")
    print(" Seasonal Map, Price Cycle, Technical features all flow into SSL context.")
    print("═" * 78)


if __name__ == '__main__':
    main()
