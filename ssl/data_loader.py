"""
ssl/data_loader.py
══════════════════════════════════════════════════════════
SSL Data Loader — يحوّل output prepare_day_trading.py إلى
batches جاهزة للـ pretraining للـ LOB Transformer و Price Cycle.

الـ workflow:
    prepare_day_trading → features.parquet + lob_tensors.npy
                                ↓
    SSLDataset.__init__(features_path, lob_tensors_path)
                                ↓
    DataLoader(dataset, batch_size=64) → batches:
        - LOB Transformer batches: (order_features, masks, context)
        - Price Cycle batches: (bar_features, structural_features)
        - SSL targets (no direction labels needed):
            next_price, next_imbalance, next_volatility, next_regime,
            wall_persist, time_to_event,
            phase, maturity, swing, cycle_position
"""
from __future__ import annotations

import os
from typing import Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Default regime code mapping (matches regime_config.REGIMES order)
REGIME_TO_CODE = {
    'trending'     : 0,
    'ranging'      : 1,
    'volatile'     : 2,
    'low_liquidity': 3,
}


def _build_context_features(
    df_row: pd.Series, available_cols: list[str], target_dim: int = 138,
) -> np.ndarray:
    """يبني context vector من DataFrame row (138-dim default)."""
    values = []
    for col in available_cols:
        v = df_row.get(col, 0.0)
        try:
            values.append(float(v) if pd.notna(v) else 0.0)
        except (ValueError, TypeError):
            values.append(0.0)
    arr = np.array(values, dtype=np.float32)
    if len(arr) < target_dim:
        arr = np.concatenate([arr, np.zeros(target_dim - len(arr), dtype=np.float32)])
    else:
        arr = arr[:target_dim]
    return arr


def _lob_tensor_to_orders_vectorized(
    lob_window: np.ndarray, n_orders: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized version of DeepLOBCNNAdapter._lob_to_orders.

    Input:  lob_window (T, P=20, C=3)
    Output: order_features (T, P, 7), order_masks (T, P) bool
    """
    T, P, C = lob_window.shape
    order_features = np.zeros((T, P, 7), dtype=np.float32)
    sides = np.where(np.arange(P) < P // 2, 0.0, 1.0)
    price_dist = (np.arange(P) - P / 2.0).astype(np.float32)
    for c in range(min(C, 3)):
        channel_data = lob_window[:, :, c]
        log_abs = np.log1p(np.abs(channel_data) + 1e-6)
        if c == 0:
            order_features[:, :, 2] = log_abs.astype(np.float32)
        elif c == 1:
            order_features[:, :, 5] = log_abs.astype(np.float32)
        else:
            order_features[:, :, 6] = log_abs.astype(np.float32)
    order_features[:, :, 0] = sides[np.newaxis, :]
    order_features[:, :, 3] = price_dist[np.newaxis, :]
    order_features[:, :, 4] = np.arange(T, dtype=np.float32)[:, np.newaxis]
    order_masks = np.ones((T, P), dtype=bool)
    return order_features, order_masks


def _compute_phase_target(df: pd.DataFrame, idx: int) -> int:
    """Wyckoff phase target من cycle_phase_*_prob columns (PR #18 features)."""
    probs = []
    for name in ('acc', 'markup', 'dist', 'markdown'):
        col = f'cycle_phase_{name}_prob'
        probs.append(float(df.iloc[idx].get(col, 0.0)))
    if sum(probs) <= 1e-6:
        return 0
    return int(np.argmax(probs))


def _compute_swing_target(df: pd.DataFrame, idx: int) -> int:
    """Swing direction target."""
    score = float(df.iloc[idx].get('cycle_structure_score', 0.0))
    if score > 0.3:
        return 0  # up
    if score < -0.3:
        return 1  # down
    return 2  # neutral


def _compute_maturity_target(df: pd.DataFrame, idx: int) -> int:
    """Trend maturity target."""
    mat = float(df.iloc[idx].get('cycle_trend_maturity', 0.0))
    return int(np.clip(round(mat * 2.0), 0, 2))


def _compute_wall_persist_array(df: pd.DataFrame, lookahead: int = 12) -> np.ndarray:
    """عدد bars حتى OBI يقلب الإشارة — vectorized."""
    n = len(df)
    if 'obi_direction' not in df.columns:
        return np.zeros(n, dtype=np.float32)
    obi = df['obi_direction'].to_numpy(dtype=np.int8)
    wall = np.zeros(n, dtype=np.float32)
    for i in range(n - 1):
        cur = int(obi[i])
        if cur == 0:
            continue
        end = min(i + lookahead, n)
        future = obi[i + 1: end]
        flip_mask = future != cur
        if flip_mask.any():
            wall[i] = float(np.argmax(flip_mask) + 1)
        else:
            wall[i] = float(end - i - 1)
    return wall


def _compute_time_to_event_array(df: pd.DataFrame, lookahead: int = 24) -> np.ndarray:
    """عدد bars حتى next is_event=1 — vectorized."""
    n = len(df)
    if 'is_event' not in df.columns:
        return np.full(n, float(lookahead), dtype=np.float32)
    is_event = df['is_event'].astype(bool).to_numpy()
    tte = np.full(n, float(lookahead), dtype=np.float32)
    for i in range(n - 1):
        end = min(i + lookahead, n)
        future = is_event[i + 1: end]
        if future.any():
            tte[i] = float(np.argmax(future) + 1)
        else:
            tte[i] = float(end - i - 1)
    return tte


class SSLDataset(Dataset):
    """Dataset for SSL pretraining من output prepare_day_trading.py.

    كل sample يحتوي على:
        - إما LOB tensor (T=50, P=20, C=3) — pseudo-orders mode (lossy)
        - أو OrderBatch (T=50, N=200, F=7) — real orders mode (الأفضل — iceberg, etc.)
        - Context features (138-dim من DataFrame row)
        - SSL targets (10 targets، كلها مشتقة من البيانات، لا labels يدوية)

    لا يستخدم bias_label أبداً — هذا هو نقطة SSL.

    Order Mode:
        - Real orders (recommended): pass order_batches_dir to use raw MBO orders
        - Pseudo-orders (fallback): builds from LOB tensors (loses iceberg info)
    """

    def __init__(
        self,
        features_parquet: str,
        lob_tensors_path: str,
        lob_timestamps_path: Optional[str] = None,
        *,
        order_batches_dir: Optional[str] = None,   # ← NEW: real orders dir
        lookback_bars: int = 50,
        n_orders: int = 20,
        context_dim: int = 138,
        min_idx: Optional[int] = None,
        max_idx: Optional[int] = None,
    ):
        self.lookback_bars = lookback_bars
        self.n_orders = n_orders
        self.context_dim = context_dim
        self.order_batches_dir = order_batches_dir
        self.use_real_orders = order_batches_dir is not None

        print(f"  📂 Loading {features_parquet}...")
        self.df = pd.read_parquet(features_parquet)
        print(f"     {len(self.df):,} rows × {len(self.df.columns)} columns")

        print(f"  📂 Loading {lob_tensors_path}...")
        self.lob_tensors = np.load(lob_tensors_path, mmap_mode='r')
        print(f"     shape={self.lob_tensors.shape}")

        if len(self.df) != len(self.lob_tensors):
            raise ValueError(
                f"Length mismatch: df={len(self.df)} vs lob={len(self.lob_tensors)}"
            )

        # ── NEW: Load real OrderBatches if provided ──
        if self.use_real_orders:
            ob_features = os.path.join(order_batches_dir, 'order_features.npy')
            ob_masks = os.path.join(order_batches_dir, 'order_masks.npy')
            if not (os.path.exists(ob_features) and os.path.exists(ob_masks)):
                print(f"  ⚠️  OrderBatches not found in {order_batches_dir}")
                print(f"      → falling back to pseudo-orders from LOB tensors")
                self.use_real_orders = False
            else:
                print(f"  📂 Loading real OrderBatches from {order_batches_dir}...")
                self.order_features_arr = np.load(ob_features, mmap_mode='r')
                self.order_masks_arr = np.load(ob_masks, mmap_mode='r')
                print(f"     order_features: {self.order_features_arr.shape}")
                print(f"     order_masks: {self.order_masks_arr.shape}")
                # Override n_orders to match
                self.n_orders = self.order_features_arr.shape[2]
                print(f"     ✅ Real orders mode enabled (n_orders={self.n_orders})")

        self.min_idx = max(lookback_bars, min_idx or 0)
        self.max_idx = min(len(self.df) - 25, max_idx or len(self.df))
        self.n_samples = max(0, self.max_idx - self.min_idx)
        print(f"     valid samples: {self.n_samples:,} (idx {self.min_idx}..{self.max_idx})")

        # Precompute regime codes
        if 'regime_label' in self.df.columns:
            self.regime_codes = self.df['regime_label'].astype(str).map(REGIME_TO_CODE).fillna(1).astype(np.int64).to_numpy()
        else:
            self.regime_codes = np.ones(len(self.df), dtype=np.int64)

        # Identify context feature columns
        exclude_cols = {
            'ts_event', 'label_end_ts', 'bias_label', 'path_outcome',
            'neutral_reason', 'soft_label', 'conf_target', 'event_label_tier',
            'is_event', 'train_event_flag', 'regime_label', 'regime_cluster',
            'market_state_label', 'market_state_code', 'neutral_type',
            'tradability_label', 'event_direction', 'kalman_direction',
        }
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.context_cols = [c for c in numeric_cols if c not in exclude_cols][:context_dim]
        print(f"     context features: {len(self.context_cols)} (capped at {context_dim})")

        self._precompute_targets()

    def _precompute_targets(self):
        """Precompute SSL targets للسرعة."""
        n = len(self.df)
        print(f"  ⚙️  Precomputing SSL targets...")

        close = pd.to_numeric(self.df['close'], errors='coerce').to_numpy(dtype=np.float64)
        log_ret = np.zeros(n, dtype=np.float32)
        log_ret[:-1] = np.log(np.maximum(close[1:], 1e-9) / np.maximum(close[:-1], 1e-9)).astype(np.float32)
        self.next_price = log_ret

        obi_col = 'obi_net' if 'obi_net' in self.df.columns else 'order_flow_imbalance'
        obi = pd.to_numeric(self.df.get(obi_col, 0.0), errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        obi = np.clip(obi, -1.0, 1.0)
        next_obi = np.zeros(n, dtype=np.float32)
        next_obi[:-1] = obi[1:]
        self.next_imbalance = next_obi

        atr_col = 'atr_14' if 'atr_14' in self.df.columns else 'atr'
        atr = pd.to_numeric(self.df.get(atr_col, 0.001), errors='coerce').fillna(0.001).to_numpy(dtype=np.float32)
        next_atr = np.zeros(n, dtype=np.float32)
        next_atr[:-1] = atr[1:]
        self.next_volatility = np.maximum(next_atr, 1e-6)

        next_regime = np.ones(n, dtype=np.int64)
        next_regime[:-1] = self.regime_codes[1:]
        self.next_regime = next_regime

        self.wall_persist = _compute_wall_persist_array(self.df)
        self.time_to_event = _compute_time_to_event_array(self.df)

        # Cycle targets
        phase = np.zeros(n, dtype=np.int64)
        swing = np.zeros(n, dtype=np.int64)
        maturity = np.zeros(n, dtype=np.int64)
        cycle_pos = np.zeros(n, dtype=np.float32)
        for i in range(n):
            phase[i] = _compute_phase_target(self.df, i)
            swing[i] = _compute_swing_target(self.df, i)
            maturity[i] = _compute_maturity_target(self.df, i)
            cycle_pos[i] = float(self.df.iloc[i].get('cycle_position', 0.0))
        self.phase_target = phase
        self.swing_target = swing
        self.maturity_target = maturity
        self.cycle_position_target = cycle_pos

        print(f"     ✅ SSL targets ready (10 tasks × {n:,} bars)")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, item: int) -> dict:
        idx = self.min_idx + item

        lob_window = np.asarray(self.lob_tensors[idx], dtype=np.float32)

        # ── Order features: real orders (NEW) or pseudo-orders (fallback) ──
        if self.use_real_orders:
            # Real orders from MBO: preserves order IDs, iceberg signals, etc.
            order_features = np.asarray(self.order_features_arr[idx], dtype=np.float32)
            order_masks = np.asarray(self.order_masks_arr[idx], dtype=bool)
        else:
            # Pseudo-orders from LOB tensor (lossy — no order IDs, no iceberg)
            order_features, order_masks = _lob_tensor_to_orders_vectorized(
                lob_window, n_orders=self.n_orders,
            )
        bar_mask = np.ones(lob_window.shape[0], dtype=bool)
        context = _build_context_features(
            self.df.iloc[idx], self.context_cols, target_dim=self.context_dim,
        )

        # Cycle window
        cycle_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'cycle_structure_score', 'cycle_trend_maturity',
            'cycle_phase_acc_prob', 'cycle_phase_markup_prob',
            'cycle_phase_dist_prob', 'cycle_phase_markdown_prob',
            'cycle_hurst', 'cycle_fractal_dim', 'cycle_mtf_alignment',
        ]
        avail_cycle = [c for c in cycle_cols if c in self.df.columns]
        T = lob_window.shape[0]
        start = max(0, idx - T + 1)
        end = idx + 1
        cycle_window = self.df[avail_cycle].iloc[start:end].to_numpy(dtype=np.float32)
        if cycle_window.shape[0] < T:
            pad = np.zeros((T - cycle_window.shape[0], cycle_window.shape[1]), dtype=np.float32)
            cycle_window = np.concatenate([pad, cycle_window], axis=0)

        return {
            'order_features': torch.from_numpy(order_features),
            'order_masks': torch.from_numpy(order_masks),
            'bar_mask': torch.from_numpy(bar_mask),
            'context': torch.from_numpy(context),
            'cycle_window': torch.from_numpy(cycle_window),
            'next_price': torch.tensor(self.next_price[idx], dtype=torch.float32),
            'next_imbalance': torch.tensor(self.next_imbalance[idx], dtype=torch.float32),
            'next_volatility': torch.tensor(self.next_volatility[idx], dtype=torch.float32),
            'next_regime': torch.tensor(self.next_regime[idx], dtype=torch.long),
            'wall_persist': torch.tensor(self.wall_persist[idx], dtype=torch.float32),
            'time_to_event': torch.tensor(self.time_to_event[idx], dtype=torch.float32),
            'phase_target': torch.tensor(self.phase_target[idx], dtype=torch.long),
            'maturity_target': torch.tensor(self.maturity_target[idx], dtype=torch.long),
            'swing_target': torch.tensor(self.swing_target[idx], dtype=torch.long),
            'cycle_position_target': torch.tensor(self.cycle_position_target[idx], dtype=torch.float32),
        }


def build_ssl_loaders(
    features_parquet: str,
    lob_tensors_path: str,
    lob_timestamps_path: Optional[str] = None,
    *,
    order_batches_dir: Optional[str] = None,    # ← NEW: real orders
    batch_size: int = 32,
    train_split: float = 0.75,
    num_workers: int = 0,
    lookback_bars: int = 50,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build train + holdout SSL data loaders بـ time-based split.

    train_split=0.75 يعني أول 75% train، آخر 25% holdout.
    لا random shuffle عبر الـ split — للحفاظ على causality.

    order_batches_dir: لو متوفر، يستخدم real orders من MBO خام بدل pseudo-orders.
        مهم لـ iceberg detection و الحفاظ على معلومات order_id/action.
    """
    df = pd.read_parquet(features_parquet)
    n = len(df)
    split_idx = int(n * train_split)
    del df

    print(f"⚙️  Building SSL data loaders...")
    print(f"   Time split: train [0..{split_idx}], holdout [{split_idx}..{n}]")
    print(f"   Order mode: {'REAL (from MBO)' if order_batches_dir else 'pseudo (from LOB tensor)'}")

    train_ds = SSLDataset(
        features_parquet, lob_tensors_path, lob_timestamps_path,
        order_batches_dir=order_batches_dir,
        lookback_bars=lookback_bars,
        min_idx=lookback_bars,
        max_idx=split_idx,
    )
    holdout_ds = SSLDataset(
        features_parquet, lob_tensors_path, lob_timestamps_path,
        order_batches_dir=order_batches_dir,
        lookback_bars=lookback_bars,
        min_idx=split_idx,
        max_idx=n - 25,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    holdout_loader = DataLoader(
        holdout_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    return train_loader, holdout_loader


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--lob-tensors', required=True)
    p.add_argument('--lob-timestamps', default=None)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()

    train_loader, holdout_loader = build_ssl_loaders(
        args.features, args.lob_tensors, args.lob_timestamps,
        batch_size=args.batch_size,
    )
    print(f"\n✅ DataLoaders ready")
    print(f"   train batches: {len(train_loader)}")
    print(f"   holdout batches: {len(holdout_loader)}")
    print(f"\n📦 First batch sample:")
    batch = next(iter(train_loader))
    for k, v in batch.items():
        print(f"   {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
