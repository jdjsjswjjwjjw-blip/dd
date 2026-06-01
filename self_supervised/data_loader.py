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

# ════════════════════════════════════════════════════════════════════════════
# LEAKAGE BLACKLISTS — columns / prefixes that must NEVER enter context
# features for SSL.
#
# Three categories:
#   1. Future-looking labels (computed by walking forward from bar i)
#   2. Derived label diagnostics (depend on labels above)
#   3. Metadata / housekeeping (not features)
#
# Maintain fail-closed: if you add a new column to the parquet that depends
# on FUTURE bars, append it to one of these sets. Anything not in here is
# assumed past-only / causal.
# ════════════════════════════════════════════════════════════════════════════

# Exact column names that are forward-looking labels or label-derived
_LEAKAGE_COLS_EXACT: frozenset[str] = frozenset({
    # Targets / labels
    'bias_label', 'bias_label_detail', 'soft_label', 'soft_label_long',
    'soft_label_short', 'path_outcome', 'neutral_reason', 'neutral_type',
    'conf_target', 'event_label_tier', 'event_direction',
    # Forward returns / horizons (computed by looking i+1..i+H)
    'forward_return', 'fwd_ret_clean', 'effective_horizon',
    'label_horizon_steps', 'label_confidence', 'label_dynamic_threshold',
    'label_stability', 'label_end_ts',
    # Sampling weights derived from labels
    'soft_sample_weight', 'mc_sample_weight', 'sample_weight',
    # Outcome flags (forward window)
    'adverse_path_flag', 'timeout_move_exceeded_band', 'signal_quality',
    'trade_duration',
    # Direction-of-event (after label decision)
    'kalman_direction',
    # Split markers (not features)
    'is_train_slice', 'is_holdout_slice', 'is_purged_slice', 'dataset_slice',
    # State labels (categorical, used elsewhere)
    'regime_label', 'regime_cluster', 'market_state_label', 'market_state_code',
    'tradability_label',
    # Event-flagging (we keep is_event out because event detection often uses
    # forward signal in some pipelines; safer to exclude than to gamble)
    'is_event', 'event_flag', 'event_score', 'train_event_flag',
    # Timestamps
    'ts_event',
    # Cycle phase probabilities — these go in cycle_window or are SSL TARGETS,
    # not in context. Including them in context would let the LOB model
    # cheat by reading the cycle SSL target's input directly.
    'cycle_phase_acc_prob', 'cycle_phase_markup_prob',
    'cycle_phase_dist_prob', 'cycle_phase_markdown_prob',
    'cycle_position',
    # C1: Multi-task diagnostic columns (compute_multitask_label_diagnostics).
    # These look forward over horizon_bars*4 windows — leakage if used as
    # SSL features. Previously uncovered: only the `mfe_*`/`mae_*` PREFIX
    # was guarded, but the bare `mfe`/`mae` columns and the other
    # diagnostics slipped through. Adding them by name closes the gap.
    'mfe', 'mae', 'stop_first_flag', 'time_to_first_touch',
    'net_expectancy_proxy',
    # C1: Dual-target heads B1/B2 — strict exec labels + continuous SSL
    # directional target. Both look forward, both are training targets,
    # neither may appear in the SSL feature vector.
    'exec_label', 'exec_path', 'exec_valid',
    'next_price_delta', 'next_price_delta_valid',
})

# Any column whose name starts with one of these prefixes is excluded
_LEAKAGE_PREFIXES: tuple[str, ...] = (
    'label_',      # any label_* — by convention all are forward-looking
    'soft_label',  # soft_label_*
    'forward_',    # forward_return etc.
    'fwd_',        # fwd_ret_clean
    'mfe_',        # max favourable excursion — forward only
    'mae_',        # max adverse excursion — forward only
    # C1: future-extension guards so any new B1/B2-family column added to
    # the parquet is excluded by default (fail-closed)
    'exec_',       # exec_label / exec_path / exec_valid + any sibling
    'next_price_', # next_price_delta / next_price_delta_valid + any sibling
)


def _is_leakage_column(col: str) -> bool:
    """True iff column should be excluded from SSL context features."""
    if col in _LEAKAGE_COLS_EXACT:
        return True
    for pfx in _LEAKAGE_PREFIXES:
        if col.startswith(pfx):
            return True
    return False


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


def _compute_phase_target_at(df: pd.DataFrame, idx: int) -> int:
    """Wyckoff phase computed AT bar idx (used by next-bar shift below)."""
    probs = []
    for name in ('acc', 'markup', 'dist', 'markdown'):
        col = f'cycle_phase_{name}_prob'
        probs.append(float(df.iloc[idx].get(col, 0.0)))
    if sum(probs) <= 1e-6:
        return 0
    return int(np.argmax(probs))


def _compute_swing_target_at(df: pd.DataFrame, idx: int) -> int:
    """Swing direction AT bar idx (used by next-bar shift below)."""
    score = float(df.iloc[idx].get('cycle_structure_score', 0.0))
    if score > 0.3:
        return 0  # up
    if score < -0.3:
        return 1  # down
    return 2  # neutral


def _compute_maturity_target_at(df: pd.DataFrame, idx: int) -> int:
    """Trend maturity AT bar idx (used by next-bar shift below)."""
    mat = float(df.iloc[idx].get('cycle_trend_maturity', 0.0))
    return int(np.clip(round(mat * 2.0), 0, 2))


def _compute_wall_persist_array(
    df: pd.DataFrame, lookahead: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Bars until OBI direction flips.

    Returns (wall_persist, valid_mask):
        wall_persist : (n,) float — bars-to-flip, capped at lookahead
        valid_mask   : (n,) bool — True iff the lookahead window is fully
                       within the dataset AND we have a non-zero direction.

    Bars whose full lookahead window would extend past the end of the
    dataset have valid_mask=False — they should be excluded from training
    so the truncated target doesn't bias the model.
    """
    n = len(df)
    wall = np.zeros(n, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    if 'obi_direction' not in df.columns:
        return wall, valid
    obi = df['obi_direction'].to_numpy(dtype=np.int8)
    for i in range(n):
        cur = int(obi[i])
        if cur == 0:
            continue
        end = i + 1 + lookahead
        if end > n:  # incomplete lookahead — invalid
            continue
        future = obi[i + 1: end]
        flip_mask = future != cur
        wall[i] = float(np.argmax(flip_mask) + 1) if flip_mask.any() else float(lookahead)
        valid[i] = True
    return wall, valid


def _compute_time_to_event_array(
    df: pd.DataFrame, lookahead: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Bars until next is_event=1.

    Returns (tte, valid_mask). Same boundary semantics as wall_persist.
    """
    n = len(df)
    tte = np.full(n, float(lookahead), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    if 'is_event' not in df.columns:
        return tte, valid
    is_event = df['is_event'].astype(bool).to_numpy()
    for i in range(n):
        end = i + 1 + lookahead
        if end > n:
            continue
        future = is_event[i + 1: end]
        tte[i] = float(np.argmax(future) + 1) if future.any() else float(lookahead)
        valid[i] = True
    return tte, valid


# ── Sample-size thresholds ──────────────────────────────────────────
# Below these counts the SSL training will overfit / fail to converge.
# Numbers chosen so the warning fires at "I would not trust this", and
# the error fires at "this run is guaranteed garbage".
SSL_MIN_SAMPLES_WARN: int = 30_000     # ~6 months at 5-min bars after embargo
SSL_MIN_SAMPLES_ERROR: int = 5_000     # below this, refuse to instantiate

# Allow operators to override via env var for tiny smoke / synthetic runs:
#   SSL_BYPASS_MIN_SAMPLES=1  → skip the hard error (warn-only)
# This is intentionally an opt-in escape hatch, not a config flag.
import os as _os
_BYPASS_MIN_SAMPLES_ENV = "SSL_BYPASS_MIN_SAMPLES"


def _check_sample_size(n_samples: int) -> None:
    """Loud-fail on degenerate sample counts; warn on suspicious ones.

    Pulled out as a free function so the smoke verifier
    (`verify_data_health.py`) can call the same logic.
    """
    if n_samples < SSL_MIN_SAMPLES_ERROR:
        if _os.environ.get(_BYPASS_MIN_SAMPLES_ENV, "") == "1":
            print(
                f"  ⚠️  Total Effective Samples: {n_samples:,} — "
                f"below ERROR floor ({SSL_MIN_SAMPLES_ERROR:,}). "
                f"Bypassed via {_BYPASS_MIN_SAMPLES_ENV}=1."
            )
            return
        raise RuntimeError(
            f"SSLDataset has only {n_samples:,} effective samples — "
            f"below the hard floor of {SSL_MIN_SAMPLES_ERROR:,}. "
            f"Check embargo_bars / session_breaks / min_idx in "
            f"the calling script. Set {_BYPASS_MIN_SAMPLES_ENV}=1 to "
            f"bypass for smoke tests with synthetic data."
        )
    if n_samples < SSL_MIN_SAMPLES_WARN:
        print(
            f"  ⚠️  Total Effective Samples: {n_samples:,} — "
            f"BELOW recommended {SSL_MIN_SAMPLES_WARN:,}. SSL likely to "
            f"overfit. Inspect embargo / session breaks / leakage filter "
            f"before launching a full training run."
        )
    else:
        print(f"  ✅ Total Effective Samples: {n_samples:,}")


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
        order_batches_dir: Optional[str] = None,
        lookback_bars: int = 50,
        n_orders: int = 20,
        context_dim: int = 138,
        min_idx: Optional[int] = None,
        max_idx: Optional[int] = None,
        # Train-period bounds for fitting normalization stats. If None,
        # defaults to (lookback_bars, max_idx) — i.e. only this Dataset's
        # own sample range. CRITICAL: pass an explicit train range when
        # building a holdout Dataset, so holdout doesn't compute its own
        # stats over leaked future data.
        stats_min_idx: Optional[int] = None,
        stats_max_idx: Optional[int] = None,
        # Right-edge embargo: drop the last `embargo_bars` valid sample
        # indices because their forward-looking targets (next_price,
        # wall_persist@12, time_to_event@24) reach beyond the parquet end.
        # Must be ≥ max(target_lookahead, 1).
        embargo_bars: int = 24,
    ):
        self.lookback_bars = lookback_bars
        self.n_orders = n_orders
        self.context_dim = context_dim
        self.order_batches_dir = order_batches_dir
        self.use_real_orders = order_batches_dir is not None
        self.embargo_bars = int(embargo_bars)

        print(f"  📂 Loading {features_parquet}...")
        self.df = pd.read_parquet(features_parquet)
        print(f"     {len(self.df):,} rows × {len(self.df.columns)} columns")

        print(f"  📂 Loading {lob_tensors_path}...")
        # D2 fix: avoid mmap_mode='r' — multi-process DataLoader workers
        # fork shared file descriptors and can hit data corruption races
        # on Linux. Loading into memory is safe for typical SSL dataset
        # sizes (~1GB for 6 months of 15min × 50 × 20 × 3 × fp32).
        self.lob_tensors = np.load(lob_tensors_path)
        print(f"     shape={self.lob_tensors.shape}, "
              f"size={self.lob_tensors.nbytes / 1e9:.2f} GB")

        if len(self.df) != len(self.lob_tensors):
            raise ValueError(
                f"Length mismatch: df={len(self.df)} vs lob={len(self.lob_tensors)}"
            )

        # ── Load real OrderBatches if provided ──
        if self.use_real_orders:
            ob_features = os.path.join(order_batches_dir, 'order_features.npy')
            ob_masks = os.path.join(order_batches_dir, 'order_masks.npy')
            if not (os.path.exists(ob_features) and os.path.exists(ob_masks)):
                print(f"  ⚠️  OrderBatches not found in {order_batches_dir}")
                print(f"      → falling back to pseudo-orders from LOB tensors")
                self.use_real_orders = False
            else:
                print(f"  📂 Loading real OrderBatches from {order_batches_dir}...")
                # D2 fix: see above. ~1.16 GB for typical SSL setup.
                self.order_features_arr = np.load(ob_features)
                self.order_masks_arr = np.load(ob_masks)
                print(f"     order_features: {self.order_features_arr.shape}, "
                      f"size={self.order_features_arr.nbytes / 1e9:.2f} GB")
                print(f"     order_masks: {self.order_masks_arr.shape}")
                self.n_orders = self.order_features_arr.shape[2]
                print(f"     ✅ Real orders mode enabled (n_orders={self.n_orders})")

        # Sample boundaries: clip max_idx by embargo so all forward-looking
        # targets are fully observable. lookback_bars guards the left edge.
        n_total = len(self.df)
        self.min_idx = max(lookback_bars, min_idx or 0)
        upper_bound = n_total - self.embargo_bars
        self.max_idx = min(upper_bound, max_idx if max_idx is not None else upper_bound)
        self.n_samples = max(0, self.max_idx - self.min_idx)
        print(f"     valid samples: {self.n_samples:,} "
              f"(idx {self.min_idx}..{self.max_idx}, embargo={self.embargo_bars})")

        # ── Segment-aware sample filter ──
        # If the parquet contains `is_session_break` (= weekend gaps and
        # quarterly-rollover gaps), filter out samples whose lookback window
        # OR forward-target window crosses a break. Prevents cross-segment
        # feature pollution (ATR rolling, LOB tensor history, cycle features)
        # that would otherwise mix data from two different contracts.
        self._build_segment_mask(n_total)
        # Build the final list of valid sample indices
        candidate = np.arange(self.min_idx, self.max_idx, dtype=np.int64)
        if self._segment_clean is not None:
            keep = self._segment_clean[candidate]
            self._valid_indices = candidate[keep]
        else:
            self._valid_indices = candidate
        self.n_samples = len(self._valid_indices)
        if self._segment_clean is not None:
            dropped = (self.max_idx - self.min_idx) - self.n_samples
            print(f"     segment filter: dropped {dropped:,} samples whose "
                  f"lookback/forward window crosses a session break")
            print(f"     final valid samples: {self.n_samples:,}")
        # Hard-fail / warn on degenerate sample counts before the operator
        # commits a GPU to a guaranteed-overfit run.
        _check_sample_size(self.n_samples)

        # Stats fit window: defaults to this Dataset's own range
        self.stats_min_idx = stats_min_idx if stats_min_idx is not None else self.min_idx
        self.stats_max_idx = stats_max_idx if stats_max_idx is not None else self.max_idx
        if self.stats_max_idx <= self.stats_min_idx:
            raise ValueError(
                f"Stats window is empty: [{self.stats_min_idx}..{self.stats_max_idx}]. "
                f"Pass valid stats_min_idx/stats_max_idx for the train slice."
            )

        # Regime codes (causal, computed across full series — OK because
        # assign_regime_label is one-sided rolling)
        if 'regime_label' in self.df.columns:
            self.regime_codes = (
                self.df['regime_label'].astype(str).map(REGIME_TO_CODE)
                .fillna(1).astype(np.int64).to_numpy()
            )
        else:
            self.regime_codes = np.ones(n_total, dtype=np.int64)

        # ── Context feature columns: blacklist-driven, fail-closed ──
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        excluded = [c for c in numeric_cols if _is_leakage_column(c)]
        kept = [c for c in numeric_cols if not _is_leakage_column(c)]
        self.context_cols = kept[:context_dim]
        print(f"     context features: {len(self.context_cols)} (capped at {context_dim}, "
              f"excluded {len(excluded)} leakage-prone)")
        if len(self.context_cols) == 0:
            raise RuntimeError("No causal context features available — check blacklist.")

        # Fit z-score on TRAIN slice only (S1 fix)
        self._compute_context_stats()
        self._precompute_targets()

    def _build_segment_mask(self, n_total: int) -> None:
        """Compute per-bar mask: True iff the bar can be used as a sample
        without its lookback window (T bars back) or its forward target
        window (24 bars ahead) crossing a session break.

        Session breaks are emitted by prepare_day_trading wherever the
        inter-bar timestamp gap exceeds 3× the modal bar duration. This
        captures both weekend gaps AND the inter-quarter gaps when the
        user concatenates multiple contracts with deliberate cuts.

        If the parquet has no `is_session_break` column, no filtering is
        applied (= legacy behavior).
        """
        if 'is_session_break' not in self.df.columns:
            self._segment_clean = None
            return
        breaks = self.df['is_session_break'].astype(bool).to_numpy()
        if not breaks.any():
            self._segment_clean = None
            return

        L = int(self.lookback_bars)
        H = int(self.embargo_bars)
        # cumulative count of breaks up to index i (inclusive)
        break_cs = np.concatenate([[0], np.cumsum(breaks)]).astype(np.int64)
        clean = np.zeros(n_total, dtype=bool)
        for i in range(n_total):
            # Lookback window = [max(0, i-L), i]; forward = [i, i+H]
            lo_lb = max(0, i - L)
            hi_fwd = min(n_total, i + H + 1)
            n_breaks_lb = int(break_cs[i + 1] - break_cs[lo_lb])
            n_breaks_fwd = int(break_cs[hi_fwd] - break_cs[i + 1])
            # A break AT index i itself disqualifies (it IS the break point)
            clean[i] = (n_breaks_lb == 0 and n_breaks_fwd == 0)
        self._segment_clean = clean


    def _compute_context_stats(self):
        """Fit mu/sigma on the train slice [stats_min_idx, stats_max_idx).

        Previously fit on the full DataFrame — this leaked holdout
        distribution statistics into the training normalizer.
        """
        a, b = self.stats_min_idx, self.stats_max_idx
        n_fit = b - a
        ctx_data = np.zeros((n_fit, len(self.context_cols)), dtype=np.float64)
        for j, col in enumerate(self.context_cols):
            vals = pd.to_numeric(self.df[col].iloc[a:b], errors='coerce').fillna(0.0).to_numpy()
            ctx_data[:, j] = np.clip(vals, -1e9, 1e9)
        self.context_mu = ctx_data.mean(axis=0).astype(np.float32)
        self.context_sigma = ctx_data.std(axis=0).astype(np.float32)
        self.context_sigma = np.maximum(self.context_sigma, 1e-6)
        print(f"     stats fit on idx [{a}..{b}] (n={n_fit}): "
              f"mu range=[{self.context_mu.min():.3e}, {self.context_mu.max():.3e}], "
              f"sigma range=[{self.context_sigma.min():.3e}, {self.context_sigma.max():.3e}]")

    def _precompute_targets(self):
        """Precompute SSL targets. All forward-looking targets are shifted
        by +1 (or by the appropriate lookahead) and out-of-bounds bars are
        marked invalid via the corresponding *_valid masks. Bars whose
        targets cross the dataset's right edge will be filtered out of the
        sampler so the model never trains on a truncated/synthetic target.
        """
        n = len(self.df)
        print(f"  ⚙️  Precomputing SSL targets (causal, embargo-aware)...")

        def _safe_col(col_name, fallback=0.0, dtype=np.float32):
            if col_name in self.df.columns:
                return pd.to_numeric(self.df[col_name], errors='coerce').fillna(fallback).to_numpy(dtype=dtype)
            return np.full(n, fallback, dtype=dtype)

        # ── close — CAUSAL handling only: ffill (past→present), no bfill ──
        # bfill would copy a FUTURE close into a past NaN position, causing
        # `next_price[i-1] = log(close[i]/close[i-1])` to encode a real
        # forward return from close[i+k] when close[i] is missing.
        close_raw = pd.to_numeric(self.df['close'], errors='coerce').to_numpy(dtype=np.float64)
        close = np.where(np.isfinite(close_raw) & (close_raw > 0), close_raw, np.nan)
        if np.any(np.isnan(close)):
            close_filled = pd.Series(close).ffill().to_numpy(dtype=np.float64)
            # Leading NaN (before first valid close) — keep NaN; the
            # next_price valid_mask below will drop these bars.
            close = close_filled

        # next_price[i] = log(close[i+1] / close[i]). Last bar invalid.
        log_ret = np.full(n, np.nan, dtype=np.float64)
        valid_pair = np.isfinite(close[:-1]) & np.isfinite(close[1:]) & (close[:-1] > 0) & (close[1:] > 0)
        log_ret[:-1] = np.where(
            valid_pair,
            np.log(np.maximum(close[1:], 1e-9) / np.maximum(close[:-1], 1e-9)),
            np.nan,
        )
        # Clip extreme returns (±10% per bar = ±1000 pip — protect from
        # MBO corruption; real GBPUSD 15m bars never move this much)
        log_ret_clipped = np.where(
            np.isfinite(log_ret), np.clip(log_ret, -0.1, 0.1), 0.0,
        )
        self.next_price = log_ret_clipped.astype(np.float32)
        valid_mask = np.isfinite(log_ret)
        # Session-break aware (B5 propagation): next_price[i] looks at
        # close[i+1]; if bar i+1 is the first bar after a session break,
        # the return is a 65h weekend jump treated as a 15m target → exclude.
        if 'is_session_break' in self.df.columns:
            sb = self.df['is_session_break'].astype(bool).to_numpy()
            valid_mask[:-1] &= ~sb[1:]
            # First bar of dataset (no prev) — also invalid for next_price
        self.next_price_valid = valid_mask

        # next_imbalance — uses obi[i+1]
        # PIN: require the column under one of its exact known names. The
        # previous fallback to 0.0 produced a silently-zero target that
        # quietly killed the next_imbalance task's gradient — we'd rather
        # fail loudly here than train a head on noise.
        if 'obi_net' in self.df.columns:
            obi_col = 'obi_net'
        elif 'order_flow_imbalance' in self.df.columns:
            obi_col = 'order_flow_imbalance'
        else:
            raise RuntimeError(
                "Pipeline misconfigured: the features parquet has neither "
                "`obi_net` nor `order_flow_imbalance`. The SSL "
                "next_imbalance task requires one of them. "
                "Re-run prepare_day_trading.py and verify the OBI "
                "computation is on (see prepare_day_trading.py:1038)."
            )
        obi = _safe_col(obi_col, fallback=0.0)
        obi = np.clip(obi, -1.0, 1.0)
        next_obi = np.zeros(n, dtype=np.float32)
        next_obi[:-1] = obi[1:]
        self.next_imbalance = next_obi
        self.next_imbalance_valid = np.zeros(n, dtype=bool)
        self.next_imbalance_valid[:-1] = True

        # next_volatility — uses atr[i+1]
        # PIN: same loud-failure rule as obi above. An all-zero ATR target
        # would degrade next_volatility training to noise.
        if 'atr_14' in self.df.columns:
            atr_col = 'atr_14'
        elif 'atr' in self.df.columns:
            atr_col = 'atr'
        else:
            raise RuntimeError(
                "Pipeline misconfigured: the features parquet has neither "
                "`atr_14` nor `atr`. The SSL next_volatility task requires "
                "one. Re-run prepare_day_trading.py with ATR enabled."
            )
        atr = _safe_col(atr_col, fallback=0.001)
        atr = np.clip(atr, 1e-5, 0.05).astype(np.float32)
        next_atr = np.zeros(n, dtype=np.float32)
        next_atr[:-1] = atr[1:]

        self.next_volatility = np.maximum(next_atr, 1e-6)
        self.next_volatility_valid = np.zeros(n, dtype=bool)
        self.next_volatility_valid[:-1] = True

        # next_regime
        next_regime = np.ones(n, dtype=np.int64)
        next_regime[:-1] = self.regime_codes[1:]
        self.next_regime = next_regime
        self.next_regime_valid = np.zeros(n, dtype=bool)
        self.next_regime_valid[:-1] = True

        # wall_persist & time_to_event with proper boundary handling
        self.wall_persist, self.wall_persist_valid = _compute_wall_persist_array(self.df)
        self.time_to_event, self.time_to_event_valid = _compute_time_to_event_array(self.df)

        # ── Cycle targets: NEXT bar (not current). The current bar's
        # cycle_phase_*_prob is already excluded from context features —
        # but to fully kill the identity-mapping risk we predict bar i+1.
        phase = np.zeros(n, dtype=np.int64)
        swing = np.zeros(n, dtype=np.int64)
        maturity = np.zeros(n, dtype=np.int64)
        cycle_pos = np.zeros(n, dtype=np.float32)
        cycle_valid = np.zeros(n, dtype=bool)
        for i in range(n - 1):  # last bar invalid (no i+1)
            tgt_idx = i + 1
            phase[i] = _compute_phase_target_at(self.df, tgt_idx)
            swing[i] = _compute_swing_target_at(self.df, tgt_idx)
            maturity[i] = _compute_maturity_target_at(self.df, tgt_idx)
            cycle_pos[i] = float(self.df.iloc[tgt_idx].get('cycle_position', 0.0))
            cycle_valid[i] = True
        self.phase_target = phase
        self.swing_target = swing
        self.maturity_target = maturity
        self.cycle_position_target = np.nan_to_num(cycle_pos, nan=0.0, posinf=1.0, neginf=-1.0)
        self.cycle_valid = cycle_valid

        # Final all-finite check on float targets
        all_finite = (
            np.isfinite(self.next_price).all()
            and np.isfinite(self.next_imbalance).all()
            and np.isfinite(self.next_volatility).all()
            and np.isfinite(self.wall_persist).all()
            and np.isfinite(self.time_to_event).all()
            and np.isfinite(self.cycle_position_target).all()
        )
        if not all_finite:
            raise RuntimeError("SSL targets contain NaN/Inf after computation — bug.")

        n_valid_price = int(self.next_price_valid[self.min_idx:self.max_idx].sum())
        n_valid_wall = int(self.wall_persist_valid[self.min_idx:self.max_idx].sum())
        n_valid_tte = int(self.time_to_event_valid[self.min_idx:self.max_idx].sum())
        n_valid_cyc = int(self.cycle_valid[self.min_idx:self.max_idx].sum())
        print(f"     ✅ Targets ready. Valid in sample range: "
              f"price={n_valid_price}, wall={n_valid_wall}, tte={n_valid_tte}, cycle={n_valid_cyc}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, item: int) -> dict:
        # Use the segment-filtered index list when present; otherwise
        # fall back to the linear (min_idx + item) mapping.
        if self._segment_clean is not None:
            idx = int(self._valid_indices[item])
        else:
            idx = self.min_idx + item

        lob_window = np.asarray(self.lob_tensors[idx], dtype=np.float32)

        # Sanitize lob_window first (mmap files can carry NaN from corrupted ticks)
        lob_window = np.nan_to_num(lob_window, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Order features: real orders (NEW) or pseudo-orders (fallback) ──
        if self.use_real_orders:
            # Real orders from MBO: preserves order IDs, iceberg signals, etc.
            order_features = np.asarray(self.order_features_arr[idx], dtype=np.float32)
            order_masks = np.asarray(self.order_masks_arr[idx], dtype=bool)
            # CRITICAL safety net: NaN/Inf في order_features = val=NaN guaranteed
            # (build_order_batches has guards لكن edge cases ممكن تسرّب)
            order_features = np.nan_to_num(order_features, nan=0.0, posinf=0.0, neginf=0.0)
            # Defense-in-depth: لو time_offset (channel 4) فيه قيم كبيرة جداً
            # (من order_batches قديمة قبل الـ fix)، نعيد تطبيعها هنا
            t_off = order_features[:, :, 4]
            if t_off.max() > 100:  # > 100 يعني raw ms (pre-fix)، normalize
                order_features[:, :, 4] = t_off / max(t_off.max(), 1.0)
        else:
            # Pseudo-orders from LOB tensor (lossy — no order IDs, no iceberg)
            order_features, order_masks = _lob_tensor_to_orders_vectorized(
                lob_window, n_orders=self.n_orders,
            )
            order_features = np.nan_to_num(order_features, nan=0.0, posinf=0.0, neginf=0.0)
        bar_mask = np.ones(lob_window.shape[0], dtype=bool)
        context = _build_context_features(
            self.df.iloc[idx], self.context_cols, target_dim=self.context_dim,
        )
        # Z-score normalization (يمنع gradient explosion من cvd/cumulative values)
        n_ctx_actual = len(self.context_cols)
        if n_ctx_actual > 0:
            context[:n_ctx_actual] = (context[:n_ctx_actual] - self.context_mu) / self.context_sigma
            context[:n_ctx_actual] = np.clip(context[:n_ctx_actual], -5.0, 5.0)  # safety clip
        # Final safety net عشان مفيش NaN يوصل للموديل
        context = np.nan_to_num(context, nan=0.0, posinf=5.0, neginf=-5.0)

        # ── Cycle window: strictly past, EXCLUSIVE of idx ──
        # Previous version included idx itself, and the cycle target was
        # computed from columns AT idx (cycle_phase_*_prob[idx]) — an
        # identity mapping. Now: window is bars [idx-T .. idx-1], target
        # is at idx+1 (set in _precompute_targets). The cycle_phase_*_prob
        # columns are ALSO excluded from the LOB context features (see
        # _LEAKAGE_COLS_EXACT), so the model has no path to cheat.
        cycle_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'cycle_structure_score', 'cycle_trend_maturity',
            'cycle_phase_acc_prob', 'cycle_phase_markup_prob',
            'cycle_phase_dist_prob', 'cycle_phase_markdown_prob',
            'cycle_hurst', 'cycle_fractal_dim', 'cycle_mtf_alignment',
        ]
        avail_cycle = [c for c in cycle_cols if c in self.df.columns]
        T = lob_window.shape[0]
        start = max(0, idx - T)
        end = idx                                   # exclusive of current bar
        cycle_window = self.df[avail_cycle].iloc[start:end].to_numpy(dtype=np.float32)
        if cycle_window.shape[0] < T:
            pad = np.zeros((T - cycle_window.shape[0], cycle_window.shape[1]), dtype=np.float32)
            cycle_window = np.concatenate([pad, cycle_window], axis=0)
        cycle_window = np.nan_to_num(cycle_window, nan=0.0, posinf=0.0, neginf=0.0)

        # LOB image tensor — (T, P, C) shape, used by the new LOB CNN branch
        # in HierarchicalLOBTransformer. Already sanitized at line ~519;
        # we just ensure dtype is float32 (mmap files default to whatever
        # was saved, may be float64).
        lob_image = lob_window.astype(np.float32, copy=False)

        return {
            'order_features': torch.from_numpy(order_features),
            'order_masks': torch.from_numpy(order_masks),
            'bar_mask': torch.from_numpy(bar_mask),
            'context': torch.from_numpy(context),
            'cycle_window': torch.from_numpy(cycle_window),
            'lob_image': torch.from_numpy(lob_image),
            # Targets
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
            # Per-target validity flags (1.0 = use this sample's loss for the
            # corresponding head; 0.0 = mask out — target is truncated /
            # missing). Training loop should weight per-task losses by these.
            'next_price_valid': torch.tensor(float(self.next_price_valid[idx]), dtype=torch.float32),
            'next_imbalance_valid': torch.tensor(float(self.next_imbalance_valid[idx]), dtype=torch.float32),
            'next_volatility_valid': torch.tensor(float(self.next_volatility_valid[idx]), dtype=torch.float32),
            'next_regime_valid': torch.tensor(float(self.next_regime_valid[idx]), dtype=torch.float32),
            'wall_persist_valid': torch.tensor(float(self.wall_persist_valid[idx]), dtype=torch.float32),
            'time_to_event_valid': torch.tensor(float(self.time_to_event_valid[idx]), dtype=torch.float32),
            'cycle_valid': torch.tensor(float(self.cycle_valid[idx]), dtype=torch.float32),
        }


def build_ssl_loaders(
    features_parquet: str,
    lob_tensors_path: str,
    lob_timestamps_path: Optional[str] = None,
    *,
    order_batches_dir: Optional[str] = None,
    batch_size: int = 32,
    train_split: float = 0.75,
    num_workers: int = 0,
    lookback_bars: int = 50,
    embargo_bars: int = 24,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build train + holdout SSL data loaders with a purged-embargo time split.

    Three protections against train/holdout leakage:

    1. **Time-based split** (no random shuffle across the boundary)
       Train idx ∈ [lookback_bars, split_idx - embargo_bars);
       Holdout idx ∈ [split_idx + lookback_bars, n - embargo_bars).

    2. **Lookback embargo** — holdout's first valid sample is shifted right
       by `lookback_bars` so its input window doesn't reach back into the
       training period. Without this, the first 50 holdout samples would
       have 49/50 input bars belonging to the training set.

    3. **Target-horizon embargo** — train's last valid sample is shifted
       left by `embargo_bars` so its forward-looking targets
       (next_price, wall_persist@12, time_to_event@24) don't reach into
       the holdout period.

    Both datasets fit z-score stats on the TRAIN slice only (passed
    explicitly via stats_min_idx/stats_max_idx).
    """
    df_for_n = pd.read_parquet(features_parquet, columns=['close'])
    n = len(df_for_n)
    del df_for_n

    split_idx = int(n * train_split)

    train_lo = lookback_bars
    train_hi = max(train_lo + 1, split_idx - embargo_bars)
    holdout_lo = split_idx + lookback_bars
    holdout_hi = max(holdout_lo + 1, n - embargo_bars)

    if train_hi <= train_lo:
        raise ValueError(f"Train slice empty after embargo: [{train_lo}..{train_hi}]")
    if holdout_hi <= holdout_lo:
        raise ValueError(f"Holdout slice empty after embargo: [{holdout_lo}..{holdout_hi}]")

    print(f"⚙️  Building SSL data loaders (purged-embargo split)")
    print(f"   Total bars:  {n}")
    print(f"   Train:       idx [{train_lo}..{train_hi}) — {train_hi - train_lo:,} samples")
    print(f"   ── gap (lookback+embargo) ──")
    print(f"   Holdout:     idx [{holdout_lo}..{holdout_hi}) — {holdout_hi - holdout_lo:,} samples")
    print(f"   Order mode:  {'REAL (from MBO)' if order_batches_dir else 'pseudo (from LOB tensor)'}")

    train_ds = SSLDataset(
        features_parquet, lob_tensors_path, lob_timestamps_path,
        order_batches_dir=order_batches_dir,
        lookback_bars=lookback_bars,
        min_idx=train_lo,
        max_idx=train_hi,
        stats_min_idx=train_lo,
        stats_max_idx=train_hi,
        embargo_bars=embargo_bars,
    )
    holdout_ds = SSLDataset(
        features_parquet, lob_tensors_path, lob_timestamps_path,
        order_batches_dir=order_batches_dir,
        lookback_bars=lookback_bars,
        min_idx=holdout_lo,
        max_idx=holdout_hi,
        # CRITICAL: holdout uses TRAIN stats (else leak)
        stats_min_idx=train_lo,
        stats_max_idx=train_hi,
        embargo_bars=embargo_bars,
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
