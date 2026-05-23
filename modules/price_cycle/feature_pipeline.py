"""
modules/price_cycle/feature_pipeline.py
─────────────────────────────────────────
End-to-end feature pipeline: DataFrame → model-ready tensors.

Combines:
    - BarSequence derived features (16-dim/bar)
    - Swing analysis (structural features)
    - Wyckoff phase features
    - Fractal features

Produces the (T, n_features) matrix the CycleEncoder consumes,
plus auto-generated training labels (for self-supervised / weak supervision).

All causal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import PriceCycleConfig
from .data_structures import BarSequence, WyckoffPhase, TrendMaturity
from .swing_analysis import (
    detect_swings, classify_swings, estimate_trend_maturity, swing_structure_score,
)
from .phase_classifier import classify_phase_sequence
from .fractal_features import compute_fractal_features


@dataclass
class CycleFeatureSet:
    """Output of the feature pipeline."""

    bar_features: np.ndarray         # (T, 16) — derived OHLCV features
    structural_features: np.ndarray  # (T, n_struct) — swing/phase/fractal
    weak_labels: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_bars(self) -> int:
        return self.bar_features.shape[0]

    def combined(self) -> np.ndarray:
        """Concatenate bar + structural features → (T, 16 + n_struct)."""
        return np.concatenate([self.bar_features, self.structural_features], axis=1)


def build_cycle_features(
    bars: BarSequence,
    config: Optional[PriceCycleConfig] = None,
    generate_weak_labels: bool = True,
) -> CycleFeatureSet:
    """Full feature pipeline.

    Parameters
    ----------
    bars : BarSequence
    config : PriceCycleConfig
    generate_weak_labels : if True, auto-generate training labels from
        rules-based analysis (Wyckoff phase, swing direction, etc.)
        These serve as weak supervision — the deep model refines them.

    Returns
    -------
    CycleFeatureSet
    """
    if config is None:
        config = PriceCycleConfig()

    T = bars.n_bars

    # ── 1. Bar-level derived features (16-dim) ──────────────────────────
    bar_feats = bars.compute_derived_features(atr_window=config.swing.atr_window)

    # ── 2. Swing analysis ───────────────────────────────────────────────
    swings, confirmations = detect_swings(
        bars.high, bars.low, bars.close, config.swing,
    )
    classified = classify_swings(swings)

    # Per-bar structural features:
    #   [0] structure_score (causal — uses only swings confirmed by bar t)
    #   [1] bars_since_last_swing
    #   [2] trend_maturity (normalized 0/0.5/1)
    #   [3] momentum_decay
    structural = np.zeros((T, 4), dtype=np.float32)
    for t in range(T):
        # Only swings confirmed at or before bar t (causal!)
        confirmed = [
            classified[i] for i in range(len(classified))
            if i < len(confirmations) and confirmations[i] <= t
        ]
        if confirmed:
            structural[t, 0] = swing_structure_score(
                confirmed, config.swing.classification_lookback,
            )
            structural[t, 1] = min((t - confirmed[-1].bar_index) / 100.0, 1.0)
            maturity = estimate_trend_maturity(bars.close[: t + 1], confirmed)
            structural[t, 2] = float(maturity.maturity) / 2.0
            structural[t, 3] = maturity.momentum_decay

    # ── 3. Wyckoff phase features ───────────────────────────────────────
    phase_assessments = classify_phase_sequence(
        bars.high, bars.low, bars.close, bars.volume, config.phase,
    )
    phase_feats = np.zeros((T, 5), dtype=np.float32)
    for t in range(T):
        pa = phase_assessments[t]
        phase_feats[t, :4] = pa.phase_probs
        phase_feats[t, 4] = pa.cycle_position

    # ── 4. Fractal features ─────────────────────────────────────────────
    fractal = compute_fractal_features(
        bars.open, bars.high, bars.low, bars.close, bars.volume, config.fractal,
    )
    fractal_feats = np.stack([
        fractal["hurst"], fractal["fractal_dim"], fractal["mtf_alignment"],
    ], axis=1).astype(np.float32)

    # Combine structural features
    structural_all = np.concatenate([structural, phase_feats, fractal_feats], axis=1)
    structural_all = np.nan_to_num(structural_all)

    # ── 5. Weak labels (rules-based supervision) ────────────────────────
    weak_labels: dict[str, np.ndarray] = {}
    if generate_weak_labels:
        weak_labels["phase"] = np.array(
            [int(pa.phase) for pa in phase_assessments], dtype=np.int64,
        )
        weak_labels["cycle_position"] = np.array(
            [pa.cycle_position for pa in phase_assessments], dtype=np.float32,
        )
        # swing direction per bar
        swing_dir = np.full(T, 2, dtype=np.int64)  # default neutral
        for t in range(T):
            if structural[t, 0] > 0.3:
                swing_dir[t] = 0  # up
            elif structural[t, 0] < -0.3:
                swing_dir[t] = 1  # down
        weak_labels["swing_direction"] = swing_dir
        # maturity
        weak_labels["maturity"] = np.clip(
            (structural[:, 2] * 2.0).round().astype(np.int64), 0, 2,
        )

    return CycleFeatureSet(
        bar_features=bar_feats,
        structural_features=structural_all,
        weak_labels=weak_labels,
        diagnostics={
            "n_bars": T,
            "n_swings": len(swings),
            "n_bar_features": bar_feats.shape[1],
            "n_structural_features": structural_all.shape[1],
            "config": config.model_name,
        },
    )


def make_training_windows(
    feature_set: CycleFeatureSet,
    window_size: int = 128,
    stride: int = 1,
    use_combined: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Slice features into overlapping windows for training.

    Parameters
    ----------
    feature_set : CycleFeatureSet
    window_size : bars per window
    stride : step between windows
    use_combined : if True, use bar+structural features; else bar only

    Returns
    -------
    windows : (n_windows, window_size, n_features)
    labels : dict {task: (n_windows,)} — label = value at the LAST bar of window
    """
    feats = feature_set.combined() if use_combined else feature_set.bar_features
    T, F = feats.shape

    if T < window_size:
        # pad at front
        pad = np.zeros((window_size - T, F), dtype=feats.dtype)
        feats = np.vstack([pad, feats])
        T = window_size

    windows = []
    label_indices = []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        windows.append(feats[start:end])
        label_indices.append(end - 1)  # label = last bar

    windows_arr = np.stack(windows).astype(np.float32)
    label_idx = np.array(label_indices)

    labels: dict[str, np.ndarray] = {}
    for task, values in feature_set.weak_labels.items():
        # account for any front-padding
        offset = len(values) - (T - (0))
        adj_idx = label_idx + offset
        adj_idx = np.clip(adj_idx, 0, len(values) - 1)
        labels[task] = values[adj_idx]

    return windows_arr, labels
