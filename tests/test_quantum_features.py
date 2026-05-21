"""
tests/test_quantum_features.py — Sprint 2 quantum_features tests.

يحرس GHZ states + tensor network + quantum walk features.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _make_synthetic_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "wall_persist_score": rng.uniform(0, 1, n),
        "sim_iceberg_strength": rng.uniform(0, 1, n),
        "sim_informed_prob": rng.uniform(0, 1, n),
        "sim_depth_pressure": rng.uniform(-1, 1, n),
        "volume_zscore": rng.randn(n),
        "atr_14": rng.uniform(0.001, 0.01, n),
        "zone_id_numeric": rng.randint(0, 16, n).astype(float),
        "level_distance": rng.uniform(0, 50, n),
        "kalman_trend": rng.randn(n).cumsum(),
        "sim_absorption": rng.uniform(0, 1, n),
        "sim_orderflow": rng.uniform(-1, 1, n),
        "sim_sweep": rng.uniform(0, 1, n),
        "sim_buy_pressure": rng.uniform(0, 1, n),
        "sim_buy_inertia": rng.uniform(0, 1, n),
        "sim_buy_continuation": rng.uniform(0, 1, n),
    })


class TestGHZStates(unittest.TestCase):
    def test_compute_ghz_basic(self):
        from modules.quantum_features import compute_ghz_features
        df = _make_synthetic_df(200)
        feats = compute_ghz_features(df)
        # 5 standard triplets should all match
        self.assertGreaterEqual(len(feats), 5)
        for k, v in feats.items():
            self.assertTrue(k.startswith("ghz_"))
            self.assertEqual(len(v), 200)
            self.assertTrue(np.all((v >= 0) & (v <= 1)),
                f"{k} should be in [0,1], got range [{v.min()}, {v.max()}]")

    def test_compute_ghz_skip_missing(self):
        from modules.quantum_features import compute_ghz_features
        df = pd.DataFrame({"only_one_col": [1.0, 2.0]})
        feats = compute_ghz_features(df, skip_missing=True)
        self.assertEqual(len(feats), 0)

    def test_compute_ghz_raise_missing(self):
        from modules.quantum_features import compute_ghz_features
        df = pd.DataFrame({"only_one_col": [1.0, 2.0]})
        with self.assertRaises(KeyError):
            compute_ghz_features(df, skip_missing=False)

    def test_add_ghz_to_df(self):
        from modules.quantum_features import add_ghz_features
        df = _make_synthetic_df(50)
        out = add_ghz_features(df)
        new_cols = [c for c in out.columns if c.startswith("ghz_")]
        self.assertGreaterEqual(len(new_cols), 5)
        # original df unchanged
        self.assertNotIn("ghz_wall_iceberg_informed", df.columns)


class TestTensorNetwork(unittest.TestCase):
    def test_outer_product_shape(self):
        from modules.quantum_features import build_outer_product_tensor
        a = np.random.RandomState(0).randn(100, 3)
        b = np.random.RandomState(1).randn(100, 4)
        c = np.random.RandomState(2).randn(100, 5)
        tensor = build_outer_product_tensor([a, b, c])
        self.assertEqual(tensor.shape, (100, 3, 4, 5))

    def test_build_feature_tensor_from_df(self):
        from modules.quantum_features import build_feature_tensor
        df = _make_synthetic_df(50)
        groups = [
            ["wall_persist_score", "sim_iceberg_strength"],
            ["sim_depth_pressure", "volume_zscore"],
        ]
        tensor = build_feature_tensor(df, groups)
        # shape (50, 2, 2)
        self.assertEqual(tensor.shape, (50, 2, 2))

    def test_flatten_tensor(self):
        from modules.quantum_features import build_feature_tensor, flatten_tensor
        df = _make_synthetic_df(30)
        tensor = build_feature_tensor(df, [
            ["wall_persist_score"],
            ["sim_iceberg_strength", "sim_informed_prob"],
        ])
        flat = flatten_tensor(tensor)
        self.assertEqual(flat.shape, (30, 1 * 2))

    def test_mps_truncate_reduces_dim(self):
        from modules.quantum_features import mps_truncate
        rng = np.random.RandomState(0)
        # full tensor (50, 4, 4) — 16 dim flat
        arr = rng.randn(50, 4, 4)
        out = mps_truncate(arr, bond_dim=3)
        self.assertEqual(out.shape, (50, 3))

    def test_tensor_correlation(self):
        from modules.quantum_features import build_outer_product_tensor, tensor_correlation
        rng = np.random.RandomState(7)
        a = rng.randn(200, 1)
        b = rng.randn(200, 1)
        tensor = build_outer_product_tensor([a, b])
        # target == a[:, 0] * b[:, 0] (perfect correlation)
        target = (a[:, 0] * b[:, 0])
        cors = tensor_correlation(tensor, target)
        # one entry should be ≈ 1
        self.assertGreater(float(np.max(np.abs(cors))), 0.9)


class TestQuantumWalk(unittest.TestCase):
    def test_adjacency_shape(self):
        from modules.quantum_features import build_topology_adjacency
        df = _make_synthetic_df(50)
        adj = build_topology_adjacency(df, n_neighbors=3)
        self.assertEqual(adj.shape, (50, 50))
        # symmetric
        self.assertTrue(np.allclose(adj, adj.T))

    def test_walk_step_preserves_norm(self):
        from modules.quantum_features import build_topology_adjacency, quantum_walk_step
        df = _make_synthetic_df(30)
        adj = build_topology_adjacency(df, n_neighbors=3)
        state = np.full(30, 1.0 / np.sqrt(30))
        new_state = quantum_walk_step(adj, state)
        self.assertAlmostEqual(float(np.sum(new_state ** 2)), 1.0, places=4)

    def test_walk_features_smoke(self):
        from modules.quantum_features import compute_quantum_walk_features
        df = _make_synthetic_df(30)
        feats = compute_quantum_walk_features(df, n_steps=2, n_neighbors=3)
        self.assertEqual(len(feats), 2)
        self.assertIn("qwalk_step_1", feats)
        self.assertIn("qwalk_step_2", feats)
        for k, v in feats.items():
            self.assertEqual(len(v), 30)

    def test_walk_features_max_nodes_sampling(self):
        """large df → subsampled أعمالاً"""
        from modules.quantum_features import compute_quantum_walk_features
        df = _make_synthetic_df(2000)
        feats = compute_quantum_walk_features(df, n_steps=1, max_nodes=200)
        self.assertEqual(len(feats), 1)


class TestQuantumIntegration(unittest.TestCase):
    """End-to-end: prepare_state + grover + measure."""

    def test_full_pipeline(self):
        from modules.quantum_core import (
            grover_iteration, measure_greedy, confidence_from_state,
        )
        rng = np.random.RandomState(42)
        # 64 candidates, top-3 marked good
        N = 64
        scores = rng.uniform(0, 1, N)
        scores[[10, 30, 50]] = 10.0

        # State preparation: uniform superposition (required for Grover)
        state = np.full(N, 1.0 / np.sqrt(N))

        # Oracle: mark scores > 5
        mask = scores > 5.0
        amplified = grover_iteration(state, mask)

        # Measure → should pick one of the marked (10, 30, 50)
        idx = measure_greedy(amplified)
        self.assertIn(int(idx), [10, 30, 50])

        # Marked indices should dominate the probability mass
        p = amplified ** 2
        marked_mass = float(np.sum(p[[10, 30, 50]]))
        self.assertGreater(marked_mass, 0.5,
            f"marked indices should hold >50% probability, got {marked_mass:.3f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
