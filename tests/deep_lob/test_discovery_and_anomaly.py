"""Tests for pattern discovery + anomaly detection."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestAttentionDiscovery(unittest.TestCase):
    def test_critical_orders_basic(self):
        from modules.deep_lob.pattern_discovery import discover_critical_orders_from_attention
        # Mock attention: 2 layers, batch=2, N=10
        attns = [torch.softmax(torch.randn(2, 10, 10), dim=-1) for _ in range(2)]
        features = torch.randn(2, 10, 7)
        masks = torch.ones(2, 10, dtype=torch.bool)
        results = discover_critical_orders_from_attention(attns, features, masks, top_k=3)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]["critical_indices"]), 3)
        self.assertEqual(len(results[0]["attention_scores"]), 3)

    def test_event_assignments(self):
        from modules.deep_lob.pattern_discovery import discover_event_assignments
        # (B=2, E=4, N=10)
        assignment = torch.softmax(torch.randn(2, 4, 10), dim=-1)
        features = torch.randn(2, 10, 7)
        masks = torch.ones(2, 10, dtype=torch.bool)
        results = discover_event_assignments(assignment, features, masks, top_orders_per_event=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]["event_assignments"]), 4)


class TestClusterDiscovery(unittest.TestCase):
    def test_kmeans_clusters_detected(self):
        from modules.deep_lob.pattern_discovery import discover_clusters_from_embeddings
        rng = np.random.RandomState(0)
        # 2 well-separated clusters with different outcomes
        c1 = rng.randn(50, 8) + np.array([0]*8)
        c2 = rng.randn(50, 8) + np.array([5]*8)
        embeddings = np.vstack([c1, c2])
        outcomes = np.concatenate([rng.uniform(-1, 0, 50), rng.uniform(0, 1, 50)])

        patterns = discover_clusters_from_embeddings(
            embeddings, outcomes, min_cluster_size=10, method="kmeans", n_clusters=2,
        )
        self.assertGreaterEqual(len(patterns), 1)
        for p in patterns:
            self.assertGreater(p.n_examples, 0)
            self.assertIn("win_rate", p.metadata)
            self.assertIn("sharpe", p.metadata)


class TestMahalanobisDetector(unittest.TestCase):
    def test_fit_and_detect_normal(self):
        from modules.deep_lob.anomaly_detection import MahalanobisAnomalyDetector
        rng = np.random.RandomState(0)
        train = rng.randn(200, 8)
        det = MahalanobisAnomalyDetector(threshold_sigma=3.0)
        det.fit(train)
        # Normal point shouldn't be anomaly
        result = det.detect(rng.randn(8))
        self.assertFalse(result.is_anomaly)

    def test_detect_anomaly(self):
        from modules.deep_lob.anomaly_detection import MahalanobisAnomalyDetector
        rng = np.random.RandomState(0)
        train = rng.randn(200, 8)
        det = MahalanobisAnomalyDetector(threshold_sigma=3.0)
        det.fit(train)
        # Far-away point should be anomaly
        anomalous = np.full(8, 100.0)
        result = det.detect(anomalous)
        self.assertTrue(result.is_anomaly)
        self.assertGreater(result.score, result.threshold)

    def test_unfitted_raises(self):
        from modules.deep_lob.anomaly_detection import MahalanobisAnomalyDetector
        det = MahalanobisAnomalyDetector()
        with self.assertRaises(RuntimeError):
            det.detect(np.zeros(8))


class TestKNNDetector(unittest.TestCase):
    def test_fit_and_detect(self):
        from modules.deep_lob.anomaly_detection import KNearestAnomalyDetector
        rng = np.random.RandomState(0)
        train = rng.randn(100, 8)
        det = KNearestAnomalyDetector(k=5, threshold_quantile=0.99)
        det.fit(train)
        normal = rng.randn(8)
        anomalous = np.full(8, 50.0)
        self.assertFalse(det.detect(normal).is_anomaly)
        self.assertTrue(det.detect(anomalous).is_anomaly)


class TestDiscrepancyDetector(unittest.TestCase):
    def test_fit_and_detect(self):
        from modules.deep_lob.anomaly_detection import MultiTaskDiscrepancyDetector
        rng = np.random.RandomState(0)
        # 3 tasks، normal errors small
        training_errors = {
            "direction": rng.uniform(0.0, 0.5, 200),
            "price": rng.uniform(0.0, 0.3, 200),
            "volatility": rng.uniform(0.0, 0.2, 200),
        }
        det = MultiTaskDiscrepancyDetector(
            n_tasks_threshold=2, per_task_sigma_threshold=2.0,
        )
        det.fit(training_errors)

        # Normal sample
        normal_errors = {"direction": 0.3, "price": 0.2, "volatility": 0.1}
        self.assertFalse(det.detect(normal_errors).is_anomaly)

        # Anomalous: high errors on 3 tasks
        anomalous_errors = {"direction": 5.0, "price": 5.0, "volatility": 5.0}
        result = det.detect(anomalous_errors)
        self.assertTrue(result.is_anomaly)
        self.assertGreaterEqual(len(result.diagnostics["anomalous_tasks"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
