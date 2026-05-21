"""
modules/deep_lob/anomaly_detection.py
──────────────────────────────────────
Detect novel market states using embedding-based anomaly detection.

3 methods:
    1. Mahalanobis distance from training distribution
    2. K-nearest neighbor distance
    3. Multi-task prediction-actual discrepancy

When the live market produces embeddings unlike anything in training,
this is an early warning of regime shift / unprecedented state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class AnomalyResult:
    """Result of anomaly check."""

    is_anomaly: bool
    score: float                      # higher = more anomalous
    threshold: float
    method: str
    closest_known_distance: float
    closest_known_index: int = -1
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════
# Mahalanobis-based detection
# ════════════════════════════════════════════════════════════════════════════


class MahalanobisAnomalyDetector:
    """Detects anomalies via Mahalanobis distance from training embedding distribution.

    D_M(x) = √((x - μ)^T Σ^(-1) (x - μ))

    Where μ, Σ are estimated from training embeddings.
    """

    def __init__(self, regularization: float = 1e-6, threshold_sigma: float = 3.0):
        self.regularization = regularization
        self.threshold_sigma = threshold_sigma
        self.mean: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.fitted = False
        self.training_distances: Optional[np.ndarray] = None
        self.threshold: float = 0.0

    def fit(self, training_embeddings: np.ndarray) -> "MahalanobisAnomalyDetector":
        """Fit on training embeddings.

        Parameters
        ----------
        training_embeddings : (N, D)
        """
        X = np.asarray(training_embeddings, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D, got shape {X.shape}")

        self.mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)

        # Regularization (handle singular cov)
        cov_reg = cov + self.regularization * np.eye(X.shape[1])
        self.cov_inv = np.linalg.inv(cov_reg)

        # Compute training distances → threshold
        self.training_distances = self._distance_batch(X)
        self.threshold = float(
            self.training_distances.mean() + self.threshold_sigma * self.training_distances.std()
        )
        self.fitted = True
        return self

    def _distance_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch Mahalanobis distance computation."""
        diff = X - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self.cov_inv, diff))

    def detect(self, embedding: np.ndarray) -> AnomalyResult:
        """Detect anomaly for single embedding.

        Parameters
        ----------
        embedding : (D,)
        """
        if not self.fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        x = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        dist = float(self._distance_batch(x)[0])

        return AnomalyResult(
            is_anomaly=dist > self.threshold,
            score=dist,
            threshold=self.threshold,
            method="mahalanobis",
            closest_known_distance=float(self.training_distances.min()),
            diagnostics={
                "training_distance_mean": float(self.training_distances.mean()),
                "training_distance_std": float(self.training_distances.std()),
                "training_distance_max": float(self.training_distances.max()),
                "sigma": (dist - self.training_distances.mean()) / max(self.training_distances.std(), 1e-9),
            },
        )

    def detect_batch(self, embeddings: np.ndarray) -> list[AnomalyResult]:
        """Detect for batch of embeddings."""
        X = np.asarray(embeddings, dtype=np.float64)
        distances = self._distance_batch(X)
        return [
            AnomalyResult(
                is_anomaly=bool(d > self.threshold),
                score=float(d),
                threshold=self.threshold,
                method="mahalanobis",
                closest_known_distance=float(self.training_distances.min()),
            )
            for d in distances
        ]


# ════════════════════════════════════════════════════════════════════════════
# KNN-based detection
# ════════════════════════════════════════════════════════════════════════════


class KNearestAnomalyDetector:
    """Detects anomalies via distance to K nearest training embeddings."""

    def __init__(self, k: int = 5, threshold_quantile: float = 0.99):
        self.k = k
        self.threshold_quantile = threshold_quantile
        self.training_embeddings: Optional[np.ndarray] = None
        self.threshold: float = 0.0
        self.fitted = False

    def fit(self, training_embeddings: np.ndarray) -> "KNearestAnomalyDetector":
        X = np.asarray(training_embeddings, dtype=np.float64)
        self.training_embeddings = X

        # Compute K-nearest distances for each training point (cross-validation style)
        training_distances = self._knn_distance_batch(X, exclude_self=True)
        self.threshold = float(np.quantile(training_distances, self.threshold_quantile))
        self.fitted = True
        return self

    def _knn_distance_batch(self, X: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        """For each x, return distance to its k-th nearest neighbor in training set."""
        # Use pairwise distance (suitable for small to medium N)
        # For large N, use kd-tree or approximate NN
        n_query = X.shape[0]
        n_train = self.training_embeddings.shape[0]
        diff = X[:, None, :] - self.training_embeddings[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))

        if exclude_self:
            # set self-distance to inf
            for i in range(min(n_query, n_train)):
                if i < n_query and i < n_train:
                    dists[i, i] = np.inf

        # Sort and take k-th
        sorted_dists = np.sort(dists, axis=1)
        k = min(self.k, sorted_dists.shape[1] - 1) if exclude_self else min(self.k, sorted_dists.shape[1])
        return sorted_dists[:, k - 1]

    def detect(self, embedding: np.ndarray) -> AnomalyResult:
        if not self.fitted:
            raise RuntimeError("Detector not fitted")

        x = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        knn_dist = float(self._knn_distance_batch(x)[0])

        # Find closest single neighbor
        diff = self.training_embeddings - x
        all_dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        closest_idx = int(np.argmin(all_dists))
        closest_dist = float(all_dists[closest_idx])

        return AnomalyResult(
            is_anomaly=knn_dist > self.threshold,
            score=knn_dist,
            threshold=self.threshold,
            method=f"k{self.k}_nearest",
            closest_known_distance=closest_dist,
            closest_known_index=closest_idx,
            diagnostics={"k": self.k},
        )


# ════════════════════════════════════════════════════════════════════════════
# Multi-task discrepancy
# ════════════════════════════════════════════════════════════════════════════


class MultiTaskDiscrepancyDetector:
    """Detects regime shift via prediction-actual discrepancy across multiple tasks.

    The intuition: if the model is suddenly wrong on multiple tasks simultaneously,
    the market has likely shifted (model's understanding no longer matches reality).
    """

    def __init__(
        self,
        n_tasks_threshold: int = 3,
        per_task_sigma_threshold: float = 2.0,
    ):
        self.n_tasks_threshold = n_tasks_threshold
        self.per_task_sigma = per_task_sigma_threshold
        self.task_baselines: dict[str, dict] = {}
        self.fitted = False

    def fit(self, training_errors: dict[str, np.ndarray]) -> "MultiTaskDiscrepancyDetector":
        """Fit per-task error baselines.

        Parameters
        ----------
        training_errors : dict {task_name: array of per-sample errors}
        """
        for task, errors in training_errors.items():
            arr = np.asarray(errors, dtype=np.float64)
            self.task_baselines[task] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "p99": float(np.quantile(arr, 0.99)),
            }
        self.fitted = True
        return self

    def detect(self, current_errors: dict[str, float]) -> AnomalyResult:
        """Detect for single sample.

        Parameters
        ----------
        current_errors : dict {task: error_value}
        """
        if not self.fitted:
            raise RuntimeError("Detector not fitted")

        sigmas = {}
        anomalous_tasks = []
        for task, err in current_errors.items():
            if task not in self.task_baselines:
                continue
            baseline = self.task_baselines[task]
            std = max(baseline["std"], 1e-9)
            sigma = abs(err - baseline["mean"]) / std
            sigmas[task] = sigma
            if sigma > self.per_task_sigma:
                anomalous_tasks.append(task)

        is_anomaly = len(anomalous_tasks) >= self.n_tasks_threshold
        score = float(np.mean(list(sigmas.values()))) if sigmas else 0.0

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            threshold=float(self.n_tasks_threshold),
            method="multi_task_discrepancy",
            closest_known_distance=0.0,
            diagnostics={
                "anomalous_tasks": anomalous_tasks,
                "n_anomalous": len(anomalous_tasks),
                "per_task_sigmas": sigmas,
            },
        )
