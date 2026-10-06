"""Unit tests for evaluation metrics."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def test_metrics_import():
    from src.evaluation.metrics import (
        compute_metrics, find_optimal_threshold, cross_validate,
    )
    assert compute_metrics is not None
    assert find_optimal_threshold is not None
    assert cross_validate is not None


def test_compute_metrics_perfect():
    from src.evaluation.metrics import compute_metrics
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auc"] == 1.0


def test_find_optimal_threshold():
    from src.evaluation.metrics import find_optimal_threshold
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    threshold = find_optimal_threshold(y_true, y_prob)
    assert 0.3 < threshold < 0.6
