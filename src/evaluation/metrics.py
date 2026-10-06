"""
Evaluation metrics and cross-validation utilities.

Provides:
    - Standard classification metrics (accuracy, precision, recall, F1, AUC)
    - Threshold tuning for F1 optimization
    - Stratified k-fold cross-validation

Authors: Md. Murad Hossain, MD. Foysal Ahammed Joy, Nuruzzaman Shuvo
Supervisor: Md. Morshed Ali
Department: Computer Science and Engineering, Uttara University
"""

from typing import Dict

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities for the positive class

    Returns:
        Dictionary with accuracy, precision, recall, f1, and auc
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
    }


def find_optimal_threshold(y_true: np.ndarray,
                            y_prob: np.ndarray) -> float:
    """
    Find the classification threshold that maximizes F1 score.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities

    Returns:
        Optimal threshold value
    """
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.15, 0.85, 0.01):
        f1 = f1_score(y_true, (y_prob >= thr).astype(int),
                      zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr


def cross_validate(model, X: np.ndarray, y: np.ndarray,
                   n_splits: int = 5,
                   seed: int = 42) -> np.ndarray:
    """
    Perform stratified k-fold cross-validation.

    Args:
        model: Classifier with fit and predict_proba methods
        X: Feature matrix
        y: Labels
        n_splits: Number of folds
        seed: Random seed

    Returns:
        Array of out-of-fold predicted probabilities
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_prob = np.zeros(len(y))

    for tr_idx, te_idx in cv.split(X, y):
        model.fit(X[tr_idx], y[tr_idx])
        y_prob[te_idx] = model.predict_proba(X[te_idx])[:, 1]

    return y_prob
