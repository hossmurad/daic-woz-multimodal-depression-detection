"""
Evaluation metrics and cross-validation utilities.

Modules:
    metrics: Standard classification metrics, threshold tuning,
             and stratified cross-validation.
"""

from .metrics import (
    compute_metrics,
    find_optimal_threshold,
    cross_validate,
)

__all__ = ["compute_metrics", "find_optimal_threshold", "cross_validate"]
