"""
Model definitions for DAIC-WOZ depression detection.

Modules:
    late_fusion_model: Late Fusion classifier with per-modality PCA
                       and SMOTE oversampling.
"""

from .late_fusion_model import LateFusionClassifier, FusionConfig

__all__ = ["LateFusionClassifier", "FusionConfig"]
