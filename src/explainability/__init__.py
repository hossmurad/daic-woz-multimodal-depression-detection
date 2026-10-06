"""
SHAP-based explainability modules.

Modules:
    shap_analysis: Global feature importance, modality contribution,
                   and per-participant explanations.
"""

from .shap_analysis import SHAPExplainer

__all__ = ["SHAPExplainer"]
