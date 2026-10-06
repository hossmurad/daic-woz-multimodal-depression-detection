"""
SHAP-based explainability for the Late Fusion model.

Provides:
    - Global feature importance via SHAP summary plots
    - Per-modality contribution analysis
    - Patient-level explanations on the fused representation

Authors: Md. Murad Hossain, MD. Foysal Ahammed Joy, Nuruzzaman Shuvo
Supervisor: Md. Morshed Ali
Department: Computer Science and Engineering, Uttara University
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import shap


class SHAPExplainer:
    """SHAP explainability for the Late Fusion classifier."""

    def __init__(self, model, figure_dir: str = "results/figures"):
        """
        Initialize the SHAP explainer.

        Args:
            model: Trained LateFusionClassifier instance
            figure_dir: Directory to save generated figures
        """
        self.model = model
        self.figure_dir = Path(figure_dir)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = model.get_classifier()
        self.explainer = shap.TreeExplainer(self.classifier)

    def compute_shap_values(self, X: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SHAP values on the fused representation.

        Args:
            X: Original 872-dim feature matrix

        Returns:
            Tuple of (SHAP values, fused representation)
        """
        X_fused = self.model.get_fused_representation(X)
        shap_values = self.explainer.shap_values(X_fused)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        return shap_values, X_fused

    def plot_global_importance(self,
                               shap_values: np.ndarray,
                               X_fused: np.ndarray,
                               feature_names: list,
                               top_k: int = 15) -> str:
        """
        Generate SHAP summary plot showing top-k most important features.

        Args:
            shap_values: SHAP values array
            X_fused: Fused feature matrix
            feature_names: Names of fused features
            top_k: Number of top features to display

        Returns:
            Path to saved figure
        """
        path = self.figure_dir / "shap_summary.png"

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_fused,
                          feature_names=feature_names,
                          max_display=top_k, show=False)
        plt.title(f"SHAP Feature Importance (Top {top_k})", fontsize=13)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        return str(path)

    def compute_modality_contribution(self,
                                       shap_values: np.ndarray,
                                       text_pca: int = 30,
                                       audio_pca: int = 15,
                                       visual_pca: int = 10
                                       ) -> Dict[str, float]:
        """
        Compute per-modality contribution as percentage of total SHAP.

        Args:
            shap_values: SHAP values on the fused representation
            text_pca: Number of text PCA components
            audio_pca: Number of audio PCA components
            visual_pca: Number of visual PCA components

        Returns:
            Dictionary mapping modality name to percentage contribution
        """
        text_slice = slice(0, text_pca)
        audio_slice = slice(text_pca, text_pca + audio_pca)
        visual_slice = slice(text_pca + audio_pca,
                             text_pca + audio_pca + visual_pca)

        contributions = {
            "Text": float(np.abs(shap_values[:, text_slice]).mean()),
            "Audio": float(np.abs(shap_values[:, audio_slice]).mean()),
            "Visual": float(np.abs(shap_values[:, visual_slice]).mean()),
        }

        total = sum(contributions.values()) or 1.0
        return {k: 100 * v / total for k, v in contributions.items()}
