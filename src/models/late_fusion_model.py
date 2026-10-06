"""
Late Fusion classifier with per-modality PCA and SMOTE.

Pipeline:
    1. Per-modality StandardScaler and PCA reduction
    2. Concatenation (Late Fusion)
    3. SMOTE oversampling for class balance
    4. Random Forest classifier with threshold tuning

Authors: Md. Murad Hossain, MD. Foysal Ahammed Joy, Nuruzzaman Shuvo
Supervisor: Md. Morshed Ali
Department: Computer Science and Engineering, Uttara University
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


@dataclass
class FusionConfig:
    """Configuration for Late Fusion classifier."""

    text_dim: int = 768
    audio_dim: int = 54
    visual_dim: int = 50

    text_pca: int = 30
    audio_pca: int = 15
    visual_pca: int = 10

    n_estimators: int = 200
    max_depth: int = 6
    min_samples_leaf: int = 3

    smote_k_neighbors: int = 3
    threshold: float = 0.40
    random_seed: int = 42


class LateFusionClassifier:
    """
    Late Fusion classifier.

    Combines per-modality PCA reduction with SMOTE oversampling and
    a Random Forest classifier optimized for imbalanced clinical data.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.pipeline = self._build_pipeline()
        self._is_fitted = False

    def _build_pipeline(self) -> ImbPipeline:
        cfg = self.config

        text_range = list(range(0, cfg.text_dim))
        audio_range = list(range(cfg.text_dim, cfg.text_dim + cfg.audio_dim))
        visual_range = list(range(
            cfg.text_dim + cfg.audio_dim,
            cfg.text_dim + cfg.audio_dim + cfg.visual_dim,
        ))

        preprocessor = ColumnTransformer([
            ("text", Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(cfg.text_pca, random_state=cfg.random_seed)),
            ]), text_range),
            ("audio", Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(cfg.audio_pca, random_state=cfg.random_seed)),
            ]), audio_range),
            ("visual", Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(cfg.visual_pca, random_state=cfg.random_seed)),
            ]), visual_range),
        ])

        classifier = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=cfg.random_seed,
            n_jobs=-1,
        )

        return ImbPipeline([
            ("fusion", preprocessor),
            ("smote", SMOTE(random_state=cfg.random_seed,
                            k_neighbors=cfg.smote_k_neighbors)),
            ("classifier", classifier),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LateFusionClassifier":
        """Fit the classifier on training data."""
        self.pipeline.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using the tuned threshold."""
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.config.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        return self.pipeline.predict_proba(X)

    def get_fused_representation(self, X: np.ndarray) -> np.ndarray:
        """Return the fused (PCA-reduced) representation."""
        fusion = self.pipeline.named_steps["fusion"]
        return fusion.transform(X)

    def get_classifier(self) -> RandomForestClassifier:
        """Return the underlying Random Forest classifier."""
        return self.pipeline.named_steps["classifier"]
