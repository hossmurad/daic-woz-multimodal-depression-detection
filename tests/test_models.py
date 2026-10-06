"""Unit tests for model modules."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def test_late_fusion_classifier_import():
    from src.models.late_fusion_model import LateFusionClassifier, FusionConfig
    assert LateFusionClassifier is not None
    assert FusionConfig is not None


def test_fusion_config_defaults():
    from src.models.late_fusion_model import FusionConfig
    config = FusionConfig()
    assert config.text_dim == 768
    assert config.audio_dim == 54
    assert config.visual_dim == 50
    assert config.threshold == 0.40
    assert config.random_seed == 42


def test_classifier_pipeline_builds():
    from src.models.late_fusion_model import LateFusionClassifier, FusionConfig
    config = FusionConfig()
    model = LateFusionClassifier(config)
    assert model.pipeline is not None


def test_classifier_fit_predict_smoke():
    from src.models.late_fusion_model import LateFusionClassifier, FusionConfig
    np.random.seed(42)
    n_samples = 40
    X = np.random.randn(n_samples, 872).astype(np.float32)
    y = np.array([0] * 30 + [1] * 10)

    config = FusionConfig(
        text_pca=5, audio_pca=3, visual_pca=2,
        n_estimators=10, max_depth=3,
    )
    model = LateFusionClassifier(config)
    model.fit(X, y)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    assert y_pred.shape == (n_samples,)
    assert y_proba.shape == (n_samples, 2)
