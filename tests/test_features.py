"""Unit tests for feature extraction modules."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def test_text_extractor_import():
    from src.features.text_features import TextFeatureExtractor
    assert TextFeatureExtractor is not None


def test_audio_extractor_import():
    from src.features.audio_features import AudioFeatureExtractor
    assert AudioFeatureExtractor is not None


def test_visual_extractor_import():
    from src.features.visual_features import VisualFeatureExtractor
    assert VisualFeatureExtractor is not None


def test_late_fusion_import():
    from src.features.late_fusion import LateFusionExtractor
    assert LateFusionExtractor is not None


def test_audio_extractor_dimensions():
    from src.features.audio_features import AudioFeatureExtractor
    extractor = AudioFeatureExtractor()
    features = extractor.extract("nonexistent", "/tmp")
    assert features.shape == (54,)
    assert np.all(features == 0)


def test_visual_extractor_dimensions():
    from src.features.visual_features import VisualFeatureExtractor
    extractor = VisualFeatureExtractor()
    features = extractor.extract("nonexistent", "/tmp")
    assert features.shape == (50,)
    assert np.all(features == 0)
