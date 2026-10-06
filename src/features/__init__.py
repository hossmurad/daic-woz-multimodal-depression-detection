"""
Feature extraction modules for DAIC-WOZ.

Modules:
    text_features:    BERT-based textual embeddings (768-dim)
    audio_features:   Librosa-based acoustic features (54-dim)
    visual_features:  OpenFace-based facial features (50-dim)
    late_fusion:      Concatenation of all modalities (872-dim)
"""

from .text_features import TextFeatureExtractor
from .audio_features import AudioFeatureExtractor
from .visual_features import VisualFeatureExtractor
from .late_fusion import LateFusionExtractor

__all__ = [
    "TextFeatureExtractor",
    "AudioFeatureExtractor",
    "VisualFeatureExtractor",
    "LateFusionExtractor",
]
