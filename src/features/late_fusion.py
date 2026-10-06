"""
Late Fusion module - combines text, audio, and visual features.

Fusion strategy:
    Fused = Text (768) concatenated with Audio (54) and Visual (50)
    Total = 872 dimensions

This preserves modality-specific semantics and avoids the noise
problems of early fusion.

Authors: Md. Murad Hossain, MD. Foysal Ahammed Joy, Nuruzzaman Shuvo
Supervisor: Md. Morshed Ali
Department: Computer Science and Engineering, Uttara University
"""

from typing import Dict

import numpy as np

from .text_features import TextFeatureExtractor
from .audio_features import AudioFeatureExtractor
from .visual_features import VisualFeatureExtractor


class LateFusionExtractor:
    """
    Combine three modalities via Late Fusion.

    Text:   768 dims (BERT)
    Audio:   54 dims (Librosa)
    Visual:  50 dims (OpenFace)
    Total:  872 dims
    """

    def __init__(self):
        self.text_extractor = TextFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        self.visual_extractor = VisualFeatureExtractor()

    def extract(self, pid, data_dir: str) -> Dict[str, np.ndarray]:
        """
        Extract all modalities for a single participant.

        Returns:
            Dictionary with keys: 'text', 'audio', 'visual', 'combined'
        """
        text = self.text_extractor.extract(pid, data_dir)
        audio = self.audio_extractor.extract(pid, data_dir)
        visual = self.visual_extractor.extract(pid, data_dir)

        return {
            "text": text,
            "audio": audio,
            "visual": visual,
            "combined": np.concatenate([text, audio, visual]),
        }
