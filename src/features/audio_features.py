"""
Audio feature extraction using Librosa.

Extracts 54-dimensional acoustic features:
    - MFCC (13 coefficients, mean and std): 26 dimensions
    - Chroma (12 semitones, mean and std): 24 dimensions
    - Pitch (mean and std): 2 dimensions
    - Energy (RMS): 1 dimension
    - Zero-Crossing Rate: 1 dimension

Authors: Md. Murad Hossain, MD. Foysal Ahammed Joy, Nuruzzaman Shuvo
Supervisor: Md. Morshed Ali
Department: Computer Science and Engineering, Uttara University
"""

import os
from typing import Optional

import numpy as np
import librosa


class AudioFeatureExtractor:
    """Extract acoustic features from interview audio."""

    def __init__(self, sample_rate: int = 16000, duration: int = 60,
                 n_mfcc: int = 13):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc

    @staticmethod
    def _find_audio_file(pid, data_dir: str) -> Optional[str]:
        files = [f for f in os.listdir(data_dir)
                 if f.startswith(str(pid)) and f.endswith(".wav")]
        if not files:
            return None
        return os.path.join(data_dir, files[0])

    def extract(self, pid, data_dir: str) -> np.ndarray:
        """Extract 54-dim audio features for a single participant."""
        audio_path = self._find_audio_file(pid, data_dir)
        if audio_path is None:
            return np.zeros(54)

        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate,
                                 duration=self.duration)
            if len(y) == 0:
                return np.zeros(54)

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)

            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            if len(pitch_values) > 0:
                pitch_mean = float(np.mean(pitch_values))
                pitch_std = float(np.std(pitch_values))
            else:
                pitch_mean, pitch_std = 0.0, 0.0

            energy = float(np.sum(y ** 2) / len(y))
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = float(np.mean(zcr))

            return np.concatenate([
                mfcc_mean, mfcc_std,
                chroma_mean, chroma_std,
                [pitch_mean, pitch_std, energy, zcr_mean],
            ])
        except Exception:
            return np.zeros(54)
