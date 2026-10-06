"""
Visual feature extraction from OpenFace 2.0 outputs.

Extracts 50-dimensional facial behavioral features:
    - Action Units (17 AU intensities, mean and std): 34 dimensions
    - Head Pose (Tx, Ty, Tz, Rx, Ry, Rz): 6 dimensions
    - Eye Gaze (gaze_0_x/y/z, gaze_1_x): 4 dimensions
    - Padding: 6 dimensions

Authors: Md. Murad Hossain, MD. Foysal Ahammed Joy, Nuruzzaman Shuvo
Supervisor: Md. Morshed Ali
Department: Computer Science and Engineering, Uttara University
"""

import os

import numpy as np
import pandas as pd


class VisualFeatureExtractor:
    """Extract facial behavioral features from OpenFace outputs."""

    def __init__(self, n_action_units: int = 17, output_dim: int = 50):
        self.n_action_units = n_action_units
        self.output_dim = output_dim

    @staticmethod
    def _find_file(pid, data_dir: str, keyword: str):
        files = [f for f in os.listdir(data_dir)
                 if f.startswith(str(pid)) and keyword in f]
        if not files:
            return None
        return os.path.join(data_dir, files[0])

    def extract(self, pid, data_dir: str) -> np.ndarray:
        """Extract 50-dim visual features for a single participant."""
        features = []

        au_path = self._find_file(pid, data_dir, "CLNF_AUs")
        au_means = np.zeros(self.n_action_units)
        au_stds = np.zeros(self.n_action_units)

        if au_path:
            try:
                df = pd.read_csv(au_path)
                au_cols = [c for c in df.columns if "AU" in c and "_r" in c]
                if au_cols:
                    au_means = df[au_cols].mean().values[:self.n_action_units]
                    au_stds = df[au_cols].std().values[:self.n_action_units]
            except Exception:
                pass

        features.extend(au_means)
        features.extend(au_stds)

        pose_path = self._find_file(pid, data_dir, "CLNF_pose")
        pose_means = np.zeros(6)
        if pose_path:
            try:
                df = pd.read_csv(pose_path)
                pose_cols = [c for c in df.columns if "pose_" in c]
                if pose_cols:
                    pose_means = df[pose_cols].mean().values[:6]
            except Exception:
                pass

        features.extend(pose_means)

        gaze_path = self._find_file(pid, data_dir, "CLNF_gaze")
        gaze_means = np.zeros(4)
        if gaze_path:
            try:
                df = pd.read_csv(gaze_path)
                gaze_cols = [c for c in df.columns if "gaze_" in c]
                if gaze_cols:
                    gaze_means = df[gaze_cols].mean().values[:4]
            except Exception:
                pass

        features.extend(gaze_means)

        result = np.array(features)
        if len(result) < self.output_dim:
            result = np.pad(result, (0, self.output_dim - len(result)))
        return result[:self.output_dim]
