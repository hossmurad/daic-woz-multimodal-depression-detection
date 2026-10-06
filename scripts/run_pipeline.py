#!/usr/bin/env python3
"""
End-to-end pipeline for DAIC-WOZ multimodal depression detection.

This script demonstrates the complete workflow:
    1. Load pre-extracted multimodal features
    2. Align with PHQ-8 labels
    3. Train the Late Fusion classifier via cross-validation
    4. Evaluate performance with standard metrics
    5. Save results

Usage:
    python scripts/run_pipeline.py

Authors: Md. Murad Hossain, MD. Foysal Ahammed Joy, Nuruzzaman Shuvo
Supervisor: Md. Morshed Ali
Department of Computer Science and Engineering, Uttara University
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.models.late_fusion_model import LateFusionClassifier, FusionConfig
from src.evaluation.metrics import (
    compute_metrics,
    find_optimal_threshold,
    cross_validate,
)


def load_features(feature_dir: str):
    """
    Load pre-extracted multimodal features from disk.

    Expected directory structure:
        feature_dir/
            batch1/
                combined_features.npy
                text_features.npy
                audio_features.npy
                visual_features.npy
                participant_ids.npy
            batch2/
                ...
            batch3/
                ...
    """
    feature_dir = Path(feature_dir)
    batch_dirs = sorted([d for d in feature_dir.iterdir() if d.is_dir()])

    if not batch_dirs:
        raise FileNotFoundError(f"No batch directories found in {feature_dir}")

    stacks = {
        "text": [],
        "audio": [],
        "visual": [],
        "combined": [],
        "ids": [],
    }

    for batch_dir in batch_dirs:
        files = {
            "text": "text_features.npy",
            "audio": "audio_features.npy",
            "visual": "visual_features.npy",
            "combined": "combined_features.npy",
            "ids": "participant_ids.npy",
        }

        for key, fname in files.items():
            path = batch_dir / fname
            if path.exists():
                stacks[key].append(np.load(path))

    return {
        "text": np.vstack(stacks["text"]),
        "audio": np.vstack(stacks["audio"]),
        "visual": np.vstack(stacks["visual"]),
        "combined": np.vstack(stacks["combined"]),
        "ids": np.concatenate(stacks["ids"]),
    }


def main():
    print("=" * 70)
    print("DAIC-WOZ Multimodal Depression Detection Pipeline")
    print("=" * 70)

    # ------------------------------
    # 1. Load features
    # ------------------------------
    feature_dir = PROJECT_ROOT / "data" / "features"

    if not feature_dir.exists():
        print(f"\nFeature directory not found: {feature_dir}")
        print("\nPlease ensure pre-extracted features are placed under:")
        print(f"  {feature_dir}/batch1/, batch2/, batch3/")
        print("\nRun the feature extraction notebook first to generate features.")
        return

    print(f"\nLoading features from: {feature_dir}")
    features = load_features(feature_dir)
    X = features["combined"]

    print(f"  Combined feature shape: {X.shape}")
    print(f"  Number of participants: {len(features['ids'])}")

    # ------------------------------
    # 2. Load labels (placeholder)
    # ------------------------------
    print("\nNote: Label loading requires the DAIC-WOZ PHQ-8 labels.")
    print("Please provide your label mapping in the code before running.")

    # ------------------------------
    # 3. Train model (example)
    # ------------------------------
    print("\nPipeline ready. To train:")
    print("  1. Load labels into variable 'y'")
    print("  2. Uncomment the training block below")

    # Example training block:
    # config = FusionConfig()
    # model = LateFusionClassifier(config)
    # y_prob = cross_validate(model, X, y, n_splits=5)
    # threshold = find_optimal_threshold(y, y_prob)
    # y_pred = (y_prob >= threshold).astype(int)
    # metrics = compute_metrics(y, y_pred, y_prob)
    # print("\nMetrics:")
    # for name, value in metrics.items():
    #     print(f"  {name}: {value:.4f}")

    print("\n" + "=" * 70)
    print("Pipeline initialization complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
