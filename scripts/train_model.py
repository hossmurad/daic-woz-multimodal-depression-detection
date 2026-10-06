#!/usr/bin/env python3
"""
Training script for the Late Fusion classifier.

Usage:
    python scripts/train_model.py --features data/features/batch1 --output results

Authors: Md. Murad Hossain, MD. Foysal Ahammed Joy, Nuruzzaman Shuvo
Supervisor: Md. Morshed Ali
Department of Computer Science and Engineering, Uttara University
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib

from src.models.late_fusion_model import LateFusionClassifier, FusionConfig
from src.evaluation.metrics import (
    compute_metrics,
    find_optimal_threshold,
    cross_validate,
)


def main():
    parser = argparse.ArgumentParser(description="Train Late Fusion classifier")
    parser.add_argument("--features", required=True,
                        help="Directory with combined_features.npy and labels.npy")
    parser.add_argument("--output", default="results",
                        help="Output directory for model and results")
    args = parser.parse_args()

    feature_dir = Path(args.features)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load features and labels
    X = np.load(feature_dir / "combined_features.npy")
    y = np.load(feature_dir / "labels.npy")

    print(f"Loaded features: {X.shape}")
    print(f"Loaded labels: {y.shape}")
    print(f"  Depressed: {int(y.sum())}")
    print(f"  Non-depressed: {int((y == 0).sum())}")

    # Train model with cross-validation
    config = FusionConfig()
    model = LateFusionClassifier(config)

    print("\nRunning 5-fold cross-validation...")
    y_prob = cross_validate(model, X, y, n_splits=5)

    threshold = find_optimal_threshold(y, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y, y_pred, y_prob)

    print(f"\nOptimal threshold: {threshold:.2f}")
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Train final model on all data
    print("\nTraining final model on all data...")
    model.fit(X, y)

    model_path = output_dir / "final_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()
