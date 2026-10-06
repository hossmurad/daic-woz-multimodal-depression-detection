# DAIC-WOZ Multimodal Depression Detection

A Multimodal Explainable AI Framework for Depression Detection
Integrating Textual, Acoustic, and Visual Cues

## Authors

- Md. Murad Hossain
- MD. Foysal Ahammed Joy
- Nuruzzaman Shuvo

**Supervisor:** Md. Morshed Ali
**Department:** Computer Science and Engineering, Uttara University

## Overview

This repository implements a Late Fusion multimodal framework for
automated depression detection using the DAIC-WOZ dataset.

### Modalities

- Textual features (768-dim BERT embeddings)
- Acoustic features (54-dim MFCC + Chroma + Pitch + Energy)
- Visual features (50-dim Action Units + Head Pose + Gaze)

### Architecture

1. Per-modality StandardScaler and PCA reduction
2. Late Fusion via concatenation (55 dimensions)
3. SMOTE oversampling for class balance
4. Random Forest classifier (200 trees)
5. Threshold tuning for F1 optimization

### Explainability

SHAP provides both global feature importance and per-modality contribution
analysis.

## Key Results

| Metric | Value |
|--------|-------|
| F1-Score | 0.569 |
| ROC-AUC | 0.697 |
| Accuracy | 0.681 |

## Modality Contribution (SHAP)

| Modality | Contribution |
|----------|--------------|
| Audio | 51.0% |
| Text | 29.2% |
| Visual | 19.8% |

## Repository Structure

    .
    ├── configs/
    │   └── config.yaml
    ├── src/
    │   ├── config.py
    │   ├── features/
    │   │   ├── text_features.py
    │   │   ├── audio_features.py
    │   │   ├── visual_features.py
    │   │   └── late_fusion.py
    │   ├── models/
    │   │   └── late_fusion_model.py
    │   ├── evaluation/
    │   │   └── metrics.py
    │   └── explainability/
    │       └── shap_analysis.py
    ├── notebooks/
    ├── results/
    │   ├── figures/
    │   └── tables/
    ├── paper/
    ├── requirements.txt
    └── README.md

## Installation

    git clone https://github.com/hossmurad/daic-woz-multimodal-depression-detection.git
    cd daic-woz-multimodal-depression-detection
    pip install -r requirements.txt

## Dataset

The DAIC-WOZ dataset must be obtained separately from the University of
Southern California (USC) Institute for Creative Technologies. Access is
controlled and requires a signed data usage agreement.

## Usage

### Feature Extraction

    from src.features.late_fusion import LateFusionExtractor

    extractor = LateFusionExtractor()
    features = extractor.extract(participant_id='300', data_dir='path/to/data')

### Training

    from src.models.late_fusion_model import LateFusionClassifier, FusionConfig
    from src.evaluation.metrics import cross_validate, compute_metrics, find_optimal_threshold

    config = FusionConfig()
    model = LateFusionClassifier(config)
    y_prob = cross_validate(model, X, y, n_splits=5)

    threshold = find_optimal_threshold(y, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y, y_pred, y_prob)

### Explainability

    from src.explainability.shap_analysis import SHAPExplainer

    explainer = SHAPExplainer(model, figure_dir='results/figures')
    shap_values, X_fused = explainer.compute_shap_values(X)
    modality_contribution = explainer.compute_modality_contribution(shap_values)

## Citation

    Hossain, M. M., Joy, M. F. A., Shuvo, N. (2026).
    A Multimodal Explainable AI Framework for Depression Detection
    Integrating Textual, Acoustic, and Visual Cues.
    Uttara University, Department of Computer Science and Engineering.

## License

MIT License
