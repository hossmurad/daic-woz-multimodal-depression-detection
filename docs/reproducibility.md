# Reproducibility Guide

## 1. Installation

    git clone https://github.com/hossmurad/daic-woz-multimodal-depression-detection.git
    cd daic-woz-multimodal-depression-detection
    pip install -r requirements.txt

## 2. Data Acquisition

Request from https://dcapswoz.ict.usc.edu

Place under data/raw/

## 3. Feature Extraction

    python scripts/extract_features.py --participants 300 301 302 --data-dir data/raw --output data/features/batch1

## 4. Model Training

    python scripts/train_model.py --features data/features/batch1 --output results

## 5. Explainability

    jupyter notebook notebooks/03_shap_explainability.ipynb

## 6. Full Pipeline

    python scripts/run_pipeline.py

## 7. Random Seeds

All seeds fixed at 42.

## 8. Expected Runtime

On Colab (T4 GPU):
- Feature extraction per participant: 30 sec
- Model training (5-fold CV): 1 min
- SHAP analysis: 2 min

## 9. Troubleshooting

MemoryError during extraction: reduce batch size.
FileNotFoundError: verify data path.
Low performance: check PHQ-8 threshold (10) and SMOTE.
