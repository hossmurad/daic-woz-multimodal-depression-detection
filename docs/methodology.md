# Methodology

Complete description of the multimodal framework for depression detection.

---

## 1. Dataset

The DAIC-WOZ dataset is the gold-standard benchmark for multimodal clinical
depression evaluation. We use the official AVEC 2017 train/dev split,
yielding 184 participants with PHQ-8 labels.

---

## 2. Feature Extraction

### 2.1 Textual Features (768-dim)

Model: BERT-base-uncased. Only participant utterances are used to prevent
interviewer bias.

### 2.2 Acoustic Features (54-dim)

Library: Librosa. MFCC (26), Chroma (24), Pitch (2), Energy (1), ZCR (1).

### 2.3 Visual Features (50-dim)

Tool: OpenFace 2.0. Action Units (34), Head Pose (6), Eye Gaze (4), Padding (6).

### 2.4 Late Fusion

Fused vector = Text (768) + Audio (54) + Visual (50) = 872 dims.

---

## 3. Model Architecture

1. Per-modality StandardScaler
2. Per-modality PCA: Text (768 to 30), Audio (54 to 15), Visual (50 to 10)
3. Concatenation to 55 dims
4. SMOTE oversampling (k=3)
5. Random Forest (n=200, depth=6)
6. Threshold tuning (optimal 0.40)

---

## 4. Evaluation Protocol

5-fold Stratified Cross-Validation. Metrics: Accuracy, Precision, Recall,
F1-Score, ROC-AUC.

---

## 5. Explainability

SHAP TreeExplainer on the fused representation. Global, per-modality, and
per-participant explanations.

---

## References

1. Gratch, G., et al. (2014). The Distress Analysis Interview Corpus. LREC.
2. Devlin, J., et al. (2019). BERT. NAACL-HLT.
3. Baltrusaitis, T., et al. (2018). OpenFace 2.0. IEEE FG.
4. Lundberg, S. M., & Lee, S.-I. (2017). SHAP. NeurIPS.
