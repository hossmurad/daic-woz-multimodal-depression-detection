# Results

## 1. Overall Performance

5-fold Stratified Cross-Validation results:

| Metric | Value |
|--------|-------|
| F1-Score | 0.569 |
| ROC-AUC | 0.697 |
| Accuracy | 0.681 |
| Recall | 0.707 |
| Precision | 0.475 |

---

## 2. Comparison with Baselines

| Model | F1 | AUC | Accuracy |
|-------|-----|-----|----------|
| Logistic Regression | 0.500 | 0.594 | 0.638 |
| Random Forest | 0.467 | 0.588 | 0.587 |
| XGBoost | 0.420 | 0.579 | 0.600 |
| Late Fusion + SMOTE | 0.531 | 0.683 | 0.696 |
| Late Fusion (Tuned) | 0.569 | 0.697 | 0.681 |

---

## 3. Ablation Study

| Configuration | F1 | AUC |
|---------------|-----|-----|
| Text only | 0.483 | 0.589 |
| Audio only | 0.505 | 0.606 |
| Visual only | 0.453 | 0.507 |
| Text + Audio | 0.530 | 0.664 |
| All 3 | 0.569 | 0.697 |
| All 3 (No SMOTE) | 0.485 | 0.631 |

---

## 4. Modality Contribution

| Modality | Contribution |
|----------|--------------|
| Text | 29.2% |
| Audio | 51.0% |
| Visual | 19.8% |

---

## 5. Case Study

Participant with PHQ-8 >= 10: predicted Depressed with probability 0.786.
Top features: audio_pca_0 (+0.052), text_pca_1 (+0.034), audio_pca_8 (+0.029).

---

## 6. Confusion Matrix

|              | Predicted ND | Predicted D |
|--------------|--------------|-------------|
| True ND      | 28           | 3           |
| True D       | 2            | 5           |
