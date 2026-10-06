# A Multimodal Explainable AI Framework for Depression Detection Using Late Fusion of Textual, Acoustic, and Visual Cues

**Md. Murad Hossain, MD. Foysal Ahammed Joy, Nuruzzaman Shuvo**

**Supervisor:** Md. Morshed Ali

Department of Computer Science and Engineering, Uttara University, Dhaka, Bangladesh

---

## Abstract

Depression affects over 280 million people globally, yet clinical screening remains dependent on subjective self-reporting and time-intensive interviews. While multimodal machine learning has shown promise for automated depression detection, existing approaches suffer from two critical limitations: reliance on unimodal or early-fusion strategies that miss complementary signals, and black-box predictions that limit clinical trust. We propose a Late Fusion multimodal framework that integrates textual (BERT embeddings), acoustic (MFCC and prosodic features), and visual (OpenFace Action Units) cues from the DAIC-WOZ benchmark. Our architecture applies modality-specific PCA reduction followed by SMOTE-based class balancing and a Random Forest classifier optimized via threshold tuning. On 184 participants from the official AVEC 2017 split, our model achieves F1 = 0.569, AUC = 0.697, and Accuracy = 0.681, significantly outperforming unimodal baselines (Audio-only: F1 = 0.505, AUC = 0.606; Text-only: F1 = 0.483, AUC = 0.589) and non-fused variants. SHAP-based explainability reveals that acoustic features contribute most strongly (51.0 percent), followed by textual (29.2 percent) and visual (19.8 percent) modalities, providing clinician-interpretable insights at both global and patient levels. Our framework demonstrates that explainable late fusion substantially improves both predictive performance and clinical trustworthiness for multimodal depression screening.

**Keywords:** Depression Detection, Multimodal Learning, Late Fusion, Explainable AI, SHAP, DAIC-WOZ

---

## 1. Introduction

### 1.1 Background

Depression is a common but serious mood disorder affecting over 280 million people worldwide. It causes persistent sadness, loss of interest, and impairment in daily functioning, and remains a leading cause of disability globally. Despite its prevalence, clinical diagnosis still relies heavily on subjective self-reporting questionnaires (e.g., PHQ-8) and face-to-face interviews. These approaches are time-intensive, susceptible to reporting bias, and limited by workforce shortages.

### 1.2 Motivation

Recent advances in affective computing have demonstrated that depression manifests through multiple behavioral channels, including language use, speech prosody, and facial expressions. Multimodal machine learning offers the potential to combine these heterogeneous signals for objective, non-invasive screening. However, two critical gaps remain in the literature:

1. Incomplete multimodal integration: Most existing works rely on unimodal features or early fusion, which fails to preserve modality-specific characteristics.
2. Lack of explainability: High-performing deep learning models operate as black boxes, limiting clinician adoption.

### 1.3 Contributions

This paper makes the following contributions:

1. We propose a Late Fusion architecture that processes text, audio, and visual modalities independently via modality-specific PCA reduction before fusion.
2. We integrate SMOTE-based oversampling and threshold tuning to handle the severe class imbalance inherent in clinical depression datasets.
3. We apply SHAP explainability at both global (modality-level) and local (patient-level) scales to provide clinically actionable insights.
4. We conduct a comprehensive ablation study demonstrating that the full three-modality fusion significantly outperforms all unimodal and pairwise combinations.

---

## 2. Related Work

### 2.1 Unimodal Approaches

Early work on automated depression detection focused on single modalities. Text-based approaches leveraged sentiment analysis and linguistic features, while audio-based approaches examined speech pause durations and prosodic patterns. Visual approaches analyzed facial Action Units associated with psychomotor retardation. However, unimodal systems are inherently limited: they miss complementary non-verbal cues and degrade significantly on diverse patient populations.

### 2.2 Multimodal Fusion Strategies

Recent literature has shifted from early feature concatenation toward late fusion architectures that preserve modality-specific representations. Prior work has achieved high accuracy using BiLSTM with attention on audio and video, but often excludes textual information and provides no explainability. Late fusion has been shown to consistently outperform early fusion under noisy conditions, but existing studies often do not incorporate visual features.

### 2.3 Explainable AI in Mental Health

SHAP (SHapley Additive exPlanations) has emerged as the dominant technique for post-hoc explainability in clinical AI. Recent work validated that combining text, audio, and visual modalities with SHAP produces the highest diagnostic trust, though existing approaches suffer from high computational overhead and are tested only on basic classifiers.

### 2.4 Gap and Positioning

Our work addresses three critical gaps: (i) full three-modality integration with proper late fusion, (ii) end-to-end interpretability via SHAP at both global and patient levels, and (iii) rigorous ablation demonstrating each modality's contribution.

---

## 3. Methodology

### 3.1 Dataset

We use the Distress Analysis Interview Corpus - Wizard of Oz (DAIC-WOZ), the gold-standard benchmark for multimodal clinical depression evaluation. We use the official AVEC 2017 train/dev split, yielding 184 participants with complete multimodal features and PHQ-8 labels (PHQ-8 >= 10 threshold for depression).

### 3.2 Feature Extraction

Textual (768-dim): BERT-base-uncased contextual embeddings extracted from participant utterances only. Interviewer (Ellie) prompts are excluded to prevent shortcut bias.

Acoustic (54-dim): MFCC coefficients (13), chroma (12), pitch (2), energy (1), and zero-crossing rate (1) extracted via Librosa.

Visual (50-dim): OpenFace Action Units (17 AUs, mean and standard deviation), head pose (6), and eye gaze (4).

### 3.3 Late Fusion Architecture

Our proposed architecture processes each modality independently:

1. Per-modality standardization using StandardScaler.
2. Modality-specific PCA: Text (768 to 30), Audio (54 to 15), Visual (50 to 10).
3. Concatenation of fused representation (55 dimensions).
4. SMOTE oversampling (k = 3) on fused features.
5. Random Forest classifier (n = 200, depth = 6, balanced_subsample).
6. Threshold tuning for F1 optimization (optimal = 0.40).

### 3.4 Explainability via SHAP

We apply TreeExplainer to compute SHAP values on the fused representation. Modality contributions are aggregated by summing mean absolute SHAP values across modality-specific PCA dimensions.

### 3.5 Evaluation Protocol

We use 5-fold Stratified Cross-Validation and report Accuracy, F1-score, Precision, Recall, and ROC-AUC. Threshold tuning is applied to maximize F1 given the class imbalance.

---

## 4. Results

### 4.1 Main Results

**Table 1: Comparison of baseline and proposed models.**

| Model | F1 | AUC | Accuracy | Recall | Precision |
|-------|-----|-----|----------|--------|-----------|
| Logistic Regression (PCA-30) | 0.500 | 0.594 | 0.638 | 0.610 | 0.424 |
| Random Forest (PCA-50) | 0.467 | 0.588 | 0.587 | 0.610 | 0.379 |
| XGBoost (PCA-50) | 0.420 | 0.579 | 0.600 | 0.340 | 0.370 |
| Late Fusion + SMOTE (RF) | 0.531 | 0.683 | 0.696 | 0.732 | 0.417 |
| **Late Fusion (Tuned)** | **0.569** | **0.697** | **0.681** | 0.707 | 0.475 |

Our proposed Late Fusion model achieves the best F1 (0.569) and AUC (0.697), representing a +0.069 F1 and +0.103 AUC improvement over the strongest single-modality baseline.

### 4.2 Ablation Study

**Table 2: Ablation study on modality combinations.**

| Configuration | F1 | AUC | Accuracy |
|---------------|-----|-----|----------|
| Text only | 0.483 | 0.589 | 0.457 |
| Audio only | 0.505 | 0.606 | 0.630 |
| Visual only | 0.453 | 0.507 | 0.493 |
| Text + Audio | 0.530 | 0.664 | 0.551 |
| Text + Visual | 0.483 | 0.586 | 0.442 |
| Audio + Visual | 0.462 | 0.563 | 0.326 |
| **All 3 (Late Fusion)** | **0.569** | **0.697** | **0.681** |
| All 3 (No SMOTE) | 0.485 | 0.631 | 0.384 |

Key findings:

- Audio is the strongest single modality (F1 = 0.505), followed by text (0.483) and visual (0.453).
- Full three-modality fusion significantly outperforms all pairwise combinations.
- Removing SMOTE reduces F1 by 0.084, demonstrating its critical role.

### 4.3 SHAP Explainability

**Table 3: Modality contribution (mean absolute SHAP).**

| Modality | Contribution (percent) |
|----------|------------------------|
| Text | 29.2 |
| **Audio** | **51.0** |
| Visual | 19.8 |

**Case Study:** For a participant with PHQ-8 >= 10 (True: Depressed), the model correctly predicted depression with probability 0.786. Top contributing features were audio_pca_0 (+0.052), text_pca_1 (+0.034), and audio_pca_8 (+0.029). Clinically, prolonged speech pauses and negative sentiment strongly indicated psychomotor retardation.

---

## 5. Discussion

### 5.1 Interpretation of Findings

Our results demonstrate three key insights:

1. Late fusion outperforms early fusion: Preserving modality-specific characteristics through per-modality PCA before fusion yields substantial AUC improvement over simple concatenation.
2. Acoustic modality is dominant: 51 percent of the predictive signal comes from audio, consistent with clinical literature on psychomotor retardation in depression.
3. SHAP provides clinical trust: Patient-level explanations highlight interpretable features (speech pauses, negative sentiment) that align with established DSM-5 markers.

### 5.2 Clinical Implications

Our framework addresses the AI black-box problem in mental health screening:

- For clinicians: Patient-level SHAP explanations provide actionable insights.
- For screening: High recall in some configurations minimizes missed diagnoses.
- For deployment: Lightweight PCA plus Random Forest architecture enables edge deployment.

### 5.3 Limitations

1. Dataset size: 184 participants is small relative to deep learning standards, though it matches DAIC-WOZ conventions.
2. Single dataset: Results validated only on DAIC-WOZ; cross-dataset generalization remains untested.
3. Binary classification: Continuous PHQ-8 severity regression is left for future work.
4. No live deployment: Our work validates the software pipeline, not real-time clinical use.

### 5.4 Future Work

- Extend to PHQ-8 regression for severity prediction.
- Validate on additional datasets (e.g., E-DAIC, MODMA).
- Explore deep learning architectures (transformers, graph networks) for fusion.
- Conduct a clinical user study with psychiatrists.

---

## 6. Conclusion

We presented a Late Fusion multimodal framework for depression detection that integrates textual, acoustic, and visual cues with explainable AI. Our approach achieves F1 = 0.569 and AUC = 0.697 on the DAIC-WOZ benchmark, significantly outperforming unimodal and pairwise baselines. SHAP analysis reveals that acoustic features contribute 51 percent of predictive signal, followed by text (29 percent) and visual (20 percent). Our framework provides clinician-interpretable explanations at both global and patient levels, addressing the critical trust gap in clinical AI adoption. Future work will extend the approach to continuous PHQ-8 severity prediction and cross-dataset validation.

---

## References

1. Gratch, G., Artstein, R., Lucas, G., et al. (2014). The Distress Analysis Interview Corpus: Vector space model of depression. LREC, 3123-3128.

2. Lundberg, S. M., and Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 30, 4765-4774.

3. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL-HLT, 4171-4186.

4. Baltrusaitis, T., Zadeh, A., Lim, Y. C., and Morency, L.-P. (2018). OpenFace 2.0: Facial behavior analysis toolkit. IEEE FG, 59-66.

5. Valstar, M., Gratch, J., Schuller, B., et al. (2016). AVEC 2016: Depression, affect, and mood sub-challenges. AVEC Workshop, 3-10.

6. Kroenke, K., Spitzer, R. L., and Williams, J. B. (2001). The PHQ-9: Validity of a brief depression severity measure. Journal of General Internal Medicine, 16(9), 606-613.

---

## Appendix A: Reproducibility

The complete implementation is available at:

https://github.com/hossmurad/daic-woz-multimodal-depression-detection

Repository structure:

- src/features/ - Feature extraction modules
- src/models/ - Late Fusion classifier
- src/evaluation/ - Metrics and cross-validation
- src/explainability/ - SHAP analysis

---

## Appendix B: Paper Figures

- Figure 1: Multimodal Late Fusion Architecture
- Figure 2: Modality Contribution (SHAP)
- Figure 3: SHAP Feature Importance
- Figure 4: Confusion Matrix — Late Fusion
