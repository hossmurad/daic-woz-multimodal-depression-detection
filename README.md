<div align="center">

# A Multimodal Explainable AI Framework for Depression Detection

### Integrating Textual, Acoustic, and Visual Cues via Late Fusion

<br>

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Type-Research-F59E0B?style=for-the-badge)]()
[![Explainable AI](https://img.shields.io/badge/XAI-SHAP-8B5CF6?style=for-the-badge)]()

<br>

**Department of Computer Science and Engineering**
**Uttara University · Dhaka, Bangladesh**

<br>

[Overview](#overview) · [Architecture](#architecture) · [Results](#results) · [Installation](#installation) · [Usage](#usage) · [Citation](#citation)

</div>

---

## Authorship

<table align="center">
  <thead>
    <tr>
      <th align="center">Role</th>
      <th align="center">Name</th>
      <th align="center">Affiliation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>Supervisor</b></td>
      <td align="center"><b>Md. Morshed Ali</b></td>
      <td align="center">Lecturer and Coordinator<br>Department of CSE, Uttara University</td>
    </tr>
    <tr>
      <td align="center">Author</td>
      <td align="center">Md. Murad Hossain</td>
      <td align="center">Department of CSE, Uttara University</td>
    </tr>
    <tr>
      <td align="center">Author</td>
      <td align="center">MD. Foysal Ahammed Joy</td>
      <td align="center">Department of CSE, Uttara University</td>
    </tr>
    <tr>
      <td align="center">Author</td>
      <td align="center">Nuruzzaman Shuvo</td>
      <td align="center">Department of CSE, Uttara University</td>
    </tr>
  </tbody>
</table>

---

## Overview

This repository presents a **Late Fusion multimodal framework** for automated depression detection on the DAIC-WOZ benchmark. The framework combines three complementary modalities to capture behavioral, linguistic, and acoustic markers of depression:

| Modality | Features | Dimension |
|:--------:|:---------|:---------:|
| **Textual** | BERT contextual embeddings from participant utterances | 768 |
| **Acoustic** | MFCC, Chroma, Pitch, Energy, Zero-Crossing Rate | 54 |
| **Visual** | Action Units, Head Pose, Eye Gaze (OpenFace) | 50 |
| **Fused** | Concatenation (Late Fusion) | **872** |

To address the "black-box" limitation of clinical AI systems, the framework integrates **SHAP (SHapley Additive exPlanations)** at both global and patient levels.

---

## Key Results

<div align="center">

### 5-Fold Stratified Cross-Validation Performance

| Metric | Score |
|:------:|:-----:|
| **F1-Score** | **0.569** |
| **ROC-AUC** | **0.697** |
| **Accuracy** | **0.681** |
| Precision | 0.475 |
| Recall | 0.707 |

### Modality Contribution (SHAP)

| Modality | Contribution |
|:--------:|:------------:|
| **Audio** | **51.0%** |
| Text | 29.2% |
| Visual | 19.8% |

</div>

---

## Architecture

<div align="center">
  <img src="paper/figures/figure1_architecture.png" alt="Multimodal Late Fusion Architecture" width="900">
</div>

<br>

**Pipeline Stages:**

| Stage | Operation | Output Shape |
|:-----:|:----------|:------------:|
| **1** | Input: DAIC-WOZ interview (transcript + audio + video) | — |
| **2** | Feature Extraction: BERT (768) + Librosa (54) + OpenFace (50) | 872 |
| **3** | Late Fusion: Concatenation of modalities | 872 |
| **4** | Per-Modality PCA: Text (30) + Audio (15) + Visual (10) | 55 |
| **5** | SMOTE Oversampling (k=3) | 55 |
| **6** | Random Forest (n=200, depth=6, balanced_subsample) | — |
| **7** | Threshold Tuning (optimal = 0.40) | — |
| **8** | Output: Depressed / Non-depressed | — |

---

## Results

### Modality Contribution

<div align="center">
  <img src="paper/figures/figure2_modality_contribution.png" alt="Modality Contribution" width="600">
</div>

Acoustic features contribute the most to predictions (51.0%), followed by textual (29.2%) and visual (19.8%) modalities.

### SHAP Feature Importance

<div align="center">
  <img src="paper/figures/figure3_shap_summary.png" alt="SHAP Summary Plot" width="800">
</div>

Top 15 most impactful features across all three modalities, ranked by mean absolute SHAP value.

### Confusion Matrix

<div align="center">
  <img src="paper/figures/figure4_confusion_matrix.png" alt="Confusion Matrix" width="500">
</div>

Confusion matrix from the Late Fusion model with tuned threshold (0.40).

---

## Repository Structure

```
daic-woz-multimodal-depression-detection/
│
├── configs/
│   └── config.yaml                     Project configuration
│
├── src/
│   ├── config.py                       Configuration loader
│   ├── features/
│   │   ├── text_features.py            BERT embeddings (768-dim)
│   │   ├── audio_features.py           Librosa features (54-dim)
│   │   ├── visual_features.py          OpenFace features (50-dim)
│   │   └── late_fusion.py              Multimodal fusion
│   ├── models/
│   │   └── late_fusion_model.py        Classifier pipeline
│   ├── evaluation/
│   │   └── metrics.py                  Metrics and cross-validation
│   └── explainability/
│       └── shap_analysis.py            SHAP explainability
│
├── scripts/
│   ├── extract_features.py             Feature extraction CLI
│   ├── train_model.py                  Training CLI
│   └── run_pipeline.py                 End-to-end pipeline
│
├── notebooks/
│   ├── 01_feature_extraction.ipynb
│   ├── 02_training_late_fusion.ipynb
│   └── 03_shap_explainability.ipynb
│
├── docs/
│   ├── methodology.md                  Detailed methodology
│   ├── results.md                      Complete results
│   ├── dataset.md                      Dataset information
│   └── reproducibility.md              Reproducibility guide
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── results/
│   ├── figures/                        Result figures
│   └── tables/                         Result tables
│
├── paper/
│   ├── paper.md                        Paper draft
│   ├── paper_draft.pdf
│   └── figures/                        Publication figures
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/hossmurad/daic-woz-multimodal-depression-detection.git
cd daic-woz-multimodal-depression-detection

# Install dependencies
pip install -r requirements.txt
```

### System Requirements

| Component | Requirement |
|:----------|:------------|
| **Python** | 3.9 or higher |
| **RAM** | 8 GB minimum (16 GB recommended) |
| **GPU** | Recommended for BERT feature extraction |
| **Storage** | 20 GB free disk space |

---

## Dataset

The **DAIC-WOZ** (Distress Analysis Interview Corpus — Wizard of Oz) dataset is the gold-standard benchmark for multimodal clinical depression evaluation.

### Access

The dataset is available under **controlled access** from the University of Southern California (USC) Institute for Creative Technologies:

**https://dcapswoz.ict.usc.edu**

A signed data usage agreement is required. Approval typically takes 3-7 business days.

> **Note:** The raw dataset is **not** included in this repository. Users must obtain it separately from USC ICT.

### Labeling

| PHQ-8 Score | Classification |
|:-----------:|:---------------|
| ≥ 10 | Depressed (label = 1) |
| < 10 | Non-depressed (label = 0) |

---

## Usage

### Feature Extraction

```python
from src.features.late_fusion import LateFusionExtractor

extractor = LateFusionExtractor()
features = extractor.extract(
    participant_id='300',
    data_dir='path/to/daic_woz_raw'
)

print(features['text'].shape)      # (768,)
print(features['audio'].shape)     # (54,)
print(features['visual'].shape)    # (50,)
print(features['combined'].shape)  # (872,)
```

### Model Training

```python
import numpy as np
from src.models.late_fusion_model import LateFusionClassifier, FusionConfig
from src.evaluation.metrics import (
    cross_validate, compute_metrics, find_optimal_threshold
)

# Load pre-extracted features and labels
X = np.load('data/features/combined_features.npy')
y = np.load('data/features/labels.npy')

# Train with 5-fold cross-validation
config = FusionConfig()
model = LateFusionClassifier(config)
y_prob = cross_validate(model, X, y, n_splits=5)

# Evaluate with tuned threshold
threshold = find_optimal_threshold(y, y_prob)
y_pred = (y_prob >= threshold).astype(int)
metrics = compute_metrics(y, y_pred, y_prob)

print(f"F1:  {metrics['f1']:.4f}")
print(f"AUC: {metrics['auc']:.4f}")
```

### Explainability with SHAP

```python
from src.explainability.shap_analysis import SHAPExplainer

explainer = SHAPExplainer(model, figure_dir='results/figures')
shap_values, X_fused = explainer.compute_shap_values(X)
modality_contribution = explainer.compute_modality_contribution(shap_values)

for modality, pct in modality_contribution.items():
    print(f"{modality}: {pct:.1f}%")
```

### Full Pipeline

```bash
python scripts/run_pipeline.py
```

---

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) folder:

| Document | Description |
|:---------|:------------|
| [Methodology](docs/methodology.md) | Complete framework description |
| [Results](docs/results.md) | All experimental results |
| [Dataset](docs/dataset.md) | DAIC-WOZ details and access |
| [Reproducibility](docs/reproducibility.md) | Step-by-step reproduction guide |

---

## Testing

Run the unit test suite:

```bash
pytest tests/ -v
```

---

## Key Findings

**1. Late fusion outperforms early fusion.**
Preserving modality-specific characteristics through per-modality PCA before fusion yields a substantial AUC improvement over simple concatenation.

**2. Acoustic modality is dominant.**
51% of the predictive signal comes from audio features, consistent with clinical literature on psychomotor retardation in depression.

**3. SMOTE is critical for class imbalance.**
Removing SMOTE oversampling reduces F1 by 0.084, demonstrating its essential role in small clinical datasets.

**4. SHAP provides clinical trust.**
Patient-level explanations highlight interpretable features (speech pauses, negative sentiment) that align with established DSM-5 markers.

---

## Limitations

- **Dataset size** — 184 participants is small relative to deep learning standards.
- **Single dataset** — Results validated only on DAIC-WOZ; cross-dataset generalization remains untested.
- **Binary classification** — Continuous PHQ-8 severity regression is left for future work.
- **No live deployment** — This work validates the software pipeline, not real-time clinical use.

---

## Future Work

- Extend to PHQ-8 regression for continuous severity prediction.
- Validate on additional datasets (E-DAIC, MODMA).
- Explore deep learning architectures (transformers, graph networks).
- Conduct a clinical user study with psychiatrists.

---

## Citation

If you use this code or framework in your research, please cite:

```bibtex
@misc{hossain2026multimodal,
  title        = {A Multimodal Explainable AI Framework for Depression
                  Detection Using Late Fusion of Textual, Acoustic,
                  and Visual Cues},
  author       = {Hossain, Md. Murad and Joy, MD. Foysal Ahammed and
                  Shuvo, Nuruzzaman},
  year         = {2026},
  note         = {Supervisor: Md. Morshed Ali},
  institution  = {Uttara University, Department of Computer Science
                  and Engineering},
  address      = {Dhaka, Bangladesh},
  url          = {https://github.com/hossmurad/daic-woz-multimodal-depression-detection}
}
```

---

## References

1. Gratch, J., Artstein, R., Lucas, G., et al. (2014). The Distress Analysis Interview Corpus: Vector space model of depression. *LREC*, 3123–3128.

2. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*, 4765–4774.

3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*, 4171–4186.

4. Baltrusaitis, T., Zadeh, A., Lim, Y. C., & Morency, L.-P. (2018). OpenFace 2.0: Facial behavior analysis toolkit. *IEEE FG*, 59–66.

5. Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001). The PHQ-9: Validity of a brief depression severity measure. *Journal of General Internal Medicine*, 16(9), 606–613.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

We express our sincere gratitude to the following individuals and institutions:

<table>
  <tr>
    <th align="left">Role</th>
    <th align="left">Name / Institution</th>
  </tr>
  <tr>
    <td><b>Supervisor</b></td>
    <td>Md. Morshed Ali — Lecturer and Coordinator<br>
        Department of Computer Science and Engineering, Uttara University</td>
  </tr>
  <tr>
    <td><b>Institution</b></td>
    <td>Uttara University — Department of Computer Science and Engineering</td>
  </tr>
  <tr>
    <td><b>Dataset Provider</b></td>
    <td>USC Institute for Creative Technologies — DAIC-WOZ Corpus</td>
  </tr>
  <tr>
    <td><b>Open-Source Tools</b></td>
    <td>PyTorch, HuggingFace Transformers, Scikit-learn, SHAP, Librosa, OpenFace</td>
  </tr>
</table>

---

## Contact

**Md. Murad Hossain** — GitHub: [@hossmurad](https://github.com/hossmurad)

For questions, issues, or collaboration, please [open an issue](https://github.com/hossmurad/daic-woz-multimodal-depression-detection/issues).

---

<div align="center">

**Department of Computer Science and Engineering**
**Uttara University · Dhaka, Bangladesh**

© 2026 · Licensed under MIT

<sub>Built with PyTorch, HuggingFace Transformers, and Scikit-learn</sub>

</div>
