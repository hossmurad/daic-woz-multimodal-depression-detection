# DAIC-WOZ Multimodal Depression Detection

A Multimodal Explainable AI Framework for Depression Detection Integrating Textual, Acoustic, and Visual Cues

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()
[![SHAP](https://img.shields.io/badge/explainability-SHAP-purple.svg)]()

---

## Overview

This repository implements a **Late Fusion multimodal framework** for automated depression detection using the DAIC-WOZ dataset. The framework combines three complementary modalities:

- **Textual features** — 768-dim BERT embeddings
- **Acoustic features** — 54-dim MFCC, Chroma, Pitch, Energy, ZCR
- **Visual features** — 50-dim Action Units, Head Pose, Eye Gaze

The framework integrates **Explainable AI (SHAP)** at both global and patient levels, addressing the "black-box" problem in clinical AI.

---

## Key Results

| Metric | Value |
|:------:|:-----:|
| **F1-Score** | **0.569** |
| **ROC-AUC** | **0.697** |
| **Accuracy** | **0.681** |
| Precision | 0.475 |
| Recall | 0.707 |

**Modality Contribution (SHAP)**

| Modality | Contribution |
|:--------:|:------------:|
| Audio | **51.0%** |
| Text | 29.2% |
| Visual | 19.8% |

---

## Architecture

<p align="center">
  <img src="paper/figures/figure1_architecture.png" alt="Multimodal Late Fusion Architecture" width="900">
</p>

**Workflow:**

1. **Input** — DAIC-WOZ interview (transcript + audio + video)
2. **Feature Extraction** — BERT (768) + Librosa (54) + OpenFace (50)
3. **Late Fusion** — Concatenation to 872 dimensions
4. **Per-Modality PCA** — 30 + 15 + 10 = 55 dimensions
5. **SMOTE Oversampling** — Class balancing
6. **Random Forest** — 200 trees, depth 6
7. **Output** — Depressed / Non-depressed

---

## Results

### Modality Contribution

<p align="center">
  <img src="paper/figures/figure2_modality_contribution.png" alt="Modality Contribution" width="600">
</p>

### SHAP Feature Importance

<p align="center">
  <img src="paper/figures/figure3_shap_summary.png" alt="SHAP Summary" width="800">
</p>

### Confusion Matrix

<p align="center">
  <img src="paper/figures/figure4_confusion_matrix.png" alt="Confusion Matrix" width="500">
</p>

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

**System Requirements:**

- Python 3.9 or higher
- 8 GB RAM minimum (16 GB recommended)
- GPU recommended for feature extraction (BERT)
- 20 GB free disk space

---

## Dataset

The **DAIC-WOZ** (Distress Analysis Interview Corpus — Wizard of Oz) dataset is the gold-standard benchmark for multimodal clinical depression evaluation.

**Access:**

The dataset is available under controlled access from the University of Southern California (USC) Institute for Creative Technologies:

**https://dcapswoz.ict.usc.edu**

A signed data usage agreement is required.

> **Note:** The raw dataset is **not** included in this repository. Users must obtain it separately.

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

| Document | Description |
|:---------|:------------|
| [Methodology](docs/methodology.md) | Complete framework description |
| [Results](docs/results.md) | All experimental results |
| [Dataset](docs/dataset.md) | DAIC-WOZ details and access |
| [Reproducibility](docs/reproducibility.md) | Step-by-step guide |

---

## Testing

```bash
pytest tests/ -v
```

---

## Key Findings

1. **Late fusion outperforms early fusion** — Preserving modality-specific characteristics through per-modality PCA before fusion yields substantial AUC improvement over simple concatenation.

2. **Acoustic modality is dominant** — 51% of the predictive signal comes from audio features, consistent with clinical literature on psychomotor retardation in depression.

3. **SMOTE is critical** — Removing SMOTE reduces F1 by 0.084, demonstrating its essential role in handling class imbalance.

4. **SHAP provides clinical trust** — Patient-level explanations highlight interpretable features (speech pauses, negative sentiment) that align with established DSM-5 markers.

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
  title   = {A Multimodal Explainable AI Framework for Depression Detection
             Using Late Fusion of Textual, Acoustic, and Visual Cues},
  author  = {Hossain, Md. Murad and Joy, MD. Foysal Ahammed and Shuvo, Nuruzzaman},
  year    = {2026},
  note    = {Supervisor: Md. Morshed Ali},
  school  = {Uttara University, Department of Computer Science and Engineering},
  url     = {https://github.com/hossmurad/daic-woz-multimodal-depression-detection}
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

- **Supervisor:** Md. Morshed Ali — Lecturer and Coordinator, Department of CSE, Uttara University
- **Dataset:** USC Institute for Creative Technologies (DAIC-WOZ)
- **Open-source tools:** PyTorch, HuggingFace, Scikit-learn, SHAP, Librosa, OpenFace

---

## Contact

- **Md. Murad Hossain** — GitHub: [@hossmurad](https://github.com/hossmurad)
- **Issues:** [Open an issue](https://github.com/hossmurad/daic-woz-multimodal-depression-detection/issues)

---

<p align="center">
  <b>Built with PyTorch, HuggingFace Transformers, and Scikit-learn</b><br>
  <sub>Department of Computer Science and Engineering · Uttara University · 2026</sub>
</p>
