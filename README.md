# DAIC-WOZ Multimodal Depression Detection

A Multimodal Explainable AI Framework for Depression Detection Integrating Textual, Acoustic, and Visual Cues

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()
[![SHAP](https://img.shields.io/badge/explainability-SHAP-purple.svg)]()

---

## Authors

<table>
  <tr>
    <td align="center" width="25%">
      <b>Md. Murad Hossain</b><br>
      <sub>Author</sub>
    </td>
    <td align="center" width="25%">
      <b>MD. Foysal Ahammed Joy</b><br>
      <sub>Author</sub>
    </td>
    <td align="center" width="25%">
      <b>Nuruzzaman Shuvo</b><br>
      <sub>Author</sub>
    </td>
    <td align="center" width="25%">
      <b>Md. Morshed Ali</b><br>
      <sub>Supervisor</sub><br>
      <sub>Lecturer and Coordinator</sub>
    </td>
  </tr>
</table>

<p align="center">
  <b>Department of Computer Science and Engineering</b><br>
  <b>Uttara University, Dhaka, Bangladesh</b>
</p>

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
