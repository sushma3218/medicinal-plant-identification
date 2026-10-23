# Indian Medicinal Plant Leaf Identification (PS-10)
## Deep Learning Computer Vision for Traditional Ayurveda & Siddha Species

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14%2Bcpu-red.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-orange.svg)](https://gradio.app/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end, leak-free, reproducible deep learning system to identify **30 indigenous Indian medicinal plant species** from single-leaf photographs. Designed to solve the small-sample challenge (~50 images per species) using transfer learning, domain-specific botanical augmentations, temperature calibration, selective safety rejection, and Grad-CAM explainability.

---

## 🌿 Key Features & Results

- **Held-Out Test Set Performance**:
  - **Top-1 Accuracy**: **95.64%**
  - **Top-3 Accuracy**: **99.27%**
  - **Macro-F1 Score**: **95.58%**
  - **5-Fold Cross Validation**: **96.24% $\pm$ 0.82%**
- **Safety Rejection**: Below confidence threshold $\tau = 0.85$, ambiguous leaves are deferred to expert botanists, achieving **99.18% accuracy** on accepted predictions.
- **Explainability**: Penultimate-layer **Grad-CAM** saliency maps verify that the network attends to authentic morphological features (margin serrations, venation patterns, blade contours) rather than background shortcuts.
- **Zero Data Leakage**: Perceptual hash clustering (`imagehash.phash`) groups near-duplicate captures into unified partitions; assertion checks confirm 0 file and 0 hash overlap across train, val, and test.
- **Offline Safe**: Pretrained ImageNet weights are cached locally in `weights/` for guaranteed execution in air-gapped lab environments.

---

## 📊 Model Benchmark Summary

| Architecture | Pretrained | Parameters | Val Accuracy | Val Macro-F1 | CPU Latency | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN (Scratch Baseline)** | No | 426,398 | 38.79% | 0.2412 | 4.3 ms | 1.6 MB |
| **ResNet-50 (Production Model)** | Yes | 24,573,534 | **96.80%** | **0.9703** | 14.2 ms | 94.1 MB |
| **EfficientNet-B0** | Yes | 4,679,834 | 95.73% | 0.9580 | 11.8 ms | 17.9 MB |
| **MobileNetV3-Large (Edge IoT)** | Yes | 3,480,398 | 94.31% | 0.9425 | **8.1 ms** | **13.3 MB** |

---

## 📁 Repository Directory Structure

```text
task/
├── data/                               # Dataset directory (30 botanical classes)
│   └── medicinal_leaves/               # Junction/link to Segmented Medicinal Leaf Images
├── weights/                            # Offline-safe pretrained ImageNet weights
│   ├── resnet50_imagenet.pth
│   ├── efficientnet_b0_imagenet.pth
│   └── mobilenet_v3_large_imagenet.pth
├── splits/                             # Reproducible leak-free CSV splits
│   ├── train_split.csv                 # 1,279 training images (70%)
│   ├── val_split.csv                   # 281 validation images (15%)
│   ├── test_split.csv                  # 275 held-out test images (15%)
│   └── cv_5folds.csv                   # 5-fold cross-validation split
├── src/                                # Reusable modular Python package
│   ├── __init__.py
│   ├── data.py                         # Dataset class, transforms, duplicate detection, loaders
│   ├── models.py                       # Custom CNN, ResNet50, EfficientNet, MobileNetV3
│   ├── train.py                        # Two-stage fine-tuning engine, discriminative LRs
│   ├── evaluate.py                     # Metrics, classification report, confusion matrix, calibration
│   └── gradcam.py                      # Explainability and Grad-CAM generation
├── outputs/                            # Persistent generated artifacts & plots
│   ├── plots/                          # Visualizations (confusion matrix, distributions, etc.)
│   ├── metrics/                        # Metrics JSON, per-class reports, ablation tables
│   ├── checkpoints/                    # Best model checkpoints (resnet50_best.pt, etc.)
│   ├── best.pt                         # Production PyTorch model checkpoint
│   └── class_names.json                # List of 30 botanical species classes
├── notebooks/
│   └── medicinal_plant_identification.ipynb  # Executable Jupyter notebook with viva notes
├── app.py                              # Interactive Gradio demo with Grad-CAM and safety warning
├── README.md                           # Quickstart and project summary
├── REPORT.md                           # Comprehensive technical report with Evidence Index
├── VIVA_PREP.md                        # Cell-by-cell walkthrough & model answers for viva
└── requirements.txt                    # Pinned package versions
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
# Verify Python version (3.10+ required)
python --version

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### 2. Run the Interactive Web Demo
Launch the Gradio application to test leaf identification with live Grad-CAM visual attention overlays:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`.

### 3. Run the Jupyter Notebook
Open and run the complete academic notebook top-to-bottom:
```bash
jupyter notebook notebooks/medicinal_plant_identification.ipynb
```

---

## ⚕️ Ethical, Biosafety & Legal Notice
- **Clinical Decision Support**: This system is designed as an educational and clinical decision-support tool for Ayurvedic physicians and pharmacognosists. It is **not** an autonomous medical device. Misidentifying toxic look-alikes (e.g. *Nerium oleander*) can cause lethal cardiotoxicity.
- **Privacy**: Leaf images do not contain personal data. The deployment app strips all EXIF GPS metadata on upload to protect user privacy and wild plant habitats.
- **Dataset Citation**: Roopashree S, Anitha J (2020), *"Medicinal Leaf Dataset"*, Mendeley Data, V1, doi: 10.17632/nnytj2v3n5.1 (CC BY 4.0).
