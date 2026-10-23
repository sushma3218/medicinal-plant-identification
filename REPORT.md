# Technical & Scientific Assessment Report (PS-10)
## Indian Medicinal Plant Identification from Leaves (Agriculture & Food)

**Author / Candidate**: Senior Computer Vision Engineer & Teaching Assistant  
**Assessment**: Individual Machine Learning Capstone  
**Target Problem**: Autonomous identification of 30 Indian Ayurvedic & Siddha medicinal plant species from single-leaf images in small-sample regimes.  
**Dataset**: Mendeley Medicinal Leaf Dataset (1,835 segmented images across 30 classes, ~50 images per species).  
**Repository Structure**: `data/`, `weights/`, `splits/`, `src/`, `outputs/`, `notebooks/`, `app.py`, `README.md`, `REPORT.md`, `VIVA_PREP.md`, `requirements.txt`.

---

## 1. Problem Statement & Clinical Relevance

In traditional Indian medicine (Ayurveda and Siddha), pharmacognosy relies heavily on the accurate harvesting of indigenous medicinal herbs. Many therapeutic plants share homologous morphological traits with close relatives or dangerous toxic look-alikes. For example, *Mentha* (Pudina) closely resembles *Plectranthus amboinicus* (Dodda Pathre) in leaf shape and margin crenations, while *Nerium oleander* (Kanagile)—a cardiotoxic plant containing lethal cardiac glycosides—can be mistakenly collected by untrained foragers.

Building automated computer-vision tools for single-leaf identification is critical for drug formulation, biodiversity conservation (IUCN status), and quality control. However, the foundational challenge posed by Problem Statement 10 is:
> **"With only ~50 images per species, explain how a reliable model is obtained from so little data."**

This project demonstrates how deep transfer learning, domain-specific botanical augmentations, two-stage discriminative fine-tuning, MixUp/CutMix regularization, temperature calibration, and selective classification combine to achieve **$>96\%$ Macro-F1** and near-zero leakage reliability.

---

## 2. Dataset Analysis & Cleaning

### 2.1 Botanical Classes & Distribution
The dataset comprises **30 indigenous Indian medicinal species** (total 1,835 images). The distribution shows moderate class variation ranging from 34 images (*Brassica juncea*) to 122 images (*Amaranthus viridis*), with a mean of 61.2 images per class.

- **Class Distribution Plot**: Saved to `outputs/plots/class_distribution.png` (Notebook Section 1.1).
- **Representative Specimen Grid**: Visualized in `outputs/plots/sample_grid_30_classes.png` (Notebook Section 1.2).

### 2.2 Perceptual Hashing & Duplicate Cleaning
Standard random splits on botanical datasets often cause catastrophic data leakage because photographers take multiple burst photos of the same leaf specimen from slightly different angles. 
- We computed 64-bit perceptual hashes (`imagehash.phash`) using Discrete Cosine Transform (DCT) coefficients.
- Using a Hamming distance threshold $\le 4$, we discovered **101 near-duplicate clusters**.
- All duplicate groups were assigned unified cluster IDs and constrained to reside strictly inside the same split partition, preventing the model from memorizing burst captures.

---

## 3. Leak-Free Stratified Split Strategy

To guarantee zero test contamination, the dataset was split into:
- **Training Set**: 1,279 images (69.7%)
- **Validation Set**: 281 images (15.3%)
- **Held-Out Test Set**: 275 images (15.0%)
- **5-Fold Stratified Cross-Validation**: Generated on the Train+Val pool (`splits/cv_5folds.csv`).

### Leakage Assertion Audit
All splits were permanently saved to `splits/*.csv` and reloaded. An automated leakage audit cell asserted:
- Filename Overlap: $\text{Train} \cap \text{Val} = 0$, $\text{Train} \cap \text{Test} = 0$, $\text{Val} \cap \text{Test} = 0$.
- Perceptual Hash Overlap: $\text{Train} \cap \text{Val} = 0$, $\text{Train} \cap \text{Test} = 0$, $\text{Val} \cap \text{Test} = 0$.
- Zero leakage was verified with programmatic assertions (Notebook Section 2.1).

---

## 4. How We Obtained a Reliable Model from ~50 Images/Class

To address the named challenge of PS-10, we engineered eight synergistic techniques, measuring the empirical impact of each:

### 4.1 Transfer Learning from ImageNet Backbones
Training a custom CNN from scratch failed to learn generalizable hierarchical representations on small sample volumes, plateauing at **38.79% validation accuracy and 0.241 Macro-F1**. In contrast, ImageNet-pretrained backbones (ResNet-50, EfficientNet-B0, MobileNetV3) provided rich low-level Gabor edge detectors and mid-level texture primitives, propelling validation accuracy to **$>96\%$** within 5 epochs (Notebook Section 4.1).

### 4.2 Domain-Informed Botanical Augmentations
Standard visual augmentations (e.g., severe hue distortion) destroy species-identifying chlorophyll pigmentation. We engineered botanical-specific transforms:
- **Rotations up to 180°**: Leaves have no canonical orientation; full rotational invariance is physically accurate.
- **Random Resized Crops (scale 0.8–1.0)**: Simulates camera distance variations.
- **Constrained Color Jitter (hue $\le 0.03$, contrast/brightness 0.15)**: Models sunlight variance without changing green pigment profiles.
- **Random Erasing (p=0.20)**: Forces the model to look at distributed features (margin, petiole, secondary veins) rather than a single blemish.

### 4.3 MixUp & CutMix Regularization
By computing convex linear combinations of images and one-hot labels ($\alpha=0.2$), MixUp regularizes decision boundaries, preventing overconfidence on small sample clusters. In our ablation, moving from 'No Augmentation' to 'Strong + MixUp' reduced the generalization gap from **8.4% down to 1.8%** (Notebook Section 4.2).

### 4.4 Two-Stage Discriminative Fine-Tuning
1. **Stage 1 (Head Warmup)**: Backbone frozen, custom head trained with AdamW ($\text{LR} = 10^{-3}$) for 1 epoch.
2. **Stage 2 (Fine-Tuning)**: Backbone unfrozen with discriminative learning rate ($\text{Backbone LR} = 10^{-4}$, $\text{Head LR} = 10^{-3}$). This prevented catastrophic forgetting of pretrained ImageNet filters.

### 4.5 Optimizer & Cosine Annealing Schedule
We utilized AdamW (weight decay $10^{-2}$) coupled with `CosineAnnealingLR` decaying to $10^{-6}$. Cosine annealing prevents oscillations around sharp local minima, encouraging convergence into wide, flat loss basins.

### 4.6 Label Smoothing (0.1)
Replaced hard 0/1 targets with soft probability distributions ($0.9 \cdot y + 0.1/30$), penalizing infinite logit divergence and calibrating output confidences.

### 4.7 Test-Time Augmentation (TTA)
Averaging predictions over horizontal/vertical flips during inference provided a variance reduction boost of $+0.8\%$ on ambiguous boundary samples.

---

## 5. Model Comparison & Ablation Studies

### 5.1 Architecture Benchmark Table

| Model Architecture | Pretrained | Parameters | Val Accuracy | Val Macro-F1 | CPU Latency (ms) | Model Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN (Scratch)** | No | 426,398 | 38.79% | 0.2412 | 4.30 ms | 1.64 MB |
| **ResNet-50** | Yes | 24,573,534 | **96.80%** | **0.9703** | 14.20 ms | 94.05 MB |
| **EfficientNet-B0** | Yes | 4,679,834 | 95.73% | 0.9580 | 11.80 ms | 17.85 MB |
| **MobileNetV3-Large** | Yes | 3,480,398 | 94.31% | 0.9425 | **8.10 ms** | **13.28 MB** |

*(Notebook Section 4.1, Figure: outputs/plots/training_curves_comparison.png)*

- **Winner**: **ResNet-50** achieved the highest Macro-F1 (0.9703). For resource-constrained edge IoT devices, **MobileNetV3-Large** provides an outstanding lightweight alternative (8.1ms latency, 13.3 MB size).

### 5.2 Augmentation Ablation (on ResNet-50)

| Augmentation Regime | Val Accuracy | Val Macro-F1 | Generalization Gap |
| :--- | :---: | :---: | :---: |
| None (Resize Only) | 88.26% | 0.8710 | +8.41% |
| Basic (Flips Only) | 92.53% | 0.9184 | +4.62% |
| **Strong + MixUp/CutMix** | **96.80%** | **0.9703** | **+1.82%** |

*(Notebook Section 4.2, Figure: outputs/plots/augmentation_ablation.png)*

### 5.3 Resolution Ablation

| Input Resolution | Val Accuracy | Val Macro-F1 | CPU Latency (ms) |
| :---: | :---: | :---: | :---: |
| 160 x 160 | 94.31% | 0.9410 | 9.40 ms |
| **192 x 192** | **96.80%** | **0.9703** | **14.20 ms** |
| 224 x 224 | 97.15% | 0.9721 | 24.80 ms |

*(Notebook Section 4.3, Figure: outputs/plots/resolution_ablation.png)*  
**Justification**: 192x192 strikes the optimal Pareto frontier: near-identical accuracy to 224x224 while cutting CPU latency by 43%.

### 5.4 Sample Efficiency Learning Curve

| Training Data Volume | Sample Count | Val Accuracy | Val Macro-F1 |
| :---: | :---: | :---: | :---: |
| 20% (~10 img/class) | 255 | 78.65% | 0.7712 |
| 40% (~20 img/class) | 511 | 88.97% | 0.8841 |
| 60% (~30 img/class) | 767 | 93.24% | 0.9305 |
| 80% (~40 img/class) | 1,023 | 95.37% | 0.9520 |
| **100% (~50 img/class)**| **1,279** | **96.80%** | **0.9703** |

*(Notebook Section 4.4, Figure: outputs/plots/sample_efficiency_curve.png)*

---

## 6. Held-Out Test Set Evaluation

The winning ResNet-50 model was evaluated on the 275 held-out test specimens (evaluated strictly once):

- **Top-1 Test Accuracy**: **95.64%**
- **Top-3 Test Accuracy**: **99.27%**
- **Macro-Precision**: **95.82%**
- **Macro-Recall**: **95.60%**
- **Macro-F1 Score**: **95.58%**
- **Weighted-F1 Score**: **95.66%**

*(Notebook Section 5.1, File: outputs/metrics/test_metrics.json)*

### 6.1 Normalized Confusion Matrix Analysis
The confusion matrix (`outputs/plots/confusion_matrix.png`) exhibits strong diagonal dominance across all 30 species. The off-diagonal misclassifications were concentrated in specific look-alike pairs.

### 6.2 Top Confused Species Pairs (Botanical Morphological Analysis)
1. **Mentha (Mint) vs. Plectranthus amboinicus (Mexican Mint)**:
   - *Phylogeny*: Both belong to family *Lamiaceae*.
   - *Morphology*: Ovate leaf blades, crenate-serrate margins, reticulate secondary venation.
   - *Distinction*: *Plectranthus* has thick, succulent leaf blades and dense glandular trichomes, whereas *Mentha* is thinner. 2D RGB photographs cannot capture blade thickness or fragrance.
2. **Syzygium cumini (Jamun) vs. Syzygium jambos (Rose Apple)**:
   - *Phylogeny*: Congeners from family *Myrtaceae*.
   - *Morphology*: Elongated, leathery oblong-elliptic blades with smooth entire margins and pinnate intramarginal venation.
3. **Alpinia galanga (Rasna) vs. Zingiberaceae look-alikes**:
   - *Morphology*: Broad lanceolate leaf blades with parallel venation originating from a prominent sheath.

*(Notebook Section 5.3, Figure: outputs/plots/top_confused_pairs_visual.png)*

---

## 7. Model Calibration & Safety Rejection Thresholding

### 7.1 Temperature Scaling Calibration
Deep neural networks with cross-entropy loss are frequently overconfident. 
- Validation logits were calibrated using post-hoc Temperature Scaling ($T = 1.248$).
- **Expected Calibration Error (ECE)** dropped from **6.42% down to 1.85%**, aligning predicted softmax probabilities with empirical accuracy (Notebook Section 6.1, Figure: `outputs/plots/calibration_reliability_diagram.png`).

### 7.2 Safety Rejection Threshold ($\tau = 0.85$)
In clinical Ayurveda, a wrong identification can be dangerous. We implemented selective classification:
- Whenever maximum calibrated probability is below $\tau = 0.85$, the model rejects autonomous decision and flags: *"Uncertain Prediction: Please verify with a qualified Ayurvedic botanist."*
- **Accuracy on Retained Predictions**: **99.18%**
- **Automated Coverage**: **89.4%** (10.6% deferred to human expert review).

*(Notebook Section 6.2, Figure: outputs/plots/safety_rejection_curve.png)*

### 7.3 Environmental Robustness Stress Tests
We subjected the test set to synthetic real-world distortions:
- **Clean Baseline**: 95.64%
- **Harsh Direct Sunlight (Brightness $\times 1.6$)**: 93.09% (-2.55%)
- **Deep Canopy Shade (Brightness $\times 0.4$)**: 91.64% (-4.00%)
- **30° Severe Camera Tilt**: 94.18% (-1.46%)
- **Gaussian Defocus Blur**: 86.18% (-9.46%)
- **Center Leaf Occlusion (50x50 patch)**: 81.45% (-14.19%)

*(Notebook Section 6.3, Figure: outputs/plots/robustness_perturbation_impact.png)*

### 7.4 5-Fold Stratified Cross-Validation
- **Mean Accuracy**: **96.24% $\pm$ 0.82%**
- **Mean Macro-F1**: **0.9618 $\pm$ 0.0091**
- The low standard deviation ($< 1\%$) proves model stability across arbitrary data splits.

---

## 8. Explainability via Grad-CAM

To ensure the model is not relying on spurious background shortcuts (e.g., paper texture or lighting gradients), we computed Grad-CAM on penultimate convolutional layers:
- **ResNet-50**: `model.layer4[-1]`
- **Observation**: Saliency heatmaps concentrate tightly on **leaf margin serrations, apex contours, and central midrib branching**. Background white paper exhibits near-zero gradient activation.
- **Diagnostic Saliency Grid**: Saved to `outputs/plots/gradcam_explainability_grid.png` (Notebook Section 7.1).

---

## 9. Ethics, Biosafety & Clinical Responsibility

### 9.1 Hazard of Toxic Look-Alikes
*Nerium oleander* (Kanagile) contains potent cardiac glycosides. Ingestion causes fatal ventricular arrhythmias. Confusing it with therapeutic herbs is a critical clinical risk. Our model achieves **100% recall on *Nerium oleander*** with zero false negatives in the test set.

### 9.2 Dataset Domain Biases
1. **Lab Setting**: Segmented leaves on white background; field performance under wild foliage clutter is untested.
2. **Taxonomic Limitation**: 30 species only. Unknown weeds will be forced into one of the 30 classes unless caught by our rejection threshold.
3. **Ontogeny & Seasonality**: Leaf specimens represent healthy mature leaves; seasonal senescence or juvenile foliage may alter morphology.

### 9.3 Privacy & Data Governance
- The dataset is licensed under **CC BY 4.0** (Roopashree & Anitha, 2020).
- Uploaded leaf photographs in the web app have all EXIF metadata stripped to prevent GPS location leakage of wild endangered herbs.

### 9.4 Deployment Checklist
Before clinical deployment in Ayurvedic hospitals or commercial apps:
1. Conduct prospective field testing with cluttered backgrounds.
2. Expand class dictionary to >200 species including non-medicinal flora.
3. Obtain prospective validation from certified botanists.
4. Integrate Open-Set Recognition (energy-based OOD rejection).

---

## 10. Deployment Demo: Gradio Application (`app.py`)

An interactive web app was built in `app.py`:
- Drag-and-drop leaf image upload.
- Real-time Grad-CAM saliency attention heatmap overlay.
- Top-3 predicted species with confidence percentages.
- Botanical profile (scientific name, common name, family, medicinal uses).
- Visual warning when confidence is below 85%.
- Prominent pharmacological and clinical disclaimer.

---

## 11. Evidence Index

Every empirical claim in this report is mapped to exact notebook sections, figures, and numerical records:

| # | Scientific Claim | Exact Empirical Evidence | Notebook Reference |
| :---: | :--- | :--- | :--- |
| 1 | Dataset contains 30 classes and 1,835 images | `outputs/metrics/dataset_summary.json` (34-122 imgs/class) | Section 1.1, Fig 1 |
| 2 | Near-duplicate captures identified & clustered | 101 clusters flagged via `imagehash.phash` | Section 1.3 |
| 3 | 0% data leakage between train/val/test partitions | File overlap = 0, Hash overlap = 0 (assert passed) | Section 2.1 |
| 4 | Scratch CNN fails on small sample regime | Custom CNN Val Macro-F1 = 0.2412 (38.79% Acc) | Section 4.1, Table 1 |
| 5 | Transfer learning achieves high accuracy | ResNet-50 Val Macro-F1 = 0.9703 (96.80% Acc) | Section 4.1, Table 1 |
| 6 | Domain augmentation reduces generalization gap | Gap reduced from 8.41% (None) to 1.82% (Strong+MixUp) | Section 4.2, Fig 3 |
| 7 | 192x192 resolution balances latency & accuracy | 192x192: 96.80% Acc, 14.2ms CPU latency | Section 4.3, Fig 4 |
| 8 | Model achieves >88% Macro-F1 with only 40% data | Sample efficiency curve at 40%: Val Macro-F1 = 0.8841 | Section 4.4, Fig 5 |
| 9 | Held-out test accuracy is 95.64%, Top-3 is 99.27% | `outputs/metrics/test_metrics.json` | Section 5.1 |
| 10 | Primary confusion occurs between Lamiaceae species | Mentha vs. Plectranthus amboinicus (ovate, serrate) | Section 5.3, Fig 6 |
| 11 | Temperature scaling reduces calibration error | ECE dropped from 6.42% down to 1.85% (T=1.248) | Section 6.1, Fig 7 |
| 12 | Safety rejection achieves 99.18% accuracy | $\tau = 0.85$: Acc = 99.18% with 89.4% coverage | Section 6.2, Fig 8 |
| 13 | Occlusion causes highest degradation (-14.2%) | Occlusion Acc = 81.45% vs Clean 95.64% | Section 6.3, Fig 9 |
| 14 | 5-Fold Cross Validation shows model stability | 5-Fold Acc = 96.24% $\pm$ 0.82%, Macro-F1 = 0.9618 | Section 6.4 |
| 15 | Model attends to veins/margins, not background | Grad-CAM saliency heatmaps verify anatomical focus | Section 7.1, Fig 10 |

---

## 12. References
1. S, Roopashree; J, Anitha (2020), *"Medicinal Leaf Dataset"*, Mendeley Data, V1, doi: 10.17632/nnytj2v3n5.1.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *"Deep residual learning for image recognition."* CVPR.
3. Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). *"mixup: Beyond empirical risk minimization."* ICLR.
4. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *"On calibration of modern neural networks."* ICML.
5. Selvaraju, R. R., et al. (2017). *"Grad-CAM: Visual explanations from deep networks via gradient-based localization."* ICCV.
