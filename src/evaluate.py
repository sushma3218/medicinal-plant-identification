"""
Evaluation and Safety Assessment Module for Indian Medicinal Plant Leaf Identification (PS-10)
Includes:
- Standard & Top-k Classification Metrics
- Normalized Confusion Matrix Plotting
- Botanical Confused Pair Visual Analysis
- Model Calibration (ECE & Temperature Scaling)
- Safety Rejection Thresholding (Accuracy vs. Coverage)
- Robustness Evaluation under Synthetic Environmental Perturbations
"""

import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import CLASS_NAMES, SPECIES_METADATA


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    topk: int = 3
) -> Dict:
    """Computes comprehensive metrics on the held-out test partition."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    all_fps = []
    topk_correct = 0
    total = 0

    for imgs, labels, fps in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)

        # Top-k accuracy
        _, topk_preds = torch.topk(probs, k=topk, dim=1)
        topk_correct += (topk_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_fps.extend(fps)
        total += imgs.size(0)

    acc = accuracy_score(all_targets, all_preds)
    topk_acc = topk_correct / max(1, total)
    macro_p = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_r = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    cls_report = classification_report(
        all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(CLASS_NAMES))))

    results = {
        "accuracy": acc,
        "top3_accuracy": topk_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "all_preds": all_preds,
        "all_targets": all_targets,
        "all_probs": all_probs,
        "all_fps": all_fps
    }
    return results


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str):
    """Generates and saves a publication-quality normalized confusion matrix."""
    plt.figure(figsize=(18, 15))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    # Shorten class names for readability
    short_names = [name.split("(")[0].strip() for name in class_names]

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=short_names,
        yticklabels=short_names,
        cbar_kws={'label': 'Normalized Prediction Fraction'},
        linewidths=0.5
    )
    plt.title("Normalized Confusion Matrix - Indian Medicinal Plant Species (PS-10)", fontsize=16, pad=15)
    plt.xlabel("Predicted Class", fontsize=13, labelpad=10)
    plt.ylabel("True Class", fontsize=13, labelpad=10)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def find_top_confused_pairs(cm: np.ndarray, class_names: List[str], top_n: int = 10) -> List[Dict]:
    """Identifies the top N most confused species pairs with botanical morphological analysis."""
    confused_pairs = []
    num_classes = len(class_names)
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": class_names[i],
                    "predicted_class": class_names[j],
                    "count": int(cm[i, j]),
                    "true_idx": i,
                    "pred_idx": j
                })
    confused_pairs = sorted(confused_pairs, key=lambda x: x["count"], reverse=True)[:top_n]
    return confused_pairs


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Expected Calibration Error (ECE) and reliability diagram statistics.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_sizes = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        bin_sizes.append(np.sum(in_bin))

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)

    return ece, np.array(bin_accs), np.array(bin_confs), np.array(bin_sizes)


class TemperatureScaler(nn.Module):
    """
    Learns a single scalar temperature T on validation logits to calibrate probabilities.
    Softmax(z / T) aligns confidence with true empirical accuracy without altering accuracy.
    """
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 100):
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self.temperature.item()


def compute_rejection_curve(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> Dict:
    """
    Evaluates safety rejection: below confidence threshold tau, defer to human expert.
    Computes accuracy on retained predictions and fraction of coverage.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.99, 50)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    coverages = []
    accuracies = []
    rejected_pcts = []

    for tau in thresholds:
        accepted = confidences >= tau
        cov = np.mean(accepted)
        coverages.append(cov)
        rejected_pcts.append(1.0 - cov)

        if np.sum(accepted) > 0:
            acc = accuracy_score(targets[accepted], predictions[accepted])
            accuracies.append(acc)
        else:
            accuracies.append(1.0)

    return {
        "thresholds": thresholds.tolist(),
        "coverages": coverages,
        "accuracies": accuracies,
        "rejected_pcts": rejected_pcts
    }


def evaluate_robustness(
    model: nn.Module,
    test_df: pd.DataFrame,
    device: torch.device,
    img_size: int = 224
) -> Dict[str, float]:
    """
    Tests model vulnerability against real-world environmental image corruptions:
    1. Clean Test baseline
    2. Gaussian Blur (simulating camera defocus)
    3. Severe Brightness Alterations (harsh direct sunlight vs deep shade)
    4. Heavy Rotation (30 deg tilt)
    5. Partial Occlusion (simulating insect damage or overlapping leaf)
    """
    model.eval()
    corruptions = ["clean", "gaussian_blur", "harsh_sunlight", "deep_shade", "rotation_30", "occlusion"]
    results = {}

    import torchvision.transforms as T
    base_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for c_name in corruptions:
        correct = 0
        total = 0
        for _, row in test_df.iterrows():
            fp = row["filepath"]
            label = row["label"]
            with Image.open(fp) as img:
                img = img.convert("RGB").resize((img_size, img_size))

                if c_name == "gaussian_blur":
                    img = img.filter(ImageFilter.GaussianBlur(radius=3.0))
                elif c_name == "harsh_sunlight":
                    img = ImageEnhance.Brightness(img).enhance(1.6)
                elif c_name == "deep_shade":
                    img = ImageEnhance.Brightness(img).enhance(0.4)
                elif c_name == "rotation_30":
                    img = img.rotate(30, fillcolor=(0, 0, 0))
                elif c_name == "occlusion":
                    arr = np.array(img).copy()
                    # Occlude a 50x50 center patch
                    h, w, _ = arr.shape
                    arr[h//3: 2*h//3, w//3: 2*w//3, :] = 0
                    img = Image.fromarray(arr)

                tensor = T.ToTensor()(img)
                tensor = base_norm(tensor).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tensor)
                    pred = torch.argmax(output, dim=1).item()
                    if pred == label:
                        correct += 1
                    total += 1
        results[c_name] = correct / max(1, total)

    return results
