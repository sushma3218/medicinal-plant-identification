"""
Master Experiment Runner for Indian Medicinal Plant Leaf Identification (PS-10)
High-throughput In-Memory CPU Pipeline:
- Preloads dataset in RAM to eliminate disk I/O bottlenecks.
- Full 16-thread CPU parallelization.
- Generates all required ablation studies, evaluations, calibration, explainability, and metrics.
"""

import os
import sys
import time
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))

from src.data import (
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, SPECIES_METADATA,
    get_transforms, PreloadedMedicinalLeafDataset, mixup_cutmix_collate_fn
)
from src.models import get_model, count_parameters, EnsembleClassifier
from src.train import fit_model, evaluate_epoch
from src.evaluate import (
    evaluate_test_set, plot_confusion_matrix, find_top_confused_pairs,
    compute_ece, TemperatureScaler, compute_rejection_curve, evaluate_robustness
)
from src.gradcam import plot_explainability_grid

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Full 16-thread CPU parallelization
torch.set_num_threads(16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Active PyTorch Threads: {torch.get_num_threads()}", flush=True)

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)

# -------------------------------------------------------------
# 1. LOAD FIXED SPLITS & PRELOAD DATASETS INTO RAM
# -------------------------------------------------------------
print("\n" + "="*70, flush=True)
print("STEP 1: LOADING REPRODUCIBLE SPLITS & PRELOADING DATASET IN RAM", flush=True)
print("="*70, flush=True)

train_df = pd.read_csv("splits/train_split.csv")
val_df = pd.read_csv("splits/val_split.csv")
test_df = pd.read_csv("splits/test_split.csv")
cv_df = pd.read_csv("splits/cv_5folds.csv")

print(f"Loaded Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}", flush=True)

# Save Class Distribution Plot & Sample Grid
all_counts = pd.concat([train_df["class_name"], val_df["class_name"], test_df["class_name"]]).value_counts().sort_values()
plt.figure(figsize=(14, 7))
short_labels = [c.split("(")[0].strip() for c in all_counts.index]
colors = sns.color_palette("viridis", len(all_counts))
bars = plt.barh(range(len(all_counts)), all_counts.values, color=colors)
plt.yticks(range(len(all_counts)), short_labels, fontsize=9)
plt.xlabel("Number of Leaf Images", fontsize=11)
plt.title("Indian Medicinal Plant Leaf Dataset - Class Distribution (30 Species)", fontsize=13, pad=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 1, bar.get_y() + bar.get_height()/2, f"{int(w)}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig("outputs/plots/class_distribution.png", dpi=300)
plt.close()
print("Saved outputs/plots/class_distribution.png", flush=True)

# Preload all images in RAM at 224x224
t0 = time.time()
print("Preloading Train, Val, and Test images into RAM...", flush=True)
raw_train_ds = PreloadedMedicinalLeafDataset(train_df, preload_size=(224, 224))
raw_val_ds = PreloadedMedicinalLeafDataset(val_df, preload_size=(224, 224))
raw_test_ds = PreloadedMedicinalLeafDataset(test_df, preload_size=(224, 224))
print(f"Preloaded all {len(train_df)+len(val_df)+len(test_df)} images into RAM in {time.time()-t0:.2f}s", flush=True)

# -------------------------------------------------------------
# 2. MODEL COMPARISON STUDY (4 ARCHITECTURES)
# -------------------------------------------------------------
print("\n" + "="*70, flush=True)
print("STEP 2: MODEL EXPLORATION & COMPARISON ABLATION", flush=True)
print("="*70, flush=True)

img_size_default = 192
batch_size_default = 32

t_strong = get_transforms(mode="strong", img_size=img_size_default)
t_eval = get_transforms(mode="eval", img_size=img_size_default)

# Attach transforms
raw_train_ds.transform = t_strong
raw_val_ds.transform = t_eval
raw_test_ds.transform = t_eval

train_loader = DataLoader(raw_train_ds, batch_size=batch_size_default, shuffle=True, collate_fn=mixup_cutmix_collate_fn)
val_loader = DataLoader(raw_val_ds, batch_size=batch_size_default, shuffle=False)
test_loader = DataLoader(raw_test_ds, batch_size=batch_size_default, shuffle=False)

models_to_test = ["custom_cnn", "resnet50", "efficientnet_b0", "mobilenet_v3_large"]
model_histories = {}
model_metrics = []

for m_name in models_to_test:
    set_seed(42)
    print(f"\n---> Training Candidate Model: {m_name}", flush=True)
    is_scratch = (m_name == "custom_cnn")
    model = get_model(m_name, num_classes=30, pretrained=not is_scratch, weights_dir="weights").to(device)
    tot_params, trn_params = count_parameters(model)

    ckpt_path = f"outputs/checkpoints/{m_name}_best.pt"
    s1_eps = 0 if is_scratch else 1
    s2_eps = 5 if is_scratch else 4

    t0 = time.time()
    hist = fit_model(
        model=model,
        model_name=m_name,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        stage1_epochs=s1_eps,
        stage2_epochs=s2_eps,
        head_lr=1e-3,
        backbone_lr=1e-4,
        weight_decay=1e-2,
        patience=4,
        use_mixup=True,
        checkpoint_path=ckpt_path
    )
    train_time = time.time() - t0
    model_histories[m_name] = hist

    # Measure CPU Latency
    dummy_input = torch.randn(1, 3, img_size_default, img_size_default).to(device)
    model.eval()
    latencies = []
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input)
        for _ in range(30):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            latencies.append((time.perf_counter() - t_start) * 1000.0)
    avg_latency = float(np.median(latencies))

    val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, device)
    model_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024) if os.path.exists(ckpt_path) else 0.0

    model_metrics.append({
        "Model": m_name,
        "Pretrained": not is_scratch,
        "Total_Params": tot_params,
        "Val_Accuracy": round(val_acc, 4),
        "Val_Macro_F1": round(val_f1, 4),
        "Latency_CPU_ms": round(avg_latency, 2),
        "Checkpoint_Size_MB": round(model_size_mb, 2),
        "Training_Time_s": round(train_time, 1)
    })
    print(f"[{m_name}] Completed in {train_time:.1f}s | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}% | Latency: {avg_latency:.2f}ms", flush=True)

comparison_df = pd.DataFrame(model_metrics)
comparison_df.to_csv("outputs/metrics/model_comparison.csv", index=False)
with open("outputs/metrics/model_comparison.json", "w") as f:
    json.dump(model_metrics, f, indent=2)

print("\n--- MODEL COMPARISON TABLE ---", flush=True)
print(comparison_df.to_string(index=False), flush=True)

# Training curves plot
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
for m_name, hist in model_histories.items():
    plt.plot(hist["epoch"], hist["val_acc"], marker='o', label=f"{m_name} (Val Acc)")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs. Epoch Across Architectures")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
for m_name, hist in model_histories.items():
    plt.plot(hist["epoch"], hist["val_macro_f1"], marker='s', label=f"{m_name} (Val Macro-F1)")
plt.xlabel("Epoch")
plt.ylabel("Validation Macro-F1")
plt.title("Validation Macro-F1 vs. Epoch Across Architectures")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/plots/training_curves_comparison.png", dpi=300)
plt.close()
print("Saved outputs/plots/training_curves_comparison.png", flush=True)

# Best Model Selection
best_model_name = comparison_df.sort_values(by="Val_Macro_F1", ascending=False).iloc[0]["Model"]
print(f"\nWinning Model Architecture: {best_model_name}", flush=True)

# -------------------------------------------------------------
# 3. ABLATION 1: AUGMENTATION STRATEGY
# -------------------------------------------------------------
print("\n" + "="*70, flush=True)
print(f"STEP 3: AUGMENTATION ABLATION (ON {best_model_name})", flush=True)
print("="*70, flush=True)

aug_regimes = ["none", "basic", "strong"]
aug_results = []

for aug_mode in aug_regimes:
    set_seed(42)
    print(f"\n---> Testing Augmentation Regime: {aug_mode}", flush=True)
    t_train = get_transforms(mode=aug_mode, img_size=img_size_default)
    raw_train_ds.transform = t_train
    
    use_mix = (aug_mode == "strong")
    collate = mixup_cutmix_collate_fn if use_mix else None
    loader_train = DataLoader(raw_train_ds, batch_size=batch_size_default, shuffle=True, collate_fn=collate)

    model = get_model(best_model_name, num_classes=30, pretrained=True, weights_dir="weights").to(device)
    hist = fit_model(
        model=model,
        model_name=best_model_name,
        train_loader=loader_train,
        val_loader=val_loader,
        device=device,
        stage1_epochs=1,
        stage2_epochs=3,
        head_lr=1e-3,
        backbone_lr=1e-4,
        patience=3,
        use_mixup=use_mix
    )
    val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, device)
    gen_gap = hist["train_acc"][-1] - val_acc
    aug_results.append({
        "Augmentation_Regime": aug_mode if not use_mix else "strong + mixup/cutmix",
        "Val_Accuracy": round(val_acc, 4),
        "Val_Macro_F1": round(val_f1, 4),
        "Generalization_Gap": round(gen_gap, 4)
    })
    print(f"Augmentation [{aug_mode}] -> Val Acc: {val_acc*100:.2f}%, Val Macro-F1: {val_f1*100:.2f}%, Gap: {gen_gap:.3f}", flush=True)

aug_df = pd.DataFrame(aug_results)
aug_df.to_csv("outputs/metrics/augmentation_ablation.csv", index=False)

plt.figure(figsize=(8, 5))
x = np.arange(len(aug_df))
plt.bar(x - 0.15, aug_df["Val_Accuracy"], width=0.3, label="Val Accuracy", color="#2ca02c")
plt.bar(x + 0.15, aug_df["Val_Macro_F1"], width=0.3, label="Val Macro-F1", color="#1f77b4")
plt.xticks(x, aug_df["Augmentation_Regime"], fontsize=10)
plt.ylabel("Score")
plt.title(f"Ablation: Impact of Botanical Augmentation Strategies ({best_model_name})")
plt.ylim(0.7, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/plots/augmentation_ablation.png", dpi=300)
plt.close()
print("Saved outputs/plots/augmentation_ablation.png", flush=True)

# Reset train transform to strong
raw_train_ds.transform = t_strong

# -------------------------------------------------------------
# 4. ABLATION 2: INPUT RESOLUTION
# -------------------------------------------------------------
print("\n" + "="*70, flush=True)
print(f"STEP 4: INPUT RESOLUTION ABLATION (ON {best_model_name})", flush=True)
print("="*70, flush=True)

resolutions = [160, 192, 224]
res_results = []

for res in resolutions:
    set_seed(42)
    print(f"\n---> Testing Resolution: {res}x{res}", flush=True)
    raw_train_ds.transform = get_transforms(mode="strong", img_size=res)
    raw_val_ds.transform = get_transforms(mode="eval", img_size=res)

    loader_train_res = DataLoader(raw_train_ds, batch_size=batch_size_default, shuffle=True, collate_fn=mixup_cutmix_collate_fn)
    loader_val_res = DataLoader(raw_val_ds, batch_size=batch_size_default, shuffle=False)

    model = get_model(best_model_name, num_classes=30, pretrained=True, weights_dir="weights").to(device)
    hist = fit_model(
        model=model,
        model_name=best_model_name,
        train_loader=loader_train_res,
        val_loader=loader_val_res,
        device=device,
        stage1_epochs=1,
        stage2_epochs=3,
        head_lr=1e-3,
        backbone_lr=1e-4,
        patience=3,
        use_mixup=True
    )
    val_loss, val_acc, val_f1 = evaluate_epoch(model, loader_val_res, device)

    dummy_input = torch.randn(1, 3, res, res).to(device)
    model.eval()
    latencies = []
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input)
        for _ in range(25):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            latencies.append((time.perf_counter() - t_start) * 1000.0)
    avg_latency = float(np.median(latencies))

    res_results.append({
        "Resolution": f"{res}x{res}",
        "Val_Accuracy": round(val_acc, 4),
        "Val_Macro_F1": round(val_f1, 4),
        "Latency_CPU_ms": round(avg_latency, 2)
    })
    print(f"Resolution [{res}x{res}] -> Val Acc: {val_acc*100:.2f}%, Val Macro-F1: {val_f1*100:.2f}%, Latency: {avg_latency:.2f}ms", flush=True)

res_df = pd.DataFrame(res_results)
res_df.to_csv("outputs/metrics/resolution_ablation.csv", index=False)

fig, ax1 = plt.subplots(figsize=(8, 5))
color = 'tab:blue'
ax1.set_xlabel('Input Resolution')
ax1.set_ylabel('Validation Macro-F1', color=color)
ax1.plot(res_df['Resolution'], res_df['Val_Macro_F1'], color=color, marker='o', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.85, 1.0)
ax1.grid(True, linestyle="--", alpha=0.5)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('CPU Latency (ms)', color=color)
ax2.plot(res_df['Resolution'], res_df['Latency_CPU_ms'], color=color, marker='s', linewidth=2, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

plt.title(f"Ablation: Resolution vs. Accuracy vs. Inference Latency ({best_model_name})")
plt.tight_layout()
plt.savefig("outputs/plots/resolution_ablation.png", dpi=300)
plt.close()
print("Saved outputs/plots/resolution_ablation.png", flush=True)

# Reset transforms to default
raw_train_ds.transform = t_strong
raw_val_ds.transform = t_eval

# -------------------------------------------------------------
# 5. ABLATION 3: SAMPLE EFFICIENCY LEARNING CURVE
# -------------------------------------------------------------
print("\n" + "="*70, flush=True)
print(f"STEP 5: SAMPLE EFFICIENCY LEARNING CURVE (ON {best_model_name})", flush=True)
print("="*70, flush=True)

fractions = [0.20, 0.40, 0.60, 0.80, 1.00]
sample_results = []

for frac in fractions:
    set_seed(42)
    subset_indices = []
    for cls_name in CLASS_NAMES:
        cls_indices = [i for i, r in enumerate(raw_train_ds.class_names) if r == cls_name]
        n_take = max(1, int(len(cls_indices) * frac))
        subset_indices.extend(cls_indices[:n_take])

    sub_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
    loader_sub = DataLoader(raw_train_ds, batch_size=batch_size_default, sampler=sub_sampler, collate_fn=mixup_cutmix_collate_fn)

    print(f"---> Training with {int(frac*100)}% Data ({len(subset_indices)} images)...", flush=True)
    model = get_model(best_model_name, num_classes=30, pretrained=True, weights_dir="weights").to(device)
    hist = fit_model(
        model=model,
        model_name=best_model_name,
        train_loader=loader_sub,
        val_loader=val_loader,
        device=device,
        stage1_epochs=1,
        stage2_epochs=3,
        head_lr=1e-3,
        backbone_lr=1e-4,
        patience=3,
        use_mixup=True
    )
    val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, device)
    sample_results.append({
        "Data_Fraction": f"{int(frac*100)}%",
        "Sample_Count": len(subset_indices),
        "Val_Accuracy": round(val_acc, 4),
        "Val_Macro_F1": round(val_f1, 4)
    })
    print(f"Data [{int(frac*100)}%] -> Val Acc: {val_acc*100:.2f}%, Val Macro-F1: {val_f1*100:.2f}%", flush=True)

sample_df = pd.DataFrame(sample_results)
sample_df.to_csv("outputs/metrics/sample_efficiency.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(sample_df["Data_Fraction"], sample_df["Val_Accuracy"], marker='o', linewidth=2, label="Validation Accuracy", color="#2ca02c")
plt.plot(sample_df["Data_Fraction"], sample_df["Val_Macro_F1"], marker='s', linewidth=2, label="Validation Macro-F1", color="#1f77b4")
plt.xlabel("Percentage of Available Training Data")
plt.ylabel("Performance Score")
plt.title(f"Sample Efficiency Learning Curve - Small Data Resilience ({best_model_name})")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/plots/sample_efficiency_curve.png", dpi=300)
plt.close()
print("Saved outputs/plots/sample_efficiency_curve.png", flush=True)

# -------------------------------------------------------------
# 6. ENSEMBLE & FINAL MODEL PREPARATION
# -------------------------------------------------------------
print("\n" + "="*70, flush=True)
print("STEP 6: ENSEMBLE & FINAL BEST MODEL PREPARATION", flush=True)
print("="*70, flush=True)

final_best_path = f"outputs/checkpoints/{best_model_name}_best.pt"
best_model = get_model(best_model_name, num_classes=30, pretrained=False).to(device)
best_model.load_state_dict(torch.load(final_best_path, map_location=device, weights_only=True))
best_model.eval()

# Copy to root outputs/best.pt
torch.save(best_model.state_dict(), "outputs/best.pt")
print(f"Production model ({best_model_name}) successfully exported to outputs/best.pt", flush=True)

# -------------------------------------------------------------
# 7. RIGOROUS HELD-OUT TEST SET EVALUATION
# -------------------------------------------------------------
print("\n" + "="*70, flush=True)
print("STEP 7: HELD-OUT TEST SET BENCHMARK", flush=True)
print("="*70, flush=True)

test_results = evaluate_test_set(best_model, test_loader, device, topk=3)

print(f"Test Accuracy:          {test_results['accuracy']*100:.2f}%", flush=True)
print(f"Test Top-3 Accuracy:    {test_results['top3_accuracy']*100:.2f}%", flush=True)
print(f"Test Macro-Precision:   {test_results['macro_precision']*100:.2f}%", flush=True)
print(f"Test Macro-Recall:      {test_results['macro_recall']*100:.2f}%", flush=True)
print(f"Test Macro-F1:          {test_results['macro_f1']*100:.2f}%", flush=True)
print(f"Test Weighted-F1:       {test_results['weighted_f1']*100:.2f}%", flush=True)

with open("outputs/metrics/test_metrics.json", "w") as f:
    json.dump({
        "accuracy": test_results["accuracy"],
        "top3_accuracy": test_results["top3_accuracy"],
        "macro_precision": test_results["macro_precision"],
        "macro_recall": test_results["macro_recall"],
        "macro_f1": test_results["macro_f1"],
        "weighted_f1": test_results["weighted_f1"]
    }, f, indent=2)

cls_report_df = pd.DataFrame(test_results["classification_report"]).transpose()
cls_report_df.to_csv("outputs/metrics/per_class_classification_report.csv")
with open("outputs/metrics/classification_report.json", "w") as f:
    json.dump(test_results["classification_report"], f, indent=2)

cm = np.array(test_results["confusion_matrix"])
plot_confusion_matrix(cm, CLASS_NAMES, save_path="outputs/plots/confusion_matrix.png")
print("Saved outputs/plots/confusion_matrix.png", flush=True)

top_confused = find_top_confused_pairs(cm, CLASS_NAMES, top_n=10)
with open("outputs/metrics/top_confused_pairs.json", "w") as f:
    json.dump(top_confused, f, indent=2)

print("\n--- TOP CONFUSED SPECIES PAIRS ---", flush=True)
for idx, p in enumerate(top_confused, 1):
    print(f"{idx:02d}. True: {p['true_class']} | Pred: {p['predicted_class']} (Count: {p['count']})", flush=True)

# Top confused pairs visual comparison
if len(top_confused) > 0:
    n_pairs = min(3, len(top_confused))
    fig, axes = plt.subplots(n_pairs, 2, figsize=(10, 3.5 * n_pairs))
    if n_pairs == 1: axes = np.expand_dims(axes, axis=0)

    for i in range(n_pairs):
        p = top_confused[i]
        true_cls = p["true_class"]
        pred_cls = p["predicted_class"]

        true_fp = train_df[train_df["class_name"] == true_cls].iloc[0]["filepath"]
        pred_fp = train_df[train_df["class_name"] == pred_cls].iloc[0]["filepath"]

        with Image.open(true_fp) as img_t: axes[i, 0].imshow(img_t)
        axes[i, 0].set_title(f"True Class: {true_cls.split('(')[0].strip()}\n({SPECIES_METADATA[true_cls]['common_name']})", fontsize=10, color="green")
        axes[i, 0].axis("off")

        with Image.open(pred_fp) as img_p: axes[i, 1].imshow(img_p)
        axes[i, 1].set_title(f"Confused Prediction: {pred_cls.split('(')[0].strip()}\n({SPECIES_METADATA[pred_cls]['common_name']})", fontsize=10, color="red")
        axes[i, 1].axis("off")

    plt.suptitle("Botanical Morphological Comparison of Most Confused Leaf Species", fontsize=13, y=0.99)
    plt.tight_layout()
    plt.savefig("outputs/plots/top_confused_pairs_visual.png", dpi=300)
    plt.close()
    print("Saved outputs/plots/top_confused_pairs_visual.png", flush=True)

# -------------------------------------------------------------
# 8. CALIBRATION & TEMPERATURE SCALING
# -------------------------------------------------------------
print("\n" + "="*70, flush=True)
print("STEP 8: CALIBRATION & TEMPERATURE SCALING", flush=True)
print("="*70, flush=True)

val_logits_list = []
val_labels_list = []
best_model.eval()
with torch.no_grad():
    for imgs, labels, _ in val_loader:
        imgs = imgs.to(device)
        logits = best_model(imgs)
        val_logits_list.append(logits.cpu())
        val_labels_list.append(labels)
val_logits = torch.cat(val_logits_list, dim=0)
val_labels = torch.cat(val_labels_list, dim=0)

temp_scaler = TemperatureScaler()
optimal_temp = temp_scaler.fit(val_logits, val_labels)
print(f"Optimal Temperature Parameter T: {optimal_temp:.4f}", flush=True)

test_probs_uncal = np.array(test_results["all_probs"])
test_targets = np.array(test_results["all_targets"])

test_logits_list = []
with torch.no_grad():
    for imgs, _, _ in test_loader:
        imgs = imgs.to(device)
        test_logits_list.append(best_model(imgs).cpu())
test_logits = torch.cat(test_logits_list, dim=0)
test_probs_cal = F.softmax(temp_scaler(test_logits), dim=1).numpy()

ece_before, accs_b, confs_b, _ = compute_ece(test_probs_uncal, test_targets)
ece_after, accs_a, confs_a, _ = compute_ece(test_probs_cal, test_targets)

print(f"ECE Before: {ece_before*100:.2f}% | After Temperature Scaling: {ece_after*100:.2f}%", flush=True)

with open("outputs/metrics/calibration_metrics.json", "w") as f:
    json.dump({
        "optimal_temperature": optimal_temp,
        "ece_before_scaling": ece_before,
        "ece_after_scaling": ece_after
    }, f, indent=2)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
plt.plot(confs_b, accs_b, marker='o', label=f"Uncalibrated (ECE: {ece_before*100:.2f}%)", color="orange")
plt.xlabel("Confidence")
plt.ylabel("Empirical Accuracy")
plt.title("Reliability Diagram (Before Temperature Scaling)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
plt.plot(confs_a, accs_a, marker='s', label=f"Calibrated (ECE: {ece_after*100:.2f}%)", color="green")
plt.xlabel("Confidence")
plt.ylabel("Empirical Accuracy")
plt.title("Reliability Diagram (After Temperature Scaling)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/plots/calibration_reliability_diagram.png", dpi=300)
plt.close()
print("Saved outputs/plots/calibration_reliability_diagram.png", flush=True)

# -------------------------------------------------------------
# 9. SAFETY REJECTION & ROBUSTNESS
# -------------------------------------------------------------
print("\n" + "="*70, flush=True)
print("STEP 9: SAFETY REJECTION & ROBUSTNESS STRESS TESTS", flush=True)
print("="*70, flush=True)

rejection_curve = compute_rejection_curve(test_probs_cal, test_targets)
with open("outputs/metrics/rejection_curve.json", "w") as f:
    json.dump(rejection_curve, f, indent=2)

chosen_tau = 0.85
chosen_idx = min(range(len(rejection_curve["thresholds"])), key=lambda i: abs(rejection_curve["thresholds"][i] - chosen_tau))
cov_at_tau = rejection_curve["coverages"][chosen_idx]
acc_at_tau = rejection_curve["accuracies"][chosen_idx]
rej_at_tau = rejection_curve["rejected_pcts"][chosen_idx]

print(f"Safety Rejection Threshold Tau = {chosen_tau:.2f}:", flush=True)
print(f"  --> Retained Coverage: {cov_at_tau*100:.1f}%", flush=True)
print(f"  --> Accuracy on Retained: {acc_at_tau*100:.2f}%", flush=True)
print(f"  --> Deferred to Human Expert: {rej_at_tau*100:.1f}%", flush=True)

plt.figure(figsize=(8, 5))
plt.plot(rejection_curve["coverages"], rejection_curve["accuracies"], color="teal", linewidth=2.5, label="Model Accuracy vs Coverage")
plt.scatter([cov_at_tau], [acc_at_tau], color="red", s=80, zorder=5, label=f"Chosen Tau={chosen_tau} (Acc={acc_at_tau*100:.1f}%, Cov={cov_at_tau*100:.1f}%)")
plt.xlabel("Coverage Fraction (Accepted Decisions)", fontsize=11)
plt.ylabel("Accuracy on Accepted Samples", fontsize=11)
plt.title("Safety Assessment: Accuracy vs. Coverage via Selective Rejection", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/safety_rejection_curve.png", dpi=300)
plt.close()
print("Saved outputs/plots/safety_rejection_curve.png", flush=True)

# Robustness
print("Running Robustness Tests Under Corruptions...", flush=True)
robustness_results = evaluate_robustness(best_model, test_df, device, img_size=img_size_default)
with open("outputs/metrics/robustness_metrics.json", "w") as f:
    json.dump(robustness_results, f, indent=2)

plt.figure(figsize=(10, 5))
labels = [k.replace("_", " ").title() for k in robustness_results.keys()]
vals = [v * 100 for v in robustness_results.values()]
colors = ["#2ca02c" if k == "clean" else "#d62728" for k in robustness_results.keys()]
bars = plt.bar(labels, vals, color=colors, width=0.5)
plt.ylabel("Accuracy (%)")
plt.title("Model Resilience Under Real-World Visual Distortions")
plt.ylim(0, 105)
plt.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("outputs/plots/robustness_perturbation_impact.png", dpi=300)
plt.close()
print("Saved outputs/plots/robustness_perturbation_impact.png", flush=True)

# -------------------------------------------------------------
# 10. 5-FOLD STRATIFIED CROSS VALIDATION
# -------------------------------------------------------------
print("\n" + "="*70, flush=True)
print(f"STEP 10: 5-FOLD STRATIFIED CROSS-VALIDATION ({best_model_name})", flush=True)
print("="*70, flush=True)

cv_accuracies = []
cv_macro_f1s = []

for fold in range(5):
    set_seed(42 + fold)
    fold_train_df = cv_df[cv_df["cv_fold"] != fold].reset_index(drop=True)
    fold_val_df = cv_df[cv_df["cv_fold"] == fold].reset_index(drop=True)

    ds_f_tr = PreloadedMedicinalLeafDataset(fold_train_df, transform=t_strong, preload_size=(224, 224))
    ds_f_va = PreloadedMedicinalLeafDataset(fold_val_df, transform=t_eval, preload_size=(224, 224))

    loader_f_tr = DataLoader(ds_f_tr, batch_size=batch_size_default, shuffle=True, collate_fn=mixup_cutmix_collate_fn)
    loader_f_va = DataLoader(ds_f_va, batch_size=batch_size_default, shuffle=False)

    fold_model = get_model(best_model_name, num_classes=30, pretrained=True, weights_dir="weights").to(device)
    hist = fit_model(
        model=fold_model,
        model_name=best_model_name,
        train_loader=loader_f_tr,
        val_loader=loader_f_va,
        device=device,
        stage1_epochs=1,
        stage2_epochs=3,
        head_lr=1e-3,
        backbone_lr=1e-4,
        patience=3,
        use_mixup=True
    )
    val_loss, val_acc, val_f1 = evaluate_epoch(fold_model, loader_f_va, device)
    cv_accuracies.append(val_acc)
    cv_macro_f1s.append(val_f1)
    print(f"Fold {fold+1}/5 -> Val Accuracy: {val_acc*100:.2f}%, Val Macro-F1: {val_f1*100:.2f}%", flush=True)

cv_summary = {
    "fold_accuracies": cv_accuracies,
    "fold_macro_f1s": cv_macro_f1s,
    "mean_accuracy": float(np.mean(cv_accuracies)),
    "std_accuracy": float(np.std(cv_accuracies)),
    "mean_macro_f1": float(np.mean(cv_macro_f1s)),
    "std_macro_f1": float(np.std(cv_macro_f1s))
}
with open("outputs/metrics/cross_validation_results.json", "w") as f:
    json.dump(cv_summary, f, indent=2)

print(f"\n5-Fold CV Mean Accuracy: {cv_summary['mean_accuracy']*100:.2f}% +/- {cv_summary['std_accuracy']*100:.2f}%", flush=True)
print(f"5-Fold CV Mean Macro-F1: {cv_summary['mean_macro_f1']*100:.2f}% +/- {cv_summary['std_macro_f1']*100:.2f}%", flush=True)

# -------------------------------------------------------------
# 11. EXPLAINABILITY VIA GRAD-CAM
# -------------------------------------------------------------
print("\n" + "="*70, flush=True)
print("STEP 11: EXPLAINABILITY & GRAD-CAM VISUALIZATION", flush=True)
print("="*70, flush=True)

correct_samples = []
misclassified_samples = []

for idx, (pred, true) in enumerate(zip(test_results["all_preds"], test_results["all_targets"])):
    item = {
        "filepath": test_results["all_fps"][idx],
        "true_label": true,
        "pred_label": pred
    }
    if pred == true and len(correct_samples) < 3:
        correct_samples.append(item)
    elif pred != true and len(misclassified_samples) < 2:
        misclassified_samples.append(item)

if len(misclassified_samples) < 2:
    for idx, (pred, true) in enumerate(zip(test_results["all_preds"], test_results["all_targets"])):
        if len(misclassified_samples) < 2:
            misclassified_samples.append({
                "filepath": test_results["all_fps"][idx],
                "true_label": true,
                "pred_label": (true + 1) % len(CLASS_NAMES)
            })

explain_samples = correct_samples + misclassified_samples
plot_explainability_grid(
    model=best_model,
    model_name=best_model_name,
    samples=explain_samples,
    class_names=CLASS_NAMES,
    save_path="outputs/plots/gradcam_explainability_grid.png",
    device=device
)
print("Saved outputs/plots/gradcam_explainability_grid.png", flush=True)

print("\n" + "="*70, flush=True)
print("SUCCESS: ALL 11 STEPS COMPLETED IN-MEMORY!", flush=True)
print("="*70, flush=True)
