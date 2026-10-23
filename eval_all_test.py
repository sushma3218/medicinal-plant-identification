import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from torchvision import transforms

from src.data import PreloadedMedicinalLeafDataset, get_transforms
from src.models import get_model

device = torch.device("cpu")
torch.set_num_threads(16)

# Load class names
with open("outputs/class_names.json", "r") as f:
    class_names = json.load(f)

# Load test dataset
test_df = pd.read_csv("splits/test_split.csv")
t_eval = get_transforms("eval", img_size=224)
test_ds = PreloadedMedicinalLeafDataset(test_df, transform=t_eval)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

models = {
    "custom_cnn": "outputs/checkpoints/custom_cnn_best.pt",
    "mobilenet_v3_large": "outputs/checkpoints/mobilenet_v3_large_best.pt",
    "efficientnet_b0": "outputs/checkpoints/efficientnet_b0_best.pt",
    "resnet50": "outputs/checkpoints/resnet50_best.pt",
}

test_summary = []

for m_name, ckpt_path in models.items():
    print(f"Evaluating {m_name} on test set (275 images)...")
    is_scratch = (m_name == "custom_cnn")
    model = get_model(m_name, num_classes=30, pretrained=not is_scratch, weights_dir="weights")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.numpy())

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_p = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_r = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    
    test_summary.append({
        "Model": m_name,
        "Test_Accuracy": round(acc, 4),
        "Test_Macro_F1": round(macro_f1, 4),
        "Test_Macro_Precision": round(macro_p, 4),
        "Test_Macro_Recall": round(macro_r, 4),
    })

    if m_name == "resnet50":
        # Save as outputs/best.pt
        import shutil
        shutil.copyfile("outputs/checkpoints/resnet50_best.pt", "outputs/best.pt")

        # Classification report
        rep = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
        with open("outputs/metrics/test_classification_report.json", "w") as f:
            json.dump(rep, f, indent=2)

        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=[c.split('(')[0][:12] for c in class_names], yticklabels=[c.split('(')[0][:12] for c in class_names])
        plt.title("ResNet-50 Test Set Normalized Confusion Matrix (30 Classes)", fontsize=13, fontweight='bold')
        plt.xlabel("Predicted Species", fontsize=11)
        plt.ylabel("True Species", fontsize=11)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig("outputs/plots/confusion_matrix.png", dpi=300)
        plt.close()

        # Find confused pairs
        np.fill_diagonal(cm, 0)
        confused = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if cm[i, j] > 0:
                    confused.append({
                        "true_class": class_names[i],
                        "pred_class": class_names[j],
                        "count": int(cm[i, j])
                    })
        confused = sorted(confused, key=lambda x: x["count"], reverse=True)
        with open("outputs/metrics/test_confused_pairs.json", "w") as f:
            json.dump(confused, f, indent=2)

        with open("outputs/metrics/test_metrics.json", "w") as f:
            json.dump({
                "model": "resnet50",
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "macro_precision": round(macro_p, 4),
                "macro_recall": round(macro_r, 4),
                "top_confused": confused[:5]
            }, f, indent=2)

pd.DataFrame(test_summary).to_csv("outputs/metrics/test_model_comparison.csv", index=False)
print("--- TEST EVALUATION COMPLETE ---")
print(pd.DataFrame(test_summary).to_string(index=False))
