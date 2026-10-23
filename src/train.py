"""
Training Pipeline for Indian Medicinal Plant Leaf Identification (PS-10)
Implements:
- Two-Stage Fine-Tuning (Head Warmup -> Discriminative LR Fine-Tuning)
- AdamW + Cosine Annealing LR Schedule
- MixUp / CutMix and Label Smoothing Loss
- Early Stopping on Validation Macro-F1
- Checkpoint persistence and history logging
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score


def soft_cross_entropy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes cross-entropy loss against soft/blended one-hot targets (MixUp/CutMix/Smoothing)."""
    log_probs = F.log_softmax(pred, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool = True):
    """Freezes or unfreezes backbone feature extractor parameters."""
    name_lower = model_name.lower()
    if name_lower == "custom_cnn":
        for p in model.parameters():
            p.requires_grad = True
    elif "resnet" in name_lower:
        for name, p in model.named_parameters():
            if "fc" not in name:
                p.requires_grad = trainable
            else:
                p.requires_grad = True
    elif "efficientnet" in name_lower or "mobilenet" in name_lower:
        for name, p in model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = trainable
            else:
                p.requires_grad = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_mixup: bool = False
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        if use_mixup and len(batch) == 4:
            imgs, targets, true_labels, _ = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            true_labels = true_labels.to(device)
        else:
            imgs, true_labels, _ = batch
            imgs = imgs.to(device)
            true_labels = true_labels.to(device)
            targets = None

        optimizer.zero_grad()
        outputs = model(imgs)

        if targets is not None:
            loss = soft_cross_entropy(outputs, targets)
        else:
            loss = F.cross_entropy(outputs, true_labels, label_smoothing=0.1)

        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == true_labels).sum().item()
        total += imgs.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float]:
    """Evaluates validation partition for Loss, Accuracy, and Macro-F1."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    total = 0

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = F.cross_entropy(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
        total += imgs.size(0)

    val_loss = running_loss / max(1, total)
    val_acc = accuracy_score(all_targets, all_preds)
    val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return val_loss, val_acc, val_macro_f1


def fit_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage1_epochs: int = 3,
    stage2_epochs: int = 12,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    weight_decay: float = 1e-2,
    patience: int = 5,
    use_mixup: bool = False,
    checkpoint_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Executes full two-stage fine-tuning with discriminative learning rates and early stopping.
    """
    history = {
        "epoch": [],
        "stage": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": []
    }

    best_val_macro_f1 = -1.0
    no_improve_epochs = 0
    current_epoch = 1

    # --- STAGE 1: Train Classification Head Only (Warmup) ---
    if stage1_epochs > 0 and model_name.lower() != "custom_cnn":
        print(f"\n[{model_name}] Starting Stage 1: Warmup classification head ({stage1_epochs} epochs)...")
        set_backbone_trainable(model, model_name, trainable=False)
        optimizer_s1 = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=head_lr,
            weight_decay=weight_decay
        )

        for ep in range(stage1_epochs):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_s1, device, use_mixup=use_mixup)
            val_loss, val_acc, val_macro_f1 = evaluate_epoch(model, val_loader, device)
            elapsed = time.time() - t0

            history["epoch"].append(current_epoch)
            history["stage"].append(1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_macro_f1"].append(val_macro_f1)
            history["lr"].append(head_lr)

            print(f"Stage 1 Ep {current_epoch:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, Macro-F1: {val_macro_f1:.3f}")

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                if checkpoint_path:
                    torch.save(model.state_dict(), checkpoint_path)
            current_epoch += 1

    # --- STAGE 2: Unfreeze Backbone with Discriminative LRs ---
    print(f"\n[{model_name}] Starting Stage 2: End-to-end fine-tuning ({stage2_epochs} epochs)...")
    set_backbone_trainable(model, model_name, trainable=True)

    if model_name.lower() == "custom_cnn":
        param_groups = [{'params': model.parameters(), 'lr': head_lr, 'weight_decay': weight_decay}]
    elif "resnet" in model_name.lower():
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
        head_params = [p for n, p in model.named_parameters() if "fc" in n]
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay}
        ]
    else:  # EfficientNet / MobileNet
        backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
        head_params = [p for n, p in model.named_parameters() if "classifier" in n]
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay}
        ]

    optimizer_s2 = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=stage2_epochs, eta_min=1e-6)

    for ep in range(stage2_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_s2, device, use_mixup=use_mixup)
        val_loss, val_acc, val_macro_f1 = evaluate_epoch(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        curr_lr = optimizer_s2.param_groups[0]['lr']
        history["epoch"].append(current_epoch)
        history["stage"].append(2)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["lr"].append(curr_lr)

        print(f"Stage 2 Ep {current_epoch:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, Macro-F1: {val_macro_f1:.3f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            no_improve_epochs = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  --> Saved new best checkpoint (Macro-F1: {best_val_macro_f1:.4f})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {patience} epochs with no validation Macro-F1 improvement.")
                break
        current_epoch += 1

    # Load best weights before returning
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded optimal checkpoint with validation Macro-F1: {best_val_macro_f1:.4f}")

    return history
