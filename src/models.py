"""
Model Architectures for Indian Medicinal Plant Leaf Identification (PS-10)
Supports:
1. Custom CNN from scratch (Baseline)
2. ResNet-50 (Pretrained Transfer Learning)
3. EfficientNet-B0 (Pretrained Transfer Learning)
4. MobileNetV3-Large (Pretrained Transfer Learning)
5. Model Ensemble (ResNet50 + EfficientNet-B0)
Includes robust offline weight loading from `weights/` directory.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple


class CustomCNNBaseline(nn.Module):
    """
    A lightweight, explainable custom Convolutional Neural Network trained from scratch.
    Serves as an empirical baseline to prove the value of transfer learning.
    Architecture: 4 Conv-BN-ReLU-Pool blocks + adaptive pooling + dense classification head.
    """
    def __init__(self, num_classes: int = 30, in_channels: int = 3):
        super(CustomCNNBaseline, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.global_pool(feat)
        out = self.classifier(pooled)
        return out


def _load_offline_or_online(model: nn.Module, weights_path: str, weights_enum, model_name: str) -> nn.Module:
    """Loads weights from weights/ if available, otherwise downloads once or uses weights_enum."""
    if os.path.exists(weights_path):
        print(f"Loading {model_name} from local offline cache: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print(f"Local cache not found for {model_name}. Attempting download...")
        try:
            downloaded = models.get_model(model_name, weights=weights_enum)
            model.load_state_dict(downloaded.state_dict())
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            torch.save(downloaded.state_dict(), weights_path)
            print(f"Cached {model_name} weights to {weights_path}")
        except Exception as e:
            print(f"Warning: Could not fetch pretrained weights ({e}). Initializing randomly.")
    return model


def build_resnet50(num_classes: int = 30, pretrained: bool = True, weights_dir: str = "weights") -> nn.Module:
    """ResNet-50 with custom classification head for 30 medicinal plant species."""
    model = models.resnet50(weights=None)
    if pretrained:
        weights_path = os.path.join(weights_dir, "resnet50_imagenet.pth")
        model = _load_offline_or_online(model, weights_path, models.ResNet50_Weights.DEFAULT, "resnet50")
    
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.20),
        nn.Linear(512, num_classes)
    )
    return model


def build_efficientnet_b0(num_classes: int = 30, pretrained: bool = True, weights_dir: str = "weights") -> nn.Module:
    """EfficientNet-B0 with custom classification head."""
    model = models.efficientnet_b0(weights=None)
    if pretrained:
        weights_path = os.path.join(weights_dir, "efficientnet_b0_imagenet.pth")
        model = _load_offline_or_online(model, weights_path, models.EfficientNet_B0_Weights.DEFAULT, "efficientnet_b0")
    
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.20),
        nn.Linear(512, num_classes)
    )
    return model


def build_mobilenet_v3_large(num_classes: int = 30, pretrained: bool = True, weights_dir: str = "weights") -> nn.Module:
    """MobileNetV3-Large with custom classification head."""
    model = models.mobilenet_v3_large(weights=None)
    if pretrained:
        weights_path = os.path.join(weights_dir, "mobilenet_v3_large_imagenet.pth")
        model = _load_offline_or_online(model, weights_path, models.MobileNet_V3_Large_Weights.DEFAULT, "mobilenet_v3_large")
    
    in_features = model.classifier[0].in_features  # 960
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.20),
        nn.Linear(512, num_classes)
    )
    return model


class EnsembleClassifier(nn.Module):
    """
    Inference ensemble combining two top-performing models via soft probability averaging.
    """
    def __init__(self, model_a: nn.Module, model_b: nn.Module, weight_a: float = 0.5):
        super(EnsembleClassifier, self).__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = weight_a
        self.weight_b = 1.0 - weight_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_a = torch.softmax(self.model_a(x), dim=1)
        out_b = torch.softmax(self.model_b(x), dim=1)
        ensemble_prob = self.weight_a * out_a + self.weight_b * out_b
        return torch.log(ensemble_prob + 1e-8)


def get_model(name: str, num_classes: int = 30, pretrained: bool = True, weights_dir: str = "weights") -> nn.Module:
    """Factory function for instantiating model architectures."""
    name_lower = name.lower()
    if name_lower == "custom_cnn":
        return CustomCNNBaseline(num_classes=num_classes)
    elif name_lower == "resnet50":
        return build_resnet50(num_classes=num_classes, pretrained=pretrained, weights_dir=weights_dir)
    elif name_lower in ["efficientnet", "efficientnet_b0"]:
        return build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained, weights_dir=weights_dir)
    elif name_lower in ["mobilenet", "mobilenet_v3", "mobilenet_v3_large"]:
        return build_mobilenet_v3_large(num_classes=num_classes, pretrained=pretrained, weights_dir=weights_dir)
    else:
        raise ValueError(f"Unsupported model architecture: {name}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
