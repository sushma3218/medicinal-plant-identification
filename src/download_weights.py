import os
import torch
import torchvision.models as models

weights_dir = "weights"
os.makedirs(weights_dir, exist_ok=True)

models_to_cache = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, "resnet50_imagenet.pth"),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, "efficientnet_b0_imagenet.pth"),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT, "mobilenet_v3_large_imagenet.pth")
}

for name, (model_fn, weights, filename) in models_to_cache.items():
    filepath = os.path.join(weights_dir, filename)
    print(f"Checking {name} weights at {filepath}...")
    if not os.path.exists(filepath):
        print(f"Downloading {name} pretrained ImageNet weights...")
        try:
            m = model_fn(weights=weights)
            torch.save(m.state_dict(), filepath)
            print(f"Saved {name} to {filepath} ({os.path.getsize(filepath) / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"Error downloading {name}: {e}")
    else:
        print(f"Found existing weights for {name} ({os.path.getsize(filepath) / (1024*1024):.2f} MB)")
