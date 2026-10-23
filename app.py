"""
Interactive Web Application for Indian Medicinal Plant Leaf Identification (PS-10)
Built with Gradio and PyTorch.
Features:
- Live Leaf Image Upload & Classification (30 Ayurvedic/Siddha Species)
- Real-Time Grad-CAM Saliency Attention Heatmap
- Top-3 Confidence Probabilities
- Full Botanical Metadata (Botanical Name, Vernacular Name, Family, Medicinal Uses)
- Safety Rejection Warning for Ambiguous/Low-Confidence Predictions
- Clinical & Ethical Decision-Support Disclaimer
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gradio as gr
import torchvision.transforms as T

from src.data import CLASS_NAMES, SPECIES_METADATA
from src.models import get_model
from src.gradcam import generate_gradcam_overlay

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)

# Load Model
MODEL_NAME = "resnet50"
CHECKPOINT_PATH = "outputs/best.pt"
if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = "outputs/checkpoints/resnet50_best.pt"

print(f"Loading production model from: {CHECKPOINT_PATH}...")
model = get_model(MODEL_NAME, num_classes=30, pretrained=False).to(device)
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    print("Loaded model weights successfully.")
else:
    print("Warning: Model checkpoint not yet found. Initialized with fresh weights.")
model.eval()

# Safety Rejection Threshold (calibrated in Step 6)
SAFETY_CONFIDENCE_THRESHOLD = 0.85


def predict_leaf(input_image: Image.Image, show_gradcam: bool = True):
    """
    Inference pipeline for input leaf photograph:
    Computes top predictions, generates Grad-CAM overlay, and verifies safety threshold.
    """
    if input_image is None:
        return None, {}, "Please upload a valid leaf image.", ""

    img_rgb = input_image.convert("RGB")

    # Generate Grad-CAM & Predictions
    rgb_float, cam_overlay, pred_idx, confidence = generate_gradcam_overlay(
        model=model,
        model_name=MODEL_NAME,
        img_pil=img_rgb,
        device=device,
        img_size=192
    )

    pred_class_name = CLASS_NAMES[pred_idx]
    meta = SPECIES_METADATA.get(pred_class_name, {
        "botanical_name": pred_class_name,
        "common_name": pred_class_name,
        "family": "Unknown",
        "medicinal_use": "Ayurvedic medicinal plant."
    })

    # Compute full probability distribution for Top-3
    norm_transform = T.Compose([
        T.Resize((192, 192)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    inp_tensor = norm_transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inp_tensor)
        probs = F.softmax(logits, dim=1)[0]
        top3_probs, top3_indices = torch.topk(probs, k=3)

    conf_dict = {
        CLASS_NAMES[idx]: float(prob)
        for prob, idx in zip(top3_probs.cpu().numpy(), top3_indices.cpu().numpy())
    }

    # Safety Rejection Analysis
    if confidence < SAFETY_CONFIDENCE_THRESHOLD:
        safety_status = (
            f"⚠️ **UNCERTAIN PREDICTION (SAFETY REJECTION TRIGGERED)**\n\n"
            f"The model confidence is **{confidence*100:.1f}%**, which is below our safety verification threshold of **{SAFETY_CONFIDENCE_THRESHOLD*100:.0f}%**.\n"
            f"**Recommendation**: In Ayurveda and Siddha systems, misidentifying look-alike species (e.g., toxic *Nerium Oleander* vs edible leaves) can lead to severe health hazards. "
            f"Please **do not use or consume** this specimen without physical verification from an expert botanist or certified practitioner."
        )
    else:
        safety_status = (
            f"✅ **HIGH CONFIDENCE IDENTIFICATION ({confidence*100:.1f}%)**\n\n"
            f"Classification confidence meets the safety threshold ($\ge {SAFETY_CONFIDENCE_THRESHOLD*100:.0f}\%$). "
            f"Saliency maps indicate attention localized to botanical leaf structures."
        )

    # Detailed Botanical Report
    botanical_report = f"""
### 🌿 Botanical Profile:
- **Species**: {pred_class_name}
- **Botanical / Scientific Name**: *{meta['botanical_name']}*
- **Vernacular / Common Name**: {meta['common_name']}
- **Botanical Family**: {meta['family']}
- **Top Confidence**: {confidence*100:.2f}%

### 💊 Documented Medicinal Applications:
{meta['medicinal_use']}
"""

    output_visual = (cam_overlay * 255).astype(np.uint8) if show_gradcam else img_rgb
    return output_visual, conf_dict, botanical_report, safety_status


# Gradio Custom Theme and Layout
theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="teal"
)

title_html = """
<div style="text-align: center; max-width: 900px; margin: 0 auto; padding: 20px;">
    <h1 style="color: #1e3a2f; font-size: 2.2rem; margin-bottom: 8px;">
        🌿 Indian Medicinal Plant Leaf Identifier
    </h1>
    <h3 style="color: #3b7a57; font-weight: 500; margin-top: 0;">
        Deep Learning Computer Vision for Traditional Ayurveda & Siddha Herbs (PS-10)
    </h3>
    <p style="color: #4a5568; font-size: 0.95rem; line-height: 1.5;">
        Identifies 30 indigenous Indian medicinal species from single leaf photographs using deep transfer learning,
        domain-informed botanical augmentations, temperature calibration, and Grad-CAM visual explainability.
    </p>
</div>
"""

disclaimer_html = """
<div style="background-color: #fff8e1; border-left: 5px solid #ffb300; padding: 15px; margin: 20px 0; border-radius: 4px;">
    <h4 style="margin: 0 0 6px 0; color: #856404; font-size: 1rem;">
        ⚕️ Clinical Decision-Support & Safety Notice
    </h4>
    <p style="margin: 0; color: #66512c; font-size: 0.88rem; line-height: 1.45;">
        This artificial intelligence system is strictly designed as an academic demonstration and clinical decision-support tool.
        It is <strong>not</strong> an autonomous medical diagnostic device. In Ayurveda and Siddha medicine, toxic look-alikes 
        (e.g., <em>Nerium oleander</em>) present severe poisoning risks if confused with therapeutic herbs. 
        Always obtain physical morphological verification from a qualified Ayurvedic physician or taxonomist before consuming any herbal medicine.
    </p>
</div>
"""

# Find sample test images for Gradio examples
examples = []
if os.path.exists("splits/test_split.csv"):
    try:
        import pandas as pd
        tdf = pd.read_csv("splits/test_split.csv")
        sample_rows = tdf.sample(min(4, len(tdf)), random_state=42)
        for _, r in sample_rows.iterrows():
            if os.path.exists(r["filepath"]):
                examples.append([r["filepath"], True])
    except Exception:
        pass

with gr.Blocks(theme=theme, title="Medicinal Leaf Identification (PS-10)") as demo:
    gr.HTML(title_html)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Leaf Photograph (Single Blade on Clear/White Background)")
            gradcam_toggle = gr.Checkbox(value=True, label="Compute & Overlay Grad-CAM Attention Heatmap")
            classify_btn = gr.Button("🔍 Identify Medicinal Plant Species", variant="primary", size="lg")
            
            if examples:
                gr.Examples(examples=examples, inputs=[input_img, gradcam_toggle])

        with gr.Column(scale=1):
            output_img = gr.Image(type="numpy", label="Model Visual Attention (Grad-CAM Saliency Heatmap)")
            top3_labels = gr.Label(num_top_classes=3, label="Top-3 Predictions & Confidence Probabilities")
            safety_box = gr.Markdown(label="Safety Verification")
            report_box = gr.Markdown(label="Botanical & Pharmacological Profile")

    classify_btn.click(
        fn=predict_leaf,
        inputs=[input_img, gradcam_toggle],
        outputs=[output_img, top3_labels, report_box, safety_box]
    )

    gr.HTML(disclaimer_html)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
