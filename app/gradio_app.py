# app/gradio_app.py
# Interactive demo: upload MRI slice → prediction + Grad-CAM overlay

import sys
sys.path.append('..')

import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.gradcam import GradCAM, load_trained_model
from src.transforms import val_transforms
from src.utils import get_device, load_config

device = get_device()
config = load_config('../configs/config.yaml')

# ── Load all 3 trained models ──────────────────────────────────
MODELS = {}
LABEL_TYPES = ['abnormal', 'acl', 'meniscus']
LABEL_DISPLAY = {
    'abnormal': 'Abnormal (any injury)',
    'acl': 'Sprain (ACL tear)',
    'meniscus': 'Structural / Meniscus damage',
}

for label_type in LABEL_TYPES:
    ckpt_path = f"../models/best_{label_type}_sagittal.pth"
    try:
        MODELS[label_type] = load_trained_model(ckpt_path, device)
        print(f"Loaded {label_type} model ✓")
    except FileNotFoundError:
        print(f"⚠ Checkpoint missing: {ckpt_path}")


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Convert uploaded image (any format) into model input tensor."""
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image[:, :, :3]

    if image_rgb.dtype != np.uint8:
        img_min, img_max = image_rgb.min(), image_rgb.max()
        image_rgb = ((image_rgb - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)

    tensor = val_transforms(image_rgb)
    return tensor.unsqueeze(0)


def predict(image: np.ndarray, label_type: str):
    if image is None:
        return None, "Please upload an MRI slice image."

    if label_type not in MODELS:
        return None, f"Model for '{label_type}' not found. Train it first."

    model = MODELS[label_type]
    input_tensor = preprocess_image(image)

    gradcam = GradCAM(model)
    prob, heatmap = gradcam.generate(input_tensor, device)

    # Build overlay using the original uploaded slice
    gray = image if image.ndim == 2 else image.mean(axis=2)
    overlay = gradcam.overlay(gray, heatmap)

    pred_label = "Positive (injury detected)" if prob >= 0.5 else "Negative (likely normal)"
    confidence = prob if prob >= 0.5 else 1 - prob

    result_text = (
        f"## {LABEL_DISPLAY[label_type]}\n\n"
        f"**Prediction:** {pred_label}\n\n"
        f"**Confidence:** {confidence:.1%}\n\n"
        f"**Raw probability:** {prob:.3f}\n\n"
        f"---\n"
        f" Research demo only — not for clinical use."
    )

    return overlay, result_text


# ── Gradio UI ───────────────────────────────────────────────────
with gr.Blocks(title="MRI Knee Injury Classifier") as demo:
    gr.Markdown(
        """
        # 🦴 MRI Knee Injury Classifier
        Upload a knee MRI slice (sagittal plane) to detect signs of
        **sprain (ACL tear)** or **structural damage (meniscus)**.
        The heatmap shows which regions influenced the model's decision.

        *Built on the Stanford MRNet dataset · ResNet50 + Grad-CAM*
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload MRI Slice", type="numpy")
            label_select = gr.Radio(
                choices=list(LABEL_TYPES),
                value="acl",
                label="Injury type to check",
            )
            submit_btn = gr.Button("Run Prediction", variant="primary")

        with gr.Column():
            heatmap_output = gr.Image(label="Grad-CAM Overlay")
            result_output = gr.Markdown()

    submit_btn.click(
        fn=predict,
        inputs=[image_input, label_select],
        outputs=[heatmap_output, result_output],
    )

    gr.Markdown(
        """
        ---
         **Disclaimer:** This is a research/educational project, not a
        diagnostic tool. Always consult a qualified radiologist.
        """
    )

if __name__ == "__main__":
    demo.launch(share=False)
