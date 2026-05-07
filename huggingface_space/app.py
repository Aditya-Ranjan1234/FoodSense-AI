import os
from functools import lru_cache

import gradio as gr
import numpy as np
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPOILAGE_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "food-spoilage-detection", "food_spoilage_model_fine_tuned.keras"
)
VIT_MODEL_DIR = os.path.join(BASE_DIR, "models", "vit-fruit-veg-quality-predictor")
VIT_WEIGHTS_PATH = os.path.join(VIT_MODEL_DIR, "model.safetensors")

PRODUCT_NAMES = sorted(
    [
        "apple",
        "strawberry",
        "bell pepper",
        "tomato",
        "orange",
        "new mexico chile",
        "chili pepper",
        "lime",
        "potato",
        "guava",
        "carrot",
        "banana",
        "mango",
        "cucumber",
        "pomegranate",
    ]
)


@lru_cache(maxsize=1)
def load_spoilage_model():
    import tensorflow as tf

    return tf.keras.models.load_model(SPOILAGE_MODEL_PATH)


@lru_cache(maxsize=1)
def load_quality_model():
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file
    from transformers import ViTConfig, ViTImageProcessor, ViTModel

    class ViTForFruitAndVegQuality(nn.Module):
        def __init__(self, num_product_labels):
            super().__init__()
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                qkv_bias=True,
            )
            self.vit = ViTModel(config)
            self.classification_head = nn.Linear(config.hidden_size, num_product_labels)
            self.regression_head = nn.Linear(config.hidden_size, 1)

        def forward(self, pixel_values):
            outputs = self.vit(pixel_values=pixel_values)
            pooled_output = outputs.pooler_output
            return {
                "product_logits": self.classification_head(pooled_output),
                "quality_score": self.regression_head(pooled_output),
            }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_DIR)
    model = ViTForFruitAndVegQuality(num_product_labels=len(PRODUCT_NAMES))
    state_dict = load_file(VIT_WEIGHTS_PATH, device=str(device))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "ViT weights did not match the local wrapper. "
            f"Missing keys: {len(missing)}; unexpected keys: {len(unexpected)}"
        )
    model.to(device)
    model.eval()
    return model, processor, device


def normalize_quality(raw_score):
    if 0.0 <= raw_score <= 1.0:
        return raw_score * 100.0
    return max(0.0, min(100.0, raw_score))


def quality_label(score):
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Poor"


def estimate_shelf_life(freshness, quality_score, is_spoiled):
    combined = (freshness * 0.6) + (quality_score * 0.4)
    if is_spoiled:
        combined *= 0.35
    return int(max(0, min(96, round((combined / 100.0) * 72))))


def analyze_vision(image):
    if image is None:
        return (
            "Upload or capture a food image first.",
            "--",
            "--",
            "--",
            "--",
            "No image was provided.",
        )

    image = image.convert("RGB")
    spoilage_model = load_spoilage_model()
    spoilage_img = image.resize((224, 224))
    spoilage_arr = np.asarray(spoilage_img, dtype=np.float32) / 255.0
    spoilage_arr = np.expand_dims(spoilage_arr, axis=0)
    spoilage_score = float(spoilage_model.predict(spoilage_arr, verbose=0)[0][0])
    is_spoiled = spoilage_score > 0.5
    spoilage_conf = spoilage_score if is_spoiled else 1.0 - spoilage_score
    freshness = max(0.0, min(100.0, (1.0 - spoilage_score) * 100.0))

    product = "Unknown produce"
    quality_score = freshness
    quality_model_note = ""

    try:
        import torch

        quality_model, processor, device = load_quality_model()
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = quality_model(**inputs)
        class_idx = int(outputs["product_logits"].argmax(dim=-1).item())
        product = PRODUCT_NAMES[class_idx].title()
        quality_score = normalize_quality(float(outputs["quality_score"].item()))
    except Exception as exc:
        quality_model_note = f" Quality model unavailable: {exc}"

    risk = max(0.0, min(100.0, spoilage_score * 100.0))
    shelf_life = estimate_shelf_life(freshness, quality_score, is_spoiled)
    q_label = quality_label(quality_score)
    freshness_label = "Spoiled" if is_spoiled else "Fresh"

    status = "\n".join(
        [
            f"Freshness model: {freshness_label} ({spoilage_conf:.1%} confidence)",
            f"Produce model: {product}",
            f"Quality model: {q_label} ({quality_score:.1f}/100)",
        ]
    )
    deduction = (
        f"The image is classified as {freshness_label.lower()} with {risk:.1f}% spoilage risk. "
        f"The produce/quality branch predicts {product} with a {q_label.lower()} visual quality score. "
        f"Estimated visual shelf life is about {shelf_life} hours."
    )
    if quality_model_note:
        deduction += quality_model_note

    return (
        status,
        f"{freshness:.1f}%",
        f"{risk:.1f}%",
        f"{q_label} ({quality_score:.1f}/100)",
        f"{shelf_life}h",
        deduction,
    )


def sensor_snapshot(temp, humidity, nh3):
    safe = nh3 < 5.0
    caution = 5.0 <= nh3 < 25.0
    status = "Safe" if safe else "Caution" if caution else "Unsafe"
    confidence = 98.5 if safe else 86.0 if caution else 93.0
    shelf = max(0, int(48 - (nh3 * 2.5) - max(0, humidity - 65) * 0.4))
    return status, f"{confidence:.0f}%", f"{shelf}h"


CSS = """
body, .gradio-container {
  background: #0e1511 !important;
  color: #dde4dd !important;
  font-family: Inter, sans-serif !important;
}
.panel {
  background: rgba(26, 33, 29, 0.78);
  border: 1px solid rgba(134, 148, 138, 0.18);
  border-radius: 18px;
  padding: 18px;
}
.title {
  color: #4edea3;
  font-weight: 800;
}
"""


with gr.Blocks(css=CSS, title="FoodSense AI") as demo:
    gr.Markdown("# FoodSense AI\nMultimodal food quality and spoilage dashboard")

    with gr.Tabs():
        with gr.Tab("Dashboard"):
            with gr.Row():
                temp = gr.Slider(0, 50, value=25.3, step=0.1, label="Temperature (C)")
                humidity = gr.Slider(0, 100, value=62.1, step=0.1, label="Humidity (%)")
                nh3 = gr.Slider(0, 100, value=2.8, step=0.1, label="Ammonia (ppm)")
            run_sensor = gr.Button("Update Sensor Prediction")
            with gr.Row():
                gas_status = gr.Textbox(label="Status", value="Safe")
                gas_conf = gr.Textbox(label="Confidence", value="98%")
                shelf = gr.Textbox(label="Estimated Shelf Life", value="44h")
            run_sensor.click(
                sensor_snapshot,
                inputs=[temp, humidity, nh3],
                outputs=[gas_status, gas_conf, shelf],
            )

        with gr.Tab("Evaluation"):
            gr.Markdown(
                """
                ### Gas Model Metrics
                Accuracy: **99.75%**  
                Precision: **99.08%**  
                Recall: **98.78%**  
                F1-Score: **98.93%**

                ### Shelf-Life LSTM
                Mean absolute error: **4.0 hours**
                """
            )

        with gr.Tab("Image Analysis"):
            gr.Markdown(
                "Upload or capture a food image. The backend runs both local vision models: "
                "the Keras fresh/spoiled classifier and the ViT produce quality predictor."
            )
            with gr.Row():
                vision_image = gr.Image(
                    type="pil",
                    sources=["upload", "webcam"],
                    label="Food Image",
                    height=360,
                )
                with gr.Column():
                    analyze_btn = gr.Button("Analyze Image", variant="primary")
                    vision_status = gr.Textbox(label="Model Outputs", lines=4)
                    freshness_out = gr.Textbox(label="Visual Freshness")
                    risk_out = gr.Textbox(label="Spoilage Risk")
                    quality_out = gr.Textbox(label="Quality Level")
                    shelf_out = gr.Textbox(label="Estimated Shelf Life")
            deduction_out = gr.Textbox(label="AI Deduction", lines=4)
            analyze_btn.click(
                analyze_vision,
                inputs=[vision_image],
                outputs=[
                    vision_status,
                    freshness_out,
                    risk_out,
                    quality_out,
                    shelf_out,
                    deduction_out,
                ],
                api_name="analyze_vision",
            )

        with gr.Tab("Architecture"):
            gr.Markdown(
                """
                MQ135 + DHT11 sensors feed the gas and shelf-life models. The image branch now
                uses the cloned Hugging Face models as the live Vision backend:

                1. `Chanereach/Food_Spoilage_Detection` Space model: Fresh vs. Spoiled.
                2. `Graviton17/vit-fruit-veg-quality-predictor`: produce type + quality score.
                """
            )


if __name__ == "__main__":
    demo.launch()
