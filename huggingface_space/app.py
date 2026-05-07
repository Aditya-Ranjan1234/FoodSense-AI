import base64
import io
import os
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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


class VisionRequest(BaseModel):
    image: str


app = FastAPI(title="FoodSense AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


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


def decode_image(data_url):
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        raw = base64.b64decode(data_url)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image payload") from exc


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


@app.post("/api/vision/analyze")
def analyze_vision(payload: VisionRequest):
    image = decode_image(payload.image)

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
    quality_note = None

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
        quality_note = str(exc)

    risk = max(0.0, min(100.0, spoilage_score * 100.0))
    shelf_life = estimate_shelf_life(freshness, quality_score, is_spoiled)
    q_label = quality_label(quality_score)
    freshness_label = "Spoiled" if is_spoiled else "Fresh"

    deduction = (
        f"The image is classified as {freshness_label.lower()} with {risk:.1f}% spoilage risk. "
        f"The produce/quality branch predicts {product} with a {q_label.lower()} visual quality score. "
        f"Estimated visual shelf life is about {shelf_life} hours."
    )
    if quality_note:
        deduction += f" Quality model fallback used: {quality_note}"

    return {
        "freshness": round(freshness, 1),
        "risk": round(risk, 1),
        "quality": q_label,
        "quality_score": round(quality_score, 1),
        "shelf_life": shelf_life,
        "product": product,
        "freshness_label": freshness_label,
        "spoilage_confidence": round(spoilage_conf * 100.0, 1),
        "deduction": deduction,
    }
