import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTModel

# This is the custom class you defined for your model
class ViTForFruitAndVegQuality(nn.Module):
    def __init__(self, model_name_or_path, num_product_labels):
        super().__init__()
        self.num_product_labels = num_product_labels
        self.vit = ViTModel.from_pretrained(model_name_or_path)
        self.classification_head = nn.Linear(self.vit.config.hidden_size, num_product_labels)
        self.regression_head = nn.Linear(self.vit.config.hidden_size, 1)

    def forward(self, pixel_values, product_labels=None, quality_labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        product_logits = self.classification_head(pooled_output)
        quality_score = self.regression_head(pooled_output)

        # The forward pass can be simplified for inference
        return {
            "product_logits": product_logits,
            "quality_score": quality_score,
        }


BASE_MODEL_CHECKPOINT = "google/vit-base-patch16-224"

image_processor = ViTImageProcessor.from_pretrained(BASE_MODEL_CHECKPOINT)
print("Image processor loaded.")

product_names = sorted([
    'apple', 'strawberry', 'bell pepper', 'tomato', 'orange',
    'new mexico chile', 'chili pepper', 'lime', 'potato', 'guava',
    'carrot', 'banana', 'mango', 'cucumber', 'pomegranate'
])

id_to_product = {i: name for i, name in enumerate(product_names)}
print("Mapping Done.")

image_path = "YOUR_IMAGE PATH" #ATTENTION

try:
    print(f"Opening image: {image_path}")
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"❌ Error: Image path '{image_path}' from manifest is not valid.")
    exit()

inputs = image_processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs['pixel_values']

print("Loading model and processor...")
device = "cuda" if torch.cuda.is_available() else "cpu"
image_processor = ViTImageProcessor.from_pretrained(BASE_MODEL_CHECKPOINT)
model = ViTForFruitAndVegQuality(
    model_name_or_path=BASE_MODEL_CHECKPOINT,
    num_product_labels=len(id_to_product)
)

SAVED_MODEL_DIR = ".../hf_models_cache/models--Graviton17--vit-fruit-veg-quality-predictor/snapshots/..." #ATTENTION

try:
    model_weights_path = os.path.join(SAVED_MODEL_DIR, 'model.safetensors')
    if not os.path.exists(model_weights_path):
        model_weights_path = os.path.join(SAVED_MODEL_DIR, 'pytorch_model.bin')

    if model_weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        model.load_state_dict(load_file(model_weights_path))
    else:
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device(device)))
    print("✅ Model weights loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model weights not found in '{SAVED_MODEL_DIR}'.")
    raise

model.to(device)
model.eval()
print(f"Model is on device: {device}")

print("Running inference...")
with torch.no_grad():
    outputs = model(pixel_values=pixel_values)

# Get classification result
product_logits = outputs['product_logits']
predicted_class_idx = torch.argmax(product_logits, dim=-1).item()
predicted_product = id_to_product[predicted_class_idx]

# Get regression result
quality_score = outputs['quality_score'].item()

# Display the comparison
print("\n" + "="*40)
print("--- 🧐 Cross-Verification Results ---")
print(f"File: ...{image_path[-40:]}")
print("-" * 40)
print(f"✅ Predicted Product: \t{predicted_product.title()}")
print("-" * 40)
print(f"✅ Predicted Quality: \t{quality_score:.2f}")
print("="*40)