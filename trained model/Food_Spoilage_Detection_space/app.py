import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Define constants
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Load the trained model
model = tf.keras.models.load_model('food_spoilage_model_fine_tuned.keras')

# Function to preprocess and predict on a single image
def predict_spoilage(image):
    # Convert Gradio image (PIL) to the format expected by the model
    img = image.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    result = "Spoiled" if prediction[0][0] > 0.5 else "Fresh"
    confidence = prediction[0][0] if result == "Spoiled" else 1 - prediction[0][0]
    
    return f"Prediction: {result} (Confidence: {confidence:.2%})"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_spoilage,
    inputs=gr.Image(type="pil", label="Upload an image of food"),
    outputs=gr.Textbox(label="Prediction"),
    title="Food Spoilage Detection",
    description="Upload an image to determine if the food is fresh or spoiled."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()