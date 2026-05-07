---
title: FoodSense AI
emoji: 🥦
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 5.28.0
app_file: app.py
pinned: true
license: mit
short_description: Multimodal AI for Early Food Spoilage Prediction
---

# FoodSense AI

This Space runs the FoodSense dashboard with a live Vision backend.

The Image Analysis tab uses two local model assets:

- `Chanereach/Food_Spoilage_Detection`: TensorFlow/Keras fresh vs. spoiled classifier.
- `Graviton17/vit-fruit-veg-quality-predictor`: ViT produce classifier plus visual quality regressor.
