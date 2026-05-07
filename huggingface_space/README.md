---
title: FoodSense AI
emoji: 🥦
colorFrom: green
colorTo: green
sdk: docker
suggested_hardware: cpu-upgrade
pinned: true
license: mit
short_description: Multimodal AI for Early Food Spoilage Prediction
---

# FoodSense AI

This Space runs the original FoodSense static dashboard with a FastAPI model backend.

The Image Analysis tab uses two local model assets:

- `Chanereach/Food_Spoilage_Detection`: TensorFlow/Keras fresh vs. spoiled classifier.
- `Graviton17/vit-fruit-veg-quality-predictor`: ViT produce classifier plus visual quality regressor.
