# FoodSense: Multimodal AI System for Early Spoilage Prediction

FoodSense is an IoT + AI multimodal system designed to continuously track factors such as temperature, humidity, and gas levels inside a food container to determine the freshness and safety of stored food. 

## Structure
- `arduino_code/`: Contains the `.ino` files for the ESP32/Arduino to read the DHT11 and MQ135 sensors.
- `dataset/`: Contains local CSVs.
- `docs/`: Reference documents and synopsis.
- `models/`: Python PyTorch scripts for training the Multi-Layer Perceptron (MLP) and Temporal LSTM models.

## Setup Instructions

1. Activate the Python Virtual Environment:
   ```bash
   .\venv\Scripts\Activate.ps1
   ```
2. Run Model Training (MLP):
   ```bash
   cd models/src
   python train_mlp.py
   ```
3. Run Model Testing:
   ```bash
   python test_mlp.py
   ```

## Next Steps
- Collect actual hardware logs from `foodsense_sensor_fusion.ino` and replace `arduino_timeseries_log.csv`.
- Train the LSTM model on real Arduino logs.
- Integrate the Vision branch using MobileViTv2.
