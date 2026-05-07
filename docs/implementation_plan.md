# FoodSense: Implementation Plan
**Multimodal AI System for Early Food Spoilage Prediction**  
Team FNO2 | Guide: Dr. Anitha Sandeep | Sem 6 IDP

---

## What Does Each Arduino File Collect?

### 📄 `dht_humidity_temperature.ino` — Standalone DHT11/DHT22
**Data collected every 2 seconds:**

| Field | Type | Range | Unit |
|---|---|---|---|
| `humidity` | float | 0 – 100 | % RH |
| `temperature` | float | -40 – 80 (DHT11: 0–50) | °C |

**Current output format (human-readable, NOT machine-parseable):**
```
Humidity: 72.30 %    Temperature: 28.50 °C
```

**What this data can be used for:**
- 🌡️ **Environmental monitoring** — detect when container conditions are ideal for bacterial growth (>80% RH, >25°C)
- 📈 **Spoilage rate estimation** — humidity and temperature together drive mold growth rate (use Arrhenius equation)
- 🔗 **Compensation input** — correct the gas sensor readings (done in `foodsense_sensor_fusion.ino`)
- 📊 **Anomaly detection** — a sudden humidity spike (food sweating/condensation) is an early spoilage signal

> [!NOTE]
> This file outputs **human-readable text**, not CSV. It cannot be directly piped into Python without parsing. The fusion file fixes this.

---

### 📄 `mq135_gas_sensor.ino` — Standalone MQ135
**Data collected every 2 seconds (after 5-minute warm-up):**

| Field | Type | Range | Notes |
|---|---|---|---|
| `adc` | int | 0 – 1023 | Raw 10-bit ADC reading (50-sample averaged) |
| `rs` | float | varies | Sensor resistance in kΩ |
| `ratio` (Rs/R0) | float | 0.1 – 10+ | Ratio used for ppm calculation |
| `nh3` | float | 0 – 100 | NH₃ concentration (clamped), **ppm** |
| `status` | String | SAFE/CAUTION/UNSAFE | Threshold-based label |

**NH₃ thresholds used:**
```
< 5 ppm   → SAFE
5–25 ppm  → CAUTION
> 25 ppm  → UNSAFE
```

**What this data can be used for:**
- 🧪 **Protein decomposition detection** — NH₃ is released when proteins break down (meat, eggs, fish, dairy)
- 🏷️ **Binary spoilage classification** — the `status` field is a ready-made label for ML training
- 📉 **Trend analysis** — rising NH₃ over hours = spoilage in progress
- ⚠️ **Alert trigger** — ppm crossing threshold triggers notification

> [!WARNING]
> The MQ135 also responds to CO₂, alcohol, benzene, and smoke — it is NOT specific to NH₃ alone. In a sealed food container, this cross-sensitivity is actually useful (spoilage produces multiple VOCs). But you must note this limitation in your report.

---

### 📄 `foodsense_sensor_fusion.ino` — **Main File: Sensor Fusion** ⭐
**Data collected every 2 seconds (after 5-minute warm-up):**

| Field | Type | Notes |
|---|---|---|
| `millis()` | unsigned long | Timestamp in ms since boot |
| `temperature` | float | From DHT11, in °C |
| `humidity` | float | From DHT11, in % RH |
| `nh3` | float | **Temperature + humidity compensated** NH₃ in ppm |
| `status` | String | SAFE / CAUTION / UNSAFE |

**Current CSV output format:**
```
2400000,28.5,72.3,3.21,SAFE
2402300,28.6,72.5,3.45,SAFE
2404600,28.7,73.1,4.10,SAFE
...
```

**Key innovation in this file — Temperature Compensation:**
```cpp
correction = 1.0 + 0.02*(temp - 20.0) + 0.01*(humidity - 65.0)
corrected_rs = raw_rs / correction
```
This makes NH₃ readings more accurate across different temperature/humidity conditions.

**What this combined data can be used for:**

| ML Task | Input Features | Target Label | Algorithm |
|---|---|---|---|
| Binary spoilage classification | temp, humidity, nh3 | SAFE / UNSAFE | Random Forest, SVM, XGBoost |
| Shelf-life regression | time series of [temp, humidity, nh3] | hours_remaining | LSTM, Temporal Transformer |
| Anomaly detection | rolling window of readings | 0=normal, 1=anomaly | Isolation Forest, Autoencoder |
| Condition monitoring | humidity, temp only | risk_category | Rule-based + ML hybrid |

---

## Data Gaps (What the Code Does NOT Collect)

| Missing Data | Why It Matters | Solution |
|---|---|---|
| ❌ No camera / image | Visual spoilage (mold, color) is the strongest signal | Add ESP32-CAM or Raspberry Pi Camera |
| ❌ No absolute timestamp | `millis()` resets on reboot, not wall-clock time | Add DS3231 RTC module or sync via Python |
| ❌ No food label / food type | NH₃ profile differs for chicken vs bread vs tomatoes | Label each recording session manually |
| ❌ No multi-gas (CO₂, H₂S, ethylene) | Spoilage produces many VOCs, MQ135 is one sensor | Add MQ-4 (methane), MQ-9 (CO), or use MiCS-5524 |
| ❌ No image metadata (when captured) | Can't sync camera frames with sensor time series | Add timestamp to each photo filename |

---

## Dataset Sources

### 🖼️ Vision Datasets (for the Camera / CV Branch)

| # | Dataset | Size | Classes | Source | License |
|---|---|---|---|---|---|
| **1** | **Fresh and Stale Fruits & Vegetables** | ~10,000 images | Fresh / Stale (multiple produce) | [Kaggle: raghavrpotdar](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables) | Open |
| **2** | **Fruits Fresh and Rotten for Classification** | 13,599 images | 6 fruits × fresh/rotten | [Kaggle: sriramr](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification) | Open |
| **3** | **Food Freshness Dataset (Large)** | 70,000+ images (~6.4 GB) | 13 produce types × fresh/rotten | [Kaggle: Food Freshness](https://www.kaggle.com/search?q=food+freshness+dataset) | CC0 |
| **4** | **PlantVillage Disease Dataset** | 54,306 images | 38 plant/disease classes | [Kaggle/PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) | CC BY 4.0 |
| **5** | **FruitVision (2025)** | ~8,000 images | Fresh/Rotten + formalin-adulterated | [Kaggle: 2025](https://www.kaggle.com/search?q=fruitvision) | Open |

**🎯 Recommended starting point: Dataset #2 (Sriram)** — 13,599 images, clean labels, widely benchmarked in papers, direct HuggingFace mirror available.

---

### 📊 Sensor / Time-Series Datasets (for the Gas + Temp/Humidity Branch)

| # | Dataset | Size | Sensors | Source |
|---|---|---|---|---|
| **1** | **UCI Gas Sensor Array Drift** | 13,910 samples | 16 metal oxide sensors | [UCI ML Repo](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+under+dynamic+gas+mixtures) |
| **2** | **UCI Wine Quality (sensor-inspired)** | 6,497 rows | 11 chemical features (analog to gas) | [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| **3** | **IoT Food Monitoring (Kaggle)** | varies | Temp, humidity, gas readings | [Kaggle: IoT sensor](https://www.kaggle.com/search?q=iot+food+sensor) |
| **4** | **Your Own Collected Data** | You control it | MQ135 + DHT11 (3 channels) | `foodsense_sensor_fusion.ino` → serial logger |

> [!IMPORTANT]
> For the sensor branch, **your own collected data is the most valuable**. UCI datasets use 16-sensor e-noses — not directly comparable to MQ135. Your contribution is specifically a **3-feature (temp, humidity, NH₃) spoilage model**, which is novel because no public dataset exists for this exact sensor combo on food containers.

---

### 🔀 Multimodal / Combined Datasets

| # | Dataset | Notes |
|---|---|---|
| **1** | **FOOD-101** | 101,000 images, 101 food classes — good for transfer learning base | 
| **2** | **Open Food Facts** | Nutritional + ingredient data — useful for shelf-life priors |
| **3** | **FoodSense (your data)** | You create this — sensor CSV + labeled photos. **This IS the novelty.** |

---

## Full System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HARDWARE LAYER                        │
│                                                          │
│  [MQ135]──────────┐                                      │
│  [DHT11]──────────┼──► Arduino Uno ──► Serial (USB)      │
│  [Camera]─────────┘     (foodsense_sensor_fusion.ino)    │
└───────────────────────────────┬─────────────────────────┘
                                │ CSV stream @ 9600 baud
                                ▼
┌─────────────────────────────────────────────────────────┐
│                    DATA COLLECTION LAYER (Python)        │
│                                                          │
│  serial_logger.py → reads COM port → saves sensor.csv   │
│  camera_capture.py → captures frames → saves images/    │
│  sync.py → aligns image timestamps with sensor readings  │
└───────────────────────────────┬─────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────┐
│                    AI PIPELINE LAYER (Python/PyTorch)    │
│                                                          │
│  Branch A (Vision):                                      │
│    images/ → ResNet50 fine-tune → 2048-dim features      │
│                                                          │
│  Branch B (Sensor):                                      │
│    sensor.csv → LSTM/MLP → 128-dim features              │
│                                                          │
│  Fusion:                                                 │
│    [2048 + 128] → Cross-Attention → Spoilage Score       │
│                 → Shelf-life estimate (hours)            │
│                 → Grad-CAM XAI visualization             │
└───────────────────────────────┬─────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                          │
│                                                          │
│  • SAFE / CAUTION / UNSAFE label                        │
│  • Estimated hours remaining before spoilage             │
│  • Grad-CAM heatmap (which food region is bad)          │
│  • SHAP sensor contribution plot                         │
│  • Optional: mobile alert via Telegram/MQTT             │
└─────────────────────────────────────────────────────────┘
```

---

## Phased Execution Roadmap

### Phase 1 — Data Collection Infrastructure (Week 1)
- [ ] Build Python `serial_logger.py` — reads serial port, saves to `sensor_data.csv`
- [ ] Add real-time wall-clock timestamp (replace `millis()` with Python-side UTC time)  
- [ ] Test hardware: collect 1-hour continuous sensor log (fresh food in container)
- [ ] Collect 48-hour log of food going stale (chicken/egg/banana)
- [ ] Download Vision Dataset #2 (Kaggle: sriram, 13,599 images)

### Phase 2 — Vision Pipeline (Week 2)
- [ ] Set up data loader with train/val/test splits (80/10/10)
- [ ] Fine-tune ResNet50 → target: >90% fresh/rotten accuracy
- [ ] Implement Grad-CAM visualization
- [ ] Optional: try MobileViTv2 for lighter model

### Phase 3 — Sensor ML Pipeline (Week 3)
- [ ] EDA on collected sensor CSV: plot temp, humidity, NH₃ over time
- [ ] Train binary classifier (XGBoost/Random Forest) on sensor features
- [ ] Train LSTM for temporal shelf-life regression
- [ ] Evaluate: accuracy, F1, MAE on shelf-life prediction

### Phase 4 — Fusion + XAI (Week 4)
- [ ] Build Cross-Attention fusion (or simpler: feature concatenation + MLP)
- [ ] End-to-end training with both branches
- [ ] SHAP for sensor branch, Grad-CAM for vision branch
- [ ] Build Streamlit dashboard: live sensor feed + image analysis result

### Phase 5 — Demo + Report (Week 5)
- [ ] Record live demo video
- [ ] Benchmark: Sensor-only vs Vision-only vs Fused model
- [ ] Write report / presentation with accuracy tables

---

## Open Questions (Need Your Input)

> [!IMPORTANT]
> **Q1: Do you have the hardware set up and working?**  
> Is the Arduino connected, MQ135 + DHT11 wired, and serial data actually coming through? This determines whether Phase 1 can start immediately.

> [!IMPORTANT]
> **Q2: Camera — what are you using?**  
> ESP32-CAM, Raspberry Pi Camera, USB webcam, or your phone? This changes Phase 2 significantly.

> [!IMPORTANT]
> **Q3: Food types to test?**  
> The synopsis doesn't specify. Recommended: **chicken, egg, banana, tomato** — these produce measurable NH₃ during spoilage within 24–48 hours at room temperature (useful for your lab timeline).

> [!NOTE]
> **Q4: Do you have a Kaggle account?**  
> Needed to download dataset #2. If yes, I'll write the downloader script. If not, we use HuggingFace mirror (no login needed).
