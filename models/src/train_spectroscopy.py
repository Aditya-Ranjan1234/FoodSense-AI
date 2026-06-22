"""
train_spectroscopy.py — FoodSense Spectroscopy Models
=====================================================
Implementations for:
1. Virtual Spectrometer Calibration Model (AI Calibration):
   - Maps raw [R, G, B, LDR] inputs to [Wavelength, Absorbance].
2. Pigment Degradation Classifier:
   - Maps key absorbances [Abs_430, Abs_540, Abs_660] to [Fresh, Ripe, Spoiled].
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# ─── Config ─────────────────────────────────────────────────────────────────
SPECTROSCOPY_DIR = os.path.join("..", "..", "spectroscopy")
WEIGHTS_DIR = os.path.join("..", "saved_weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Device: {device}")

# ─── Data Preprocessing for Calibration Model ──────────────────────────────
CSV_PATH = os.path.join(SPECTROSCOPY_DIR, "data.csv")

def parse_rgb(rgb_str):
    try:
        rgb_str = rgb_str.replace("(", "").replace(")", "")
        return [float(x.strip()) for x in rgb_str.split(",")]
    except:
        return [0.0, 0.0, 0.0]

print(f"[*] Loading spectroscopy dataset from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Clean wavelength column if it has 'nm'
df['Wavelength (nm)'] = df['Wavelength (nm)'].astype(str).str.replace(' nm', '').astype(float)

# Parse RGB
rgb_parsed = df['RGB Values'].apply(parse_rgb).tolist()
df_rgb = pd.DataFrame(rgb_parsed, columns=['R', 'G', 'B'])
df = pd.concat([df, df_rgb], axis=1)

# Features: R, G, B, LDR Value
# Targets: Wavelength (nm), Absorbance (Abs)
X_cal = df[['R', 'G', 'B', 'LDR Value']].values.astype(np.float32)
y_cal = df[['Wavelength (nm)', 'Absorbance (Abs)']].values.astype(np.float32)

# Scale
scaler_X_cal = MinMaxScaler()
scaler_y_cal = MinMaxScaler()

X_cal_scaled = scaler_X_cal.fit_transform(X_cal).astype(np.float32)
y_cal_scaled = scaler_y_cal.fit_transform(y_cal).astype(np.float32)

# Train/Test Split
X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(
    X_cal_scaled, y_cal_scaled, test_size=0.2, random_state=42
)

# Loaders
train_cal_ds = TensorDataset(torch.from_numpy(X_train_cal).to(device), torch.from_numpy(y_train_cal).to(device))
test_cal_ds = TensorDataset(torch.from_numpy(X_test_cal).to(device), torch.from_numpy(y_test_cal).to(device))
train_cal_loader = DataLoader(train_cal_ds, batch_size=32, shuffle=True)

# ─── 1. Virtual Spectrometer Calibration Model Architecture ────────────────
class CalibrationMLP(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super(CalibrationMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Train Calibration Model
cal_model = CalibrationMLP().to(device)
criterion_cal = nn.MSELoss()
optimizer_cal = optim.Adam(cal_model.parameters(), lr=1e-3)

print("\n[*] Training Virtual Spectrometer Calibration Model...")
epochs_cal = 100
for epoch in range(epochs_cal):
    cal_model.train()
    total_loss = 0
    for batch_X, batch_y in train_cal_loader:
        optimizer_cal.zero_grad()
        preds = cal_model(batch_X)
        loss = criterion_cal(preds, batch_y)
        loss.backward()
        optimizer_cal.step()
        total_loss += loss.item() * len(batch_X)
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:03d} | Train MSE: {total_loss/len(X_train_cal):.6f}")

# Evaluation
cal_model.eval()
with torch.no_grad():
    test_inputs = torch.from_numpy(X_test_cal).to(device)
    test_targets_scaled = torch.from_numpy(y_test_cal).to(device)
    test_preds_scaled = cal_model(test_inputs).cpu().numpy()
    
# Unscale targets & predictions
test_targets = scaler_y_cal.inverse_transform(y_test_cal)
test_preds = scaler_y_cal.inverse_transform(test_preds_scaled)

mse_w = mean_squared_error(test_targets[:, 0], test_preds[:, 0])
r2_w = r2_score(test_targets[:, 0], test_preds[:, 0])
mse_a = mean_squared_error(test_targets[:, 1], test_preds[:, 1])
r2_a = r2_score(test_targets[:, 1], test_preds[:, 1])

print("\n[+] Calibration Model Results:")
print(f"  Wavelength  -> MSE: {mse_w:.4f} | R²: {r2_w:.4f}")
print(f"  Absorbance  -> MSE: {mse_a:.4f} | R²: {r2_a:.4f}")

# Save Calibration Model
torch.save(cal_model.state_dict(), os.path.join(WEIGHTS_DIR, "calibration_model.pth"))
print(f"[+] Saved Calibration Model to: {os.path.join(WEIGHTS_DIR, 'calibration_model.pth')}")


# ─── 2. Pigment Degradation Classifier (Simulation & Training) ─────────────
print("\n[*] Simulating Pigment Degradation Dataset (Fresh/Ripe/Spoiled)...")

def generate_synthetic_pigment_data(n_samples=1200):
    """
    Generates synthetic absorbance profiles at key wavelengths:
    - Abs_430 (Chlorophyll absorption peak in blue)
    - Abs_540 (Carotenoids & browning compounds in green/yellow)
    - Abs_660 (Chlorophyll absorption peak in red)
    
    Labels:
    - 0 = Fresh (High Chlorophyll, low browning): High 430 & 660, Low 540
    - 1 = Ripe (Medium Chlorophyll, medium browning): Medium 430 & 660, Medium 540
    - 2 = Spoiled (Low Chlorophyll, high browning/melanins): Low 430 & 660, High 540
    """
    np.random.seed(42)
    samples = []
    labels = []
    
    # Class 0: Fresh
    for _ in range(n_samples // 3):
        a430 = np.random.normal(1.2, 0.15)
        a540 = np.random.normal(0.2, 0.05)
        a660 = np.random.normal(1.0, 0.12)
        samples.append([a430, a540, a660])
        labels.append(0)
        
    # Class 1: Ripe
    for _ in range(n_samples // 3):
        a430 = np.random.normal(0.6, 0.1)
        a540 = np.random.normal(0.4, 0.08)
        a660 = np.random.normal(0.5, 0.08)
        samples.append([a430, a540, a660])
        labels.append(1)
        
    # Class 2: Spoiled
    for _ in range(n_samples // 3):
        a430 = np.random.normal(0.2, 0.05)
        a540 = np.random.normal(0.9, 0.12)
        a660 = np.random.normal(0.15, 0.04)
        samples.append([a430, a540, a660])
        labels.append(2)
        
    return np.array(samples, dtype=np.float32), np.array(labels, dtype=np.int64)

X_pig, y_pig = generate_synthetic_pigment_data()

# Scale Features
scaler_pig = MinMaxScaler()
X_pig_scaled = scaler_pig.fit_transform(X_pig).astype(np.float32)

# Train/Test Split
X_train_pig, X_test_pig, y_train_pig, y_test_pig = train_test_split(
    X_pig_scaled, y_pig, test_size=0.2, random_state=42, stratify=y_pig
)

# Loaders
train_pig_ds = TensorDataset(torch.from_numpy(X_train_pig).to(device), torch.from_numpy(y_train_pig).to(device))
train_pig_loader = DataLoader(train_pig_ds, batch_size=32, shuffle=True)

class PigmentClassifierNN(nn.Module):
    def __init__(self, input_dim=3, num_classes=3):
        super(PigmentClassifierNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Train Pigment Classifier
pig_model = PigmentClassifierNN().to(device)
criterion_pig = nn.CrossEntropyLoss()
optimizer_pig = optim.Adam(pig_model.parameters(), lr=1e-3)

print("[*] Training Pigment Degradation Classifier...")
epochs_pig = 80
for epoch in range(epochs_pig):
    pig_model.train()
    total_loss = 0
    for batch_X, batch_y in train_pig_loader:
        optimizer_pig.zero_grad()
        preds = pig_model(batch_X)
        loss = criterion_pig(preds, batch_y)
        loss.backward()
        optimizer_pig.step()
        total_loss += loss.item() * len(batch_X)
        
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:02d} | Train Loss: {total_loss/len(X_train_pig):.6f}")

# Evaluation
pig_model.eval()
with torch.no_grad():
    test_inputs = torch.from_numpy(X_test_pig).to(device)
    logits = pig_model(test_inputs)
    preds = torch.argmax(logits, dim=1).cpu().numpy()

acc_pig = accuracy_score(y_test_pig, preds)
print("\n[+] Pigment Classification Results:")
print(f"  Accuracy: {acc_pig*100:.2f}%")
print(classification_report(y_test_pig, preds, target_names=["Fresh", "Ripe", "Spoiled"]))

# Save Pigment Model
torch.save(pig_model.state_dict(), os.path.join(WEIGHTS_DIR, "pigment_model.pth"))
print(f"[+] Saved Pigment Model to: {os.path.join(WEIGHTS_DIR, 'pigment_model.pth')}")
