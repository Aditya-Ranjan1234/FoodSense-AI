"""
train_mlp.py — FoodSense Gas Sensor Model (FIXED)
===================================================
ROOT CAUSE of the previous 100% accuracy bug:
  - We used UCI dataset ID 270 which is a REGRESSION dataset (concentration values).
  - The target column contained libsvm format strings like '1:15596.16' — NOT class labels.
  - `y == 6` was NEVER True → model predicted ALL samples as "Not Ammonia" (class 0).
  - A dataset where 100% of samples are class 0 → trivially 100% accuracy, 0% F1.

FIX:
  - Download UCI Gas Sensor Array Drift dataset (ID 224) directly as a zip.
  - This is the CORRECT multiclass gas classification dataset (6 gas types, 13910 samples).
  - Gas class mapping: 1=Ethanol, 2=Ethylene, 3=Ammonia, 4=Acetaldehyde, 5=Acetone, 6=Toluene
  - We do BINARY classification: Ammonia (class 3) vs All Others.
  - Apply class_weight='balanced' equivalent via pos_weight in BCELoss to handle imbalance.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import requests
import zipfile
import io

# ─── Config ─────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join("..", "..", "dataset")
WEIGHTS_DIR = os.path.join("..", "saved_weights")
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Device: {device}")

# ─── 1. Download Gas Sensor Array Drift Dataset ────────────────────────────
DRIFT_CSV = os.path.join(DATASET_DIR, "gas_sensor_drift.csv")

if not os.path.exists(DRIFT_CSV):
    print("[*] Downloading UCI Gas Sensor Array Drift Dataset (direct)...")
    url = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
    r = requests.get(url, timeout=60)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    # Each batch is a separate dat file (libsvm format); we merge all 10 batches
    all_rows = []
    for fname in sorted(z.namelist()):
        if fname.endswith(".dat"):
            print(f"  Parsing: {fname}")
            lines = z.read(fname).decode("utf-8").strip().split("\n")
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                label = int(parts[0])  # gas class 1-6
                features = []
                for kv in parts[1:]:
                    _, val = kv.split(":")
                    features.append(float(val))
                all_rows.append([label] + features)
    
    cols = ["class"] + [f"f{i}" for i in range(len(all_rows[0]) - 1)]
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(DRIFT_CSV, index=False)
    print(f"[*] Saved {len(df)} samples to {DRIFT_CSV}")
else:
    print(f"[*] Loading cached dataset from {DRIFT_CSV}")
    df = pd.read_csv(DRIFT_CSV)

# ─── 2. Prepare Binary Classification ─────────────────────────────────────
# Class 3 = Ammonia (the gas food spoilage releases)
print("\n[*] Class distribution:")
print(df["class"].value_counts().sort_index())
print("    (Class 3 = Ammonia — our SPOILAGE target)")

X = df.drop(columns=["class"]).values.astype(np.float32)
y_raw = df["class"].values
y = (y_raw == 3).astype(np.float32)  # 1 = Ammonia/Spoiled, 0 = Safe
n_pos = y.sum()
n_neg = (1 - y).sum()
print(f"\n[*] Ammonia samples: {int(n_pos)}  |  Other samples: {int(n_neg)}")
print(f"[*] Imbalance ratio: {n_neg/n_pos:.1f}:1 → using pos_weight to compensate")

# ─── 3. Train/Test Split & Scaling ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

X_tr_t = torch.from_numpy(X_train_s).to(device)
y_tr_t = torch.from_numpy(y_train).unsqueeze(1).to(device)
X_te_t = torch.from_numpy(X_test_s).to(device)
y_te_t = torch.from_numpy(y_test).unsqueeze(1).to(device)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ─── 4. Model ─────────────────────────────────────────────────────────────
class GasSensorMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = GasSensorMLP(X_train_s.shape[1]).to(device)

# Weight the positive class (Ammonia) by the imbalance ratio so model can't cheat
pos_weight = torch.tensor([n_neg / n_pos]).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Use Sigmoid separately since BCEWithLogitsLoss needs raw logits
# Rebuild without final Sigmoid for training
class GasSensorMLPLogits(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, 1)   # raw logit — sigmoid applied in loss
        )
    def forward(self, x):
        return self.net(x)

model = GasSensorMLPLogits(X_train_s.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# ─── 5. Train ─────────────────────────────────────────────────────────────
print(f"\n[*] Training MLP for {EPOCHS} epochs on {device}...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for bX, by in train_ld:
        optimizer.zero_grad()
        logits = model(bX)
        loss   = criterion(logits, by)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {epoch_loss/len(train_ld):.4f}")

# ─── 6. Evaluate ──────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    logits  = model(X_te_t)
    probs   = torch.sigmoid(logits).cpu().numpy().flatten()
    y_pred  = (probs >= 0.5).astype(int)
    y_true  = y_te_t.cpu().numpy().flatten().astype(int)

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)
cm   = confusion_matrix(y_true, y_pred)

print("\n" + "="*50)
print("   HONEST MODEL EVALUATION (Test Set 20%)")
print("="*50)
print(f"  Accuracy:  {acc*100:.2f}%")
print(f"  Precision: {prec*100:.2f}%  (of predicted Ammonia, how many truly were?)")
print(f"  Recall:    {rec*100:.2f}%  (of actual Ammonia, how many did we catch?)")
print(f"  F1-Score:  {f1*100:.2f}%  (balance of precision & recall)")
print(f"\n  Confusion Matrix:")
print(f"  (rows=actual, cols=predicted)  [0=Safe, 1=Ammonia]")
print(f"  {cm}")
print("\n" + classification_report(y_true, y_pred, target_names=["Safe","Ammonia"]))

# ─── 7. Save weights ──────────────────────────────────────────────────────
weights_path = os.path.join(WEIGHTS_DIR, "sensor_mlp.pth")
torch.save(model.state_dict(), weights_path)
print(f"[*] Model saved to {weights_path}")
