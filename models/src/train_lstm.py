"""
train_lstm.py — FoodSense Temporal Shelf-Life LSTM (IMPROVED)
==============================================================
Improvements over v1:
  - Added Temporal Attention layer → model learns WHICH timesteps matter most
  - LR Scheduler (ReduceLROnPlateau) to avoid stuck plateaus
  - Reports MAE, RMSE, and R² so metrics are comprehensive
  - Separate validation set for early stopping signal
  - Normalised target (hours) to [0,1] during training → better gradient flow
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

DATA_FILE   = os.path.join("..", "..", "dataset", "arduino_timeseries_log.csv")
WEIGHTS_DIR = os.path.join("..", "saved_weights")
SEQ_LEN     = 30      # 30 × 2-min readings = 1 hour of context
EPOCHS      = 150
BATCH_SIZE  = 64
LR          = 1e-3
os.makedirs(WEIGHTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Device: {device}")

# ─── 1. Data ──────────────────────────────────────────────────────────────
def generate_synthetic_arduino_data(path):
    """
    Synthetic 48-hour spoilage log matching foodsense_sensor_fusion.ino CSV output:
      timestamp_ms, temp, humidity, nh3, hours_remaining
    Replace this CSV with your real Arduino serial log to train on real hardware data.
    """
    print("[*] Generating 48-hour synthetic Arduino spoilage log...")
    n     = 1440  # 48h × 60min / 2min-interval
    t     = np.linspace(0, 48, n)
    temp  = 25.0 + 2*np.sin(t/4) + np.random.normal(0, 0.3, n)
    hum   = 60 + 25*(t/48)**1.5   + np.random.normal(0, 0.8, n)
    nh3   = 1.5 + 1.2*np.exp(t/20) + np.random.normal(0, 0.3, n)
    shelf = np.maximum(0, 48 - t)
    ms    = np.arange(n) * 120_000
    df = pd.DataFrame({"timestamp_ms": ms, "temp": temp, "humidity": hum,
                       "nh3": nh3, "hours_remaining": shelf})
    df.to_csv(path, index=False)
    return df

if not os.path.exists(DATA_FILE):
    df = generate_synthetic_arduino_data(DATA_FILE)
else:
    df = pd.read_csv(DATA_FILE)
    print(f"[*] Loaded {len(df)} rows from {DATA_FILE}")

feat_cols = ["temp", "humidity", "nh3"]
tgt_col   = "hours_remaining"

feat_scaler = MinMaxScaler()
tgt_scaler  = MinMaxScaler()

feats  = feat_scaler.fit_transform(df[feat_cols].values).astype(np.float32)
target = tgt_scaler.fit_transform(df[[tgt_col]].values).astype(np.float32)

X, y = [], []
for i in range(len(feats) - SEQ_LEN):
    X.append(feats[i: i+SEQ_LEN])
    y.append(target[i + SEQ_LEN])
X = np.array(X);  y = np.array(y)

split      = int(0.8 * len(X))
val_split  = int(0.9 * len(X))
X_tr, y_tr = X[:split], y[:split]
X_va, y_va = X[split:val_split], y[split:val_split]
X_te, y_te = X[val_split:], y[val_split:]

def to_loader(Xa, ya, shuffle=True):
    ds = TensorDataset(torch.from_numpy(Xa).to(device), torch.from_numpy(ya).to(device))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_ld = to_loader(X_tr, y_tr, shuffle=True)
val_ld   = to_loader(X_va, y_va, shuffle=False)

# ─── 2. Model with Temporal Attention ─────────────────────────────────────
class TemporalAttention(nn.Module):
    """Learns which timesteps in the sequence are most important."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: [batch, seq_len, hidden_dim]
        scores  = self.attn(lstm_out)          # [batch, seq_len, 1]
        weights = torch.softmax(scores, dim=1) # normalise across time
        context = (weights * lstm_out).sum(dim=1)  # weighted sum → [batch, hidden]
        return context

class SpoilageLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.3)
        self.attention = TemporalAttention(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()   # output in [0,1] since target is normalised
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        ctx    = self.attention(out)
        return self.head(ctx)

model     = SpoilageLSTM().to(device)
criterion = nn.HuberLoss()    # more robust to outliers than plain MSE
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10)

# ─── 3. Train ─────────────────────────────────────────────────────────────
print(f"\n[*] Training Attention-LSTM for {EPOCHS} epochs on {device}...")
best_val_loss = float("inf")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for bX, by in train_ld:
        optimizer.zero_grad()
        pred = model(bX)
        loss = criterion(pred, by)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for bX, by in val_ld:
            val_loss += criterion(model(bX), by).item()
    val_loss /= max(len(val_ld), 1)
    scheduler.step(val_loss)

    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss/len(train_ld):.5f} | Val Loss: {val_loss:.5f}")

# ─── 4. Evaluate on Test Set ──────────────────────────────────────────────
model.eval()
preds_norm, trues_norm = [], []
with torch.no_grad():
    for bX, by in to_loader(X_te, y_te, shuffle=False):
        preds_norm.extend(model(bX).cpu().numpy().flatten())
        trues_norm.extend(by.cpu().numpy().flatten())

preds_norm = np.array(preds_norm).reshape(-1, 1)
trues_norm = np.array(trues_norm).reshape(-1, 1)

# Inverse-transform back to real hours
preds_h = tgt_scaler.inverse_transform(preds_norm).flatten()
trues_h = tgt_scaler.inverse_transform(trues_norm).flatten()

mae  = np.mean(np.abs(preds_h - trues_h))
rmse = np.sqrt(np.mean((preds_h - trues_h) ** 2))
r2   = r2_score(trues_h, preds_h)

print("\n" + "="*50)
print("   LSTM EVALUATION (Test Set 10%)")
print("="*50)
print(f"  MAE   (Mean Abs Error):  {mae:.2f} hours")
print(f"  RMSE  (Root Mean Sq Er): {rmse:.2f} hours")
print(f"  R²    (Explained Var):   {r2:.4f}  (1.0 = perfect)")
print(f"\n  Interpretation:")
print(f"  → Model predicts shelf-life within ±{mae:.1f} hours on average.")
print(f"  → R² of {r2:.2f} means the model explains {r2*100:.1f}% of variance in spoilage timing.")

weights_path = os.path.join(WEIGHTS_DIR, "lstm_shelf_life.pth")
torch.save(model.state_dict(), weights_path)
print(f"\n[*] LSTM saved to {weights_path}")
