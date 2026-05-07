import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# Define model architecture (must match training script)
import torch.nn as nn
class GasSensorNN(nn.Module):
    def __init__(self, input_dim):
        super(GasSensorNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

def parse_feature(val):
    try:
        if isinstance(val, str) and ':' in val:
            return float(val.split(':')[1])
        return float(val)
    except:
        return 0.0

print("[*] Loading dataset to fit the Scaler...")
df = pd.read_csv("../../dataset/gas_sensor_data.csv")
X_df = df.iloc[:, :-1].map(parse_feature).fillna(0.0)
X = X_df.values
y = df.iloc[:, -1].values

scaler = StandardScaler()
scaler.fit(X)

print("[*] Loading trained PyTorch model...")
input_dim = X.shape[1]
model = GasSensorNN(input_dim)
model.load_state_dict(torch.load("../../models/saved_weights/sensor_model.pth", weights_only=True))
model.eval()

# Select samples
ammonia_sample = X[0:1] # Just take row 0
other_sample = X[-1:]   # Take the last row

true_label_1 = y[0]
true_label_2 = y[-1]

print(f"\n--- TEST 1: Real Gas Sample (True Class: {true_label_1}) ---")
sample_scaled = scaler.transform(ammonia_sample)
tensor_input = torch.FloatTensor(sample_scaled)
with torch.no_grad():
    prediction = model(tensor_input).item()
confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
result = "SPOILED (Ammonia Detected)" if prediction > 0.5 else "SAFE (Other Gas)"
print(f"Raw Output: {prediction:.4f}")
print(f"Prediction: {result} | Confidence: {confidence:.2f}%")

print(f"\n--- TEST 2: Real Gas Sample (True Class: {true_label_2}) ---")
sample_scaled = scaler.transform(other_sample)
tensor_input = torch.FloatTensor(sample_scaled)
with torch.no_grad():
    prediction = model(tensor_input).item()
confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
result = "SPOILED (Ammonia Detected)" if prediction > 0.5 else "SAFE (Other Gas)"
print(f"Raw Output: {prediction:.4f}")
print(f"Prediction: {result} | Confidence: {confidence:.2f}%")

# Compute Full Metrics
print("\n--- FULL DATASET EVALUATION ---")
y_binary = (y == 6).astype(int)
all_scaled = scaler.transform(X)
all_tensor = torch.FloatTensor(all_scaled)

with torch.no_grad():
    all_preds = model(all_tensor).numpy()
    
all_preds_binary = (all_preds >= 0.5).astype(int)

acc = accuracy_score(y_binary, all_preds_binary)
prec = precision_score(y_binary, all_preds_binary, zero_division=0)
rec = recall_score(y_binary, all_preds_binary, zero_division=0)
f1 = f1_score(y_binary, all_preds_binary, zero_division=0)

print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")
