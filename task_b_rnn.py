"""
Task B: RNN / LSTM / GRU – Airline Passengers Time-Series
Deep Learning Assignment 2 - IS4412-2
Dataset: International Airline Passengers (1949–1960), 144 monthly records
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import urllib.request
import math
import argparse
from tqdm import tqdm

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Directories ──────────────────────────────────────────────────────────────
os.makedirs("outputs/plots",  exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
SEQ_LEN    = 12
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT    = 0.2
EPOCHS     = 150
LR         = 1e-3
BATCH_SIZE = 16
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Load Airline Passengers Dataset ──────────────────────────────────────────
def load_data():
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
           "airline-passengers.csv")
    path = "data/airline-passengers.csv"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(path):
        print("Downloading Airline Passengers dataset...")
        urllib.request.urlretrieve(url, path)

    data = []
    with open(path) as f:
        next(f)                              # skip header
        for line in f:
            _, val = line.strip().split(',')
            data.append(float(val))
    return np.array(data, dtype=np.float32)

def make_sequences(series, seq_len):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def get_datasets():
    raw = load_data()                        # shape (144,)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(raw.reshape(-1, 1)).flatten()

    split  = int(len(scaled) * 0.75)
    train  = scaled[:split]
    test   = scaled[split:]

    X_tr, y_tr = make_sequences(train, SEQ_LEN)
    X_te, y_te = make_sequences(test,  SEQ_LEN)

    # shapes → (N, SEQ_LEN, 1)
    X_tr = torch.tensor(X_tr).unsqueeze(-1)
    X_te = torch.tensor(X_te).unsqueeze(-1)
    y_tr = torch.tensor(y_tr).unsqueeze(-1)
    y_te = torch.tensor(y_te).unsqueeze(-1)

    train_ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    test_ds  = torch.utils.data.TensorDataset(X_te, y_te)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, X_te, y_te, scaler

# ── Model ─────────────────────────────────────────────────────────────────────
class RecurrentModel(nn.Module):
    def __init__(self, cell_type='lstm'):
        super().__init__()
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell_type.lower()]
        self.rnn = rnn_cls(
            input_size=1,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])        # last time-step

# ── Training ──────────────────────────────────────────────────────────────────
def train_model(cell_type):
    train_loader, test_loader, X_te, y_te, scaler = get_datasets()

    model     = RecurrentModel(cell_type).to(DEVICE)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5, verbose=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {cell_type.upper()} | Params: {total_params:,}")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # ── train
        model.train()
        ep_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
        ep_loss /= len(train_loader)

        # ── validate
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item()
            val_loss /= len(test_loader)

        scheduler.step(val_loss)
        train_losses.append(ep_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       f'outputs/models/{cell_type.upper()}_best.pth')

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{EPOCHS} | "
                  f"Train Loss: {ep_loss:.6f} | Val Loss: {val_loss:.6f}")

    # ── Metrics on test set
    model.load_state_dict(torch.load(f'outputs/models/{cell_type.upper()}_best.pth'))
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_te.to(DEVICE)).cpu().numpy()

    preds_orig = scaler.inverse_transform(preds_scaled)
    y_orig     = scaler.inverse_transform(y_te.numpy())

    rmse = math.sqrt(mean_squared_error(y_orig, preds_orig))
    mae  = mean_absolute_error(y_orig, preds_orig)
    r2   = r2_score(y_orig, preds_orig)

    print(f"\n{'='*50}")
    print(f"Results – {cell_type.upper()}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  R²   : {r2:.4f}")

    return train_losses, val_losses, preds_orig, y_orig, rmse, mae, r2, total_params

# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_losses(results):
    plt.figure(figsize=(10, 5))
    for name, (tr, vl, *_) in results.items():
        plt.plot(tr, label=f'{name} Train')
        plt.plot(vl, label=f'{name} Val', linestyle='--')
    plt.title('Task B – Training & Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Huber Loss')
    plt.legend(); plt.tight_layout()
    plt.savefig('outputs/plots/B_loss_curves.png', dpi=150)
    plt.close()
    print("Saved → outputs/plots/B_loss_curves.png")

def plot_predictions(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, (_, _, preds, actual, *_)) in zip(axes, results.items()):
        ax.plot(actual, label='Actual', color='black')
        ax.plot(preds,  label='Predicted')
        ax.set_title(f'{name} Predictions'); ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/plots/B_predictions.png', dpi=150)
    plt.close()
    print("Saved → outputs/plots/B_predictions.png")

def plot_comparison(results):
    names  = list(results.keys())
    rmses  = [v[4] for v in results.values()]
    maes   = [v[5] for v in results.values()]
    r2s    = [v[6] for v in results.values()]
    params = [v[7] for v in results.values()]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].bar(x, rmses, color=['#e74c3c','#3498db','#2ecc71'])
    axes[0].set_xticks(x); axes[0].set_xticklabels(names)
    axes[0].set_title('RMSE (lower = better)')

    axes[1].bar(x, r2s,  color=['#e74c3c','#3498db','#2ecc71'])
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_title('R² Score (higher = better)')

    axes[2].bar(x, params, color=['#e74c3c','#3498db','#2ecc71'])
    axes[2].set_xticks(x); axes[2].set_xticklabels(names)
    axes[2].set_title('Parameter Count')

    plt.tight_layout()
    plt.savefig('outputs/plots/B_comparison.png', dpi=150)
    plt.close()
    print("Saved → outputs/plots/B_comparison.png")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["rnn", "lstm", "gru", "all"],
                        help="Which model to train")
    args = parser.parse_args()

    models_to_run = ["rnn", "lstm", "gru"] if args.model == "all" else [args.model]

    results = {}
    for cell in models_to_run:
        tr_l, vl_l, preds, actual, rmse, mae, r2, params = train_model(cell)
        results[cell.upper()] = (tr_l, vl_l, preds, actual, rmse, mae, r2, params)

    if len(results) > 1:
        plot_losses(results)
        plot_predictions(results)
        plot_comparison(results)
        print("\n── Summary ──────────────────────────────────────")
        print(f"{'Model':<10} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Params':>10}")
        for name, vals in results.items():
            print(f"{name:<10} {vals[4]:>8.2f} {vals[5]:>8.2f} "
                  f"{vals[6]:>8.4f} {vals[7]:>10,}")
    else:
        cell = list(results.keys())[0]
        tr_l, vl_l, preds, actual, *_ = results[cell]
        plt.figure(figsize=(10,4))
        plt.plot(tr_l, label='Train Loss')
        plt.plot(vl_l, label='Val Loss')
        plt.title(f'{cell} Loss Curves'); plt.legend()
        plt.savefig(f'outputs/plots/B_loss_curves_{cell}.png', dpi=150)
        plt.close()
