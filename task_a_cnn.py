"""
Task A: CNN Image Classification on CIFAR-10
Deep Learning Assignment 2 - IS4412-2
Implements: Custom CNN vs ResNet-18 Transfer Learning
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import argparse

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Directories ──────────────────────────────────────────────────────────────
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']
BATCH_SIZE  = 128
EPOCHS      = 20
LR          = 1e-3
WEIGHT_DECAY= 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Data ─────────────────────────────────────────────────────────────────────
def get_loaders():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.247,  0.243,  0.261)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                                shuffle=True,  num_workers=2)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=BATCH_SIZE,
                                                shuffle=False, num_workers=2)
    return train_loader, test_loader

# ── Custom CNN ────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        return self.block(x)

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   64),   # 32→16
            ConvBlock(64,  128),  # 16→8
            ConvBlock(128, 256),  # 8→4
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ── ResNet-18 Transfer Learning ───────────────────────────────────────────────
def get_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False                      # freeze backbone
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 10),             # only this trains
    )
    return model

# ── Training & Evaluation ─────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type):
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels

# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_curves(train_losses, val_losses, train_accs, val_accs, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses,   label='Val Loss')
    ax1.set_title(f'{name} – Loss Curves'); ax1.legend()
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs,   label='Val Acc')
    ax2.set_title(f'{name} – Accuracy Curves'); ax2.legend()
    plt.tight_layout()
    plt.savefig(f'outputs/plots/A_training_curves_{name}.png', dpi=150)
    plt.close()
    print(f"Saved training curves → outputs/plots/A_training_curves_{name}.png")

def plot_confusion(preds, labels, name):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix – {name}')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'outputs/plots/A_confusion_matrix_{name}.png', dpi=150)
    plt.close()
    print(f"Saved confusion matrix → outputs/plots/A_confusion_matrix_{name}.png")

# ── Main ──────────────────────────────────────────────────────────────────────
def run(model_name):
    train_loader, test_loader = get_loaders()

    if model_name == "custom":
        model = CustomCNN().to(DEVICE)
        label = "CustomCNN"
    else:
        model = get_resnet18().to(DEVICE)
        label = "ResNet18_TL"

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {label}")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler()

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        vl_loss, vl_acc, _, _ = evaluate(model, test_loader, criterion)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), f'outputs/models/{label}_best.pth')

    # Final evaluation
    model.load_state_dict(torch.load(f'outputs/models/{label}_best.pth'))
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion)
    print(f"\n{'='*50}")
    print(f"Final Test Accuracy ({label}): {test_acc*100:.2f}%")
    print(classification_report(labels, preds, target_names=CLASSES))

    plot_curves(train_losses, val_losses, train_accs, val_accs, label)
    plot_confusion(preds, labels, label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model",   default="custom",
                        choices=["custom", "resnet18"])
    args = parser.parse_args()
    run(args.model)
