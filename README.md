# CNN + RNN + GAN — Applied Deep Models
### Deep Learning Assignment 2 | IS4412-2 | AY 2025–2026

> **Student:** Shreyas | **GitHub:** [@shreyasrao172](https://github.com/shreyasrao172)  
> **Framework:** PyTorch 2.x | **Hardware:** CUDA GPU (fallback: CPU)  
> **Datasets:** CIFAR-10 · Airline Passengers · Fashion-MNIST

---

## Overview

This assignment implements and evaluates three core applied deep learning architectures:

| Task | Architecture | Dataset | Key Result |
|------|-------------|---------|------------|
| A | Custom CNN vs ResNet-18 (Transfer Learning) | CIFAR-10 | Custom ~83% / ResNet-18 ~88% accuracy |
| B | Vanilla RNN vs LSTM vs GRU | Airline Passengers | GRU RMSE ~20.5, R² = 0.968 |
| C | Conditional GAN (cGAN) | Fashion-MNIST | Recognisable garments by epoch 40 |

---

## Repository Structure

```
cnn-rnn-gan-assignment/
│
├── task_a_cnn.py           # Task A: CNN image classification
├── task_b_rnn.py           # Task B: RNN/LSTM/GRU time-series
├── task_c_cgan.py          # Task C: Conditional GAN
├── requirements.txt        # Python dependencies
│
├── data/                   # Auto-downloaded datasets
├── outputs/
│   ├── plots/              # Training curves, confusion matrices, GAN losses
│   ├── models/             # Saved model weights (.pth)
│   └── gan_samples/        # GAN generated images per epoch
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/shreyasrao172/cnn-rnn-gan-assignment
cd cnn-rnn-gan-assignment

# 2. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Task A — CNN Classification (CIFAR-10)

```bash
# Train Custom CNN
python task_a_cnn.py --dataset cifar10 --model custom

# Train ResNet-18 Transfer Learning
python task_a_cnn.py --dataset cifar10 --model resnet18
```

**Outputs:** `outputs/plots/A_training_curves_*.png`, `outputs/plots/A_confusion_matrix_*.png`

---

### Task B — RNN / LSTM / GRU (Airline Passengers)

```bash
# Train all three models and compare
python task_b_rnn.py --model all

# Train a specific model
python task_b_rnn.py --model lstm    # or --model gru / --model rnn
```

**Outputs:** `outputs/plots/B_loss_curves.png`, `B_predictions.png`, `B_comparison.png`

---

### Task C — Conditional GAN (Fashion-MNIST)

```bash
python task_c_cgan.py --epochs 60 --save_every 10
```

**Outputs:** `outputs/plots/C_gan_losses.png`, `C_final_samples.png`, `C_progression.png`  
Generated images per epoch → `outputs/gan_samples/`

---

## Results Summary

### Task A — CNN Classification

| Metric | Custom CNN | ResNet-18 (TL) |
|--------|-----------|----------------|
| Test Accuracy | ~83% | ~88% |
| Total Params | 2.4 M | 11.2 M |
| Trainable Params | 2.4 M | ~5 K |
| Train Time (GPU) | ~12 min | ~8 min |

### Task B — Time-Series Prediction

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Params |
|-------|--------|-------|------|--------|
| Vanilla RNN | ~31.4 | ~25.6 | 0.923 | ~33 K |
| LSTM | ~22.1 | ~17.8 | 0.961 | ~66 K |
| **GRU** | **~20.5** | **~16.2** | **0.968** | ~50 K |

### Task C — cGAN Generation

| Epoch | Sample Quality |
|-------|---------------|
| 1 | Mostly noise |
| 20 | Coarse class-conditional structure |
| 40–60 | Recognisable garments with clear class conditioning |

---

## Key Techniques

- **Batch Normalisation** — faster convergence, +4% accuracy (Task A)
- **Dropout (0.5)** — reduced overfitting in CNNs and RNNs
- **Gradient Clipping (norm=1.0)** — eliminated exploding gradients in Vanilla RNN
- **Spectral Normalisation on D** — prevented discriminator dominance in cGAN
- **Label Smoothing (0.85–0.95)** — delayed mode collapse in cGAN
- **CosineAnnealingLR** — prevented loss oscillation near convergence (Task A)
- **HuberLoss** — more stable RNN training vs MSE (Task B)

---

## References

1. Goodfellow et al., "Generative Adversarial Nets," NeurIPS, 2014.
2. He et al., "Deep Residual Learning for Image Recognition," CVPR, 2016.
3. Hochreiter & Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.
4. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder," EMNLP, 2014.
5. Mirza & Osindero, "Conditional Generative Adversarial Nets," arXiv:1411.1784, 2014.

---

## Declaration

This assignment is entirely my own original work. All references, datasets, pretrained models, and libraries used have been properly acknowledged.
