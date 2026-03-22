"""
Task C: Conditional GAN (cGAN) – Fashion-MNIST
Deep Learning Assignment 2 - IS4412-2
Generates class-conditioned 28x28 grayscale images
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Directories ──────────────────────────────────────────────────────────────
os.makedirs("outputs/plots",       exist_ok=True)
os.makedirs("outputs/models",      exist_ok=True)
os.makedirs("outputs/gan_samples", exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
LATENT_DIM  = 100
EMBED_DIM   = 32
NUM_CLASSES = 10
BATCH_SIZE  = 128
LR_G        = 2e-4
LR_D        = 1e-4
BETAS       = (0.5, 0.999)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['T-shirt','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle Boot']

print(f"Using device: {DEVICE}")

# ── Data ─────────────────────────────────────────────────────────────────────
def get_loader():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),   # output in [-1, 1]
    ])
    dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=tf)
    return torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ── Generator ────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.fc        = nn.Linear(LATENT_DIM + EMBED_DIM, 256 * 7 * 7)

        self.deconv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7 → 14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 14 → 28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        emb = self.label_emb(labels)              # (B, EMBED_DIM)
        x   = torch.cat([noise, emb], dim=1)      # (B, LATENT+EMBED)
        x   = self.fc(x).view(-1, 256, 7, 7)
        return self.deconv(x)                      # (B, 1, 28, 28)

# ── Discriminator ─────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.proj      = nn.Linear(EMBED_DIM, 28 * 28)

        def sn_conv(in_ch, out_ch, stride=2):
            return nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=False))

        self.net = nn.Sequential(
            sn_conv(2, 64),                        # 2 channels: img + label map
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, img, labels):
        emb       = self.label_emb(labels)                     # (B, EMBED_DIM)
        label_map = self.proj(emb).view(-1, 1, 28, 28)        # (B, 1, 28, 28)
        x         = torch.cat([img, label_map], dim=1)        # (B, 2, 28, 28)
        return self.net(x)

# ── Weight Init ───────────────────────────────────────────────────────────────
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ── Sample Grid ───────────────────────────────────────────────────────────────
@torch.no_grad()
def save_samples(G, epoch, fixed_noise, fixed_labels):
    G.eval()
    imgs = G(fixed_noise, fixed_labels).cpu()          # (-1, 1)
    imgs = (imgs + 1) / 2                              # (0, 1)
    grid = torchvision.utils.make_grid(imgs, nrow=10, padding=2, normalize=False)
    plt.figure(figsize=(14, 2))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f'Epoch {epoch} – one sample per class')
    plt.tight_layout()
    plt.savefig(f'outputs/gan_samples/epoch_{epoch:03d}.png', dpi=150)
    plt.close()
    G.train()

# ── Training ──────────────────────────────────────────────────────────────────
def train(epochs, save_every):
    loader = get_loader()

    G = Generator().to(DEVICE);     G.apply(weights_init)
    D = Discriminator().to(DEVICE); D.apply(weights_init)

    opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=BETAS)
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=BETAS)
    criterion = nn.BCEWithLogitsLoss()

    # Fixed noise for consistent visualisation
    fixed_noise  = torch.randn(10, LATENT_DIM, device=DEVICE)
    fixed_labels = torch.arange(10, device=DEVICE)

    g_losses, d_losses = [], []

    for epoch in range(1, epochs + 1):
        ep_g, ep_d, n_batches = 0, 0, 0

        for real_imgs, real_labels in tqdm(loader, leave=False,
                                           desc=f"Epoch {epoch}/{epochs}"):
            real_imgs   = real_imgs.to(DEVICE)
            real_labels = real_labels.to(DEVICE)
            bsz         = real_imgs.size(0)

            # ── One-sided label smoothing for real targets
            real_targets = torch.empty(bsz, 1, device=DEVICE).uniform_(0.85, 0.95)
            fake_targets = torch.zeros(bsz, 1, device=DEVICE)

            # ── Train Discriminator (×1) ──────────────────────────────────
            opt_D.zero_grad()

            # real
            d_real = D(real_imgs, real_labels)
            loss_d_real = criterion(d_real, real_targets)

            # fake
            noise      = torch.randn(bsz, LATENT_DIM, device=DEVICE)
            fake_labels = torch.randint(0, NUM_CLASSES, (bsz,), device=DEVICE)
            fake_imgs  = G(noise, fake_labels).detach()
            d_fake     = D(fake_imgs, fake_labels)
            loss_d_fake = criterion(d_fake, fake_targets)

            loss_D = (loss_d_real + loss_d_fake) / 2
            loss_D.backward()
            opt_D.step()

            # ── Train Generator (×2) ─────────────────────────────────────
            loss_G_total = 0
            for _ in range(2):
                opt_G.zero_grad()
                noise      = torch.randn(bsz, LATENT_DIM, device=DEVICE)
                fake_labels = torch.randint(0, NUM_CLASSES, (bsz,), device=DEVICE)
                fake_imgs  = G(noise, fake_labels)
                g_out      = D(fake_imgs, fake_labels)
                loss_G     = criterion(g_out, torch.ones(bsz, 1, device=DEVICE))
                loss_G.backward()
                opt_G.step()
                loss_G_total += loss_G.item()

            ep_g += loss_G_total / 2
            ep_d += loss_D.item()
            n_batches += 1

        avg_g = ep_g / n_batches
        avg_d = ep_d / n_batches
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        print(f"Epoch {epoch:03d}/{epochs} | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")

        if epoch % save_every == 0 or epoch == epochs:
            save_samples(G, epoch, fixed_noise, fixed_labels)
            torch.save(G.state_dict(), f'outputs/models/cGAN_G_ep{epoch}.pth')
            torch.save(D.state_dict(), f'outputs/models/cGAN_D_ep{epoch}.pth')

    # ── Plot losses ──────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch'); plt.ylabel('BCE Loss')
    plt.title('cGAN Training Losses')
    plt.legend(); plt.tight_layout()
    plt.savefig('outputs/plots/C_gan_losses.png', dpi=150)
    plt.close()
    print("Saved → outputs/plots/C_gan_losses.png")

    # ── Final sample grid ────────────────────────────────────────────────────
    G.eval()
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flatten()):
        with torch.no_grad():
            noise = torch.randn(1, LATENT_DIM, device=DEVICE)
            lbl   = torch.tensor([i], device=DEVICE)
            img   = G(noise, lbl).cpu().squeeze()
            img   = (img + 1) / 2                  # rescale to [0,1]
        ax.imshow(img, cmap='gray')
        ax.set_title(CLASS_NAMES[i], fontsize=8)
        ax.axis('off')
    plt.suptitle('cGAN Final Samples – One Per Class')
    plt.tight_layout()
    plt.savefig('outputs/plots/C_final_samples.png', dpi=150)
    plt.close()
    print("Saved → outputs/plots/C_final_samples.png")

    # ── Progression collage ──────────────────────────────────────────────────
    sample_epochs = [e for e in [1, 10, 20, 30, 40, epochs]
                     if os.path.exists(f'outputs/gan_samples/epoch_{e:03d}.png')]
    if sample_epochs:
        fig, axes = plt.subplots(1, len(sample_epochs), figsize=(4*len(sample_epochs), 3))
        if len(sample_epochs) == 1:
            axes = [axes]
        for ax, ep in zip(axes, sample_epochs):
            img = plt.imread(f'outputs/gan_samples/epoch_{ep:03d}.png')
            ax.imshow(img); ax.set_title(f'Epoch {ep}'); ax.axis('off')
        plt.suptitle('cGAN Sample Progression')
        plt.tight_layout()
        plt.savefig('outputs/plots/C_progression.png', dpi=150)
        plt.close()
        print("Saved → outputs/plots/C_progression.png")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=60)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()
    train(args.epochs, args.save_every)
