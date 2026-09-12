"""
COMP3710 Lab 2 - Part 4 Task 1: Variational Autoencoder of OASIS brain MRI (PyTorch)

Trains a VAE on OASIS brain MRI slices and visualises the learned manifold,
both by:
  (a) decoding a 2D grid of latent points into a montage of brains, and
  (b) a UMAP projection of encoded latent vectors (if umap-learn is installed).

Data (on Rangpur):
  /home/groups/comp3710/OASIS/keras_png_slices_train  (MRI, 256x256 grey)

Run on Rangpur with sbatch (A100).
"""

import os
import glob
import time
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

torch.manual_seed(0)

DATA_ROOT = "/home/groups/comp3710/OASIS"
IMG_SIZE = 128          # downsize 256->128 to keep the VAE light and fast
LATENT_DIM = 2          # 2D latent so the manifold can be visualised directly

# ----------------------------------------------------------------------
# 1. Dataset (images only, no labels needed for a VAE)
# ----------------------------------------------------------------------
class OASISImages(Dataset):
    def __init__(self, img_dir, size):
        self.paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).resize((self.size, self.size))
        img = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0)   # (1, H, W)


train_ds = OASISImages(os.path.join(DATA_ROOT, "keras_png_slices_train"), IMG_SIZE)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=2, pin_memory=True)
print("Training images:", len(train_ds))

# ----------------------------------------------------------------------
# 2. VAE model
# ----------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=2, size=128):
        super().__init__()
        self.size = size
        # encoder: 128 -> 64 -> 32 -> 16 -> 8
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(inplace=True),    # 64
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),   # 32
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),  # 16
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True), # 8
        )
        self.feat = size // 16          # 128/16 = 8
        self.flat = 256 * self.feat * self.feat
        self.fc_mu = nn.Linear(self.flat, latent_dim)
        self.fc_logvar = nn.Linear(self.flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),  # 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),   # 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),    # 64
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),              # 128
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    # z = μ + ε·σ（backpropagation works through this reparameterisation trick）
    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, self.feat, self.feat)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE(LATENT_DIM, IMG_SIZE).to(device)
print("VAE parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

# ----------------------------------------------------------------------
# 3. VAE loss = reconstruction (BCE) + KL divergence
# Their trade-off yields a latent space that is both faithful in
# reconstruction and continuous and smooth.
# ----------------------------------------------------------------------
def vae_loss(recon, x, mu, logvar):
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld, bce, kld


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------------------------------------------------
# 4. Train
# Feed the image into the model → reconstruction and the distribution parameters
#  → compute the loss → backpropagate → update the parameters, using the Adam optimizer.
# ----------------------------------------------------------------------
EPOCHS = 20
print(f"\nTraining VAE for {EPOCHS} epochs...")
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for xb in train_loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(xb)
        loss, bce, kld = vae_loss(recon, xb, mu, logvar)
        loss.backward()
        optimizer.step()
        running += loss.item()
    elapsed = time.time() - start
    print(f"Epoch {epoch+1:2d}/{EPOCHS}  loss/img={running/len(train_ds):.2f}  ({elapsed:.0f}s)")

torch.save(model.state_dict(), "vae_oasis.pt")
print("Saved vae_oasis.pt")

# ----------------------------------------------------------------------
# 5a. Reconstruction sanity check
# ----------------------------------------------------------------------
model.eval()
xb = next(iter(train_loader))[:8].to(device)
with torch.no_grad():
    recon, _, _ = model(xb)
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    axes[0, i].imshow(xb[i, 0].cpu(), cmap="gray"); axes[0, i].axis("off")
    axes[1, i].imshow(recon[i, 0].cpu(), cmap="gray"); axes[1, i].axis("off")
axes[0, 0].set_ylabel("original")
axes[1, 0].set_ylabel("recon")
plt.suptitle("VAE reconstructions (top: original, bottom: reconstructed)")
plt.tight_layout()
plt.savefig("vae_reconstructions.png", dpi=120)
plt.close()
print("Saved vae_reconstructions.png")

# ----------------------------------------------------------------------
# 5b. Manifold visualisation by decoding a 2D grid of latent points
# ----------------------------------------------------------------------
n = 20
grid = np.linspace(-3, 3, n)   # sample latent space over +/-3 sigma
canvas = np.zeros((n * IMG_SIZE, n * IMG_SIZE))
model.eval()
with torch.no_grad():
    for i, yi in enumerate(grid):
        for j, xi in enumerate(grid):
            z = torch.tensor([[xi, yi]], dtype=torch.float32, device=device)
            img = model.decode(z)[0, 0].cpu().numpy()
            canvas[i * IMG_SIZE:(i + 1) * IMG_SIZE,
                   j * IMG_SIZE:(j + 1) * IMG_SIZE] = img

plt.figure(figsize=(12, 12))
plt.imshow(canvas, cmap="gray")
plt.title("VAE manifold: brains decoded across the 2D latent grid")
plt.axis("off")
plt.tight_layout()
plt.savefig("vae_manifold.png", dpi=120)
plt.close()
print("Saved vae_manifold.png")

# ----------------------------------------------------------------------
# 5c. Latent scatter (and UMAP if available)
# ----------------------------------------------------------------------
mus = []
model.eval()
with torch.no_grad():
    for k, xb in enumerate(train_loader):
        xb = xb.to(device)
        mu, _ = model.encode(xb)
        mus.append(mu.cpu().numpy())
        if k > 30:   # a few thousand points is plenty
            break
mus = np.concatenate(mus, axis=0)

plt.figure(figsize=(8, 8))
plt.scatter(mus[:, 0], mus[:, 1], s=3, alpha=0.4)
plt.title("VAE latent space (encoded means)")
plt.xlabel("z1"); plt.ylabel("z2")
plt.tight_layout()
plt.savefig("vae_latent_scatter.png", dpi=120)
plt.close()
print("Saved vae_latent_scatter.png")

print("\nDone.")
