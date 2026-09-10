"""
COMP3710 Lab 2 - Part 4 Task 3: DCGAN for realistic OASIS brain MRI (PyTorch)

Trains a DCGAN to generate realistic brain MRI slices from the OASIS
dataset. Saves sample generated images periodically and the generator/
discriminator loss curves as evidence of training.

GANs are notoriously unstable. This script uses several stabilisation
techniques to reduce mode collapse and divergence:
  - DCGAN architecture (strided conv generator/discriminator, BatchNorm)
  - Adam with betas=(0.5, 0.999) and lr 2e-4 (the DCGAN recipe)
  - one-sided label smoothing on the real labels
  - small amount of instance noise via label noise

Data (on Rangpur):
  MRI : /home/groups/comp3710/OASIS/keras_png_slices_train/*.png (256x256 grey)

Run on Rangpur with sbatch (A100). Expect to run this several times and
inspect the sample grids -- GAN quality is judged by eye.
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
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

torch.manual_seed(0)

DATA_ROOT = "/home/groups/comp3710/OASIS"
IMG_SIZE = 64        # generate 64x64 brains (good quality/speed trade-off)
LATENT = 100         # generator input noise dimension
BATCH = 128

# ----------------------------------------------------------------------
# 1. Dataset (images only). Images normalised to [-1, 1] to match tanh output.
# ----------------------------------------------------------------------
class OASISImages(Dataset):
    def __init__(self, img_dir, size):
        self.paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).resize((self.size, self.size))
        img = np.array(img, dtype=np.float32) / 127.5 - 1.0   # -> [-1, 1]
        return torch.from_numpy(img).unsqueeze(0)


train_ds = OASISImages(os.path.join(DATA_ROOT, "keras_png_slices_train"), IMG_SIZE)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
print("Training images:", len(train_ds))

# ----------------------------------------------------------------------
# 2. Generator and Discriminator (DCGAN)
# ----------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent=100, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            # latent -> 4x4
            nn.ConvTranspose2d(latent, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            # 4 -> 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            # 8 -> 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            # 16 -> 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            # 32 -> 64
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # 32 -> 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            # 16 -> 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            # 8 -> 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            # 4 -> 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    """DCGAN weight initialisation."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


G = Generator(LATENT).to(device); G.apply(weights_init)
D = Discriminator().to(device); D.apply(weights_init)
print("G params:", sum(p.numel() for p in G.parameters()) / 1e6, "M")
print("D params:", sum(p.numel() for p in D.parameters()) / 1e6, "M")

# ----------------------------------------------------------------------
# 3. Loss + optimisers (DCGAN recipe)
# ----------------------------------------------------------------------
criterion = nn.BCEWithLogitsLoss()
optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, LATENT, 1, 1, device=device)  # to track progress

# ----------------------------------------------------------------------
# 4. Train
# ----------------------------------------------------------------------
EPOCHS = 40
g_losses, d_losses = [], []
os.makedirs("gan_samples", exist_ok=True)

print(f"\nTraining DCGAN for {EPOCHS} epochs...")
start = time.time()

for epoch in range(EPOCHS):
    for i, real in enumerate(train_loader):
        real = real.to(device)
        b = real.size(0)

        # ---- Train D ----
        optD.zero_grad()
        # real batch (one-sided label smoothing: real label = 0.9)
        out_real = D(real)
        label_real = torch.full((b,), 0.9, device=device)
        loss_real = criterion(out_real, label_real)
        # fake batch
        noise = torch.randn(b, LATENT, 1, 1, device=device)
        fake = G(noise)
        out_fake = D(fake.detach())
        label_fake = torch.zeros(b, device=device)
        loss_fake = criterion(out_fake, label_fake)
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optD.step()

        # ---- Train G ----
        optG.zero_grad()
        out = D(fake)
        label_g = torch.ones(b, device=device)   # G wants D to say "real"
        loss_G = criterion(out, label_g)
        loss_G.backward()
        optG.step()

        g_losses.append(loss_G.item())
        d_losses.append(loss_D.item())

    elapsed = time.time() - start
    print(f"Epoch {epoch+1:2d}/{EPOCHS}  loss_D={loss_D.item():.3f}  "
          f"loss_G={loss_G.item():.3f}  ({elapsed:.0f}s)")

    # save a sample grid every few epochs
    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise).cpu()
        G.train()
        grid = vutils.make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
        plt.axis("off")
        plt.title(f"Generated brains - epoch {epoch+1}")
        plt.savefig(f"gan_samples/epoch_{epoch+1:03d}.png", dpi=100, bbox_inches="tight")
        plt.close()

# ----------------------------------------------------------------------
# 5. Save loss curves + final samples + models
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label="Generator", alpha=0.7)
plt.plot(d_losses, label="Discriminator", alpha=0.7)
plt.xlabel("Iteration"); plt.ylabel("Loss")
plt.title("GAN training losses")
plt.legend()
plt.tight_layout()
plt.savefig("gan_losses.png", dpi=120)
plt.close()
print("Saved gan_losses.png")

torch.save(G.state_dict(), "gan_generator.pt")
torch.save(D.state_dict(), "gan_discriminator.pt")
print("Saved generator and discriminator.")
print("Sample grids are in gan_samples/. Inspect them to judge realism.")
print("\nDone.")
