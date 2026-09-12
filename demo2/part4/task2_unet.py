"""
COMP3710 Lab 2 - Part 4 Task 2: UNet segmentation of OASIS brain MRI (PyTorch)

Segments OASIS brain MRI slices into 4 classes using a UNet with skip
connections. Targets Dice similarity coefficient (DSC) > 0.9 for every
label, with one-hot / categorical output as required.

Data (on Rangpur):
  MRI : /home/groups/comp3710/OASIS/keras_png_slices_train/case_XXX_slice_Y.nii.png
  SEG : /home/groups/comp3710/OASIS/keras_png_slices_seg_train/seg_XXX_slice_Y.nii.png
  (the seg file matches the MRI file with "case_" replaced by "seg_")

Segmentation label values {0, 85, 170, 255} are remapped to class
indices {0, 1, 2, 3}.

Run on Rangpur with an allocated GPU (A100):
  python task2_unet.py --mode train
  python task2_unet.py --mode inference --checkpoint unet_oasis.pt
Inference loads saved weights and only needs the test split.
"""

import argparse
from pathlib import Path
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

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
DATA_ROOT = "/home/groups/comp3710/OASIS"
N_CLASSES = 4
LABEL_MAP = {0: 0, 85: 1, 170: 2, 255: 3}   # remap pixel values to class ids

# ----------------------------------------------------------------------
# 1. Dataset
#    MRI file : case_XXX_slice_Y.nii.png  in keras_png_slices_<split>
#    SEG file : seg_XXX_slice_Y.nii.png   in keras_png_slices_seg_<split>
#    -> the seg path is the MRI basename with "case_" replaced by "seg_".
# ----------------------------------------------------------------------
class OASISDataset(Dataset):
    def __init__(self, img_dir, seg_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.seg_dir = seg_dir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        seg_name = os.path.basename(img_path).replace("case_", "seg_")
        seg_path = os.path.join(self.seg_dir, seg_name)

        img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
        seg = np.array(Image.open(seg_path))

        # remap label values -> class indices
        label = np.zeros_like(seg, dtype=np.int64)
        for v, c in LABEL_MAP.items():
            label[seg == v] = c

        img = torch.from_numpy(img).unsqueeze(0)      # (1, H, W)
        label = torch.from_numpy(label).long()        # (H, W)
        return img, label


def make_loader(split, batch, shuffle, data_root=DATA_ROOT):
    img_dir = os.path.join(data_root, f"keras_png_slices_{split}")
    seg_dir = os.path.join(data_root, f"keras_png_slices_seg_{split}")
    ds = OASISDataset(img_dir, seg_dir)
    if not len(ds):
        raise ValueError(f"No MRI PNG files found in {img_dir}")
    # this partition's compute nodes have few CPU cores, so keep workers low
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


# ----------------------------------------------------------------------
# 2. UNet
# ----------------------------------------------------------------------
def double_conv(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_classes=4, base=32):
        super().__init__()
        self.d1 = double_conv(1, base)
        self.d2 = double_conv(base, base * 2)
        self.d3 = double_conv(base * 2, base * 4)
        self.d4 = double_conv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = double_conv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.u4 = double_conv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = double_conv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = double_conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = double_conv(base * 2, base)

        self.out = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        b = self.bottleneck(self.pool(c4))

        x = self.u4(torch.cat([self.up4(b), c4], dim=1))
        x = self.u3(torch.cat([self.up3(x), c3], dim=1))
        x = self.u2(torch.cat([self.up2(x), c2], dim=1))
        x = self.u1(torch.cat([self.up1(x), c1], dim=1))
        return self.out(x)              # logits (N, n_classes, H, W)


# ----------------------------------------------------------------------
# 3. Dice loss + Dice metric (one-hot / categorical)
# ----------------------------------------------------------------------
def dice_loss(logits, target, eps=1e-6):
    probs = F.softmax(logits, dim=1)                        # (N, C, H, W)
    target_1h = F.one_hot(target, N_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * target_1h).sum(dims)
    denom = probs.sum(dims) + target_1h.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def dice_per_class(logits, target, eps=1e-6):
    preds = logits.argmax(1)                                # (N, H, W)
    dices = []
    for c in range(N_CLASSES):
        p = (preds == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dices.append(((2 * inter + eps) / (denom + eps)).item())
    return dices


# ----------------------------------------------------------------------
# 4. Train
# ----------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    totals = np.zeros(N_CLASSES)
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            totals += np.array(dice_per_class(out, yb))
            n += 1
    return totals / n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "inference"], default="train")
    parser.add_argument("--checkpoint", type=Path, default=Path("unet_oasis.pt"))
    parser.add_argument("--output", type=Path, default=Path("unet_segmentation.png"))
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.mode == "inference" and not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Train first or supply its path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(0)

    model = UNet(N_CLASSES).to(device)
    print("UNet parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    # Inference restores the original state_dict format, including BatchNorm buffers.
    # It skips training data, optimizer construction, and checkpoint writing.
    if args.mode == "inference":
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded {args.checkpoint}")
    test_loader = make_loader("test", args.batch_size, False, args.data_root)
    if args.mode == "train":
        train_loader = make_loader("train", args.batch_size, True, args.data_root)
        val_loader = make_loader("validate", args.batch_size, False, args.data_root)
        ce = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

        EPOCHS = args.epochs
        print(f"\nTraining UNet for {EPOCHS} epochs...")
        start = time.time()

        for epoch in range(EPOCHS):
            model.train()
            running = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = ce(out, yb) + dice_loss(out, yb)   # combined loss
                loss.backward()
                optimizer.step()
                running += loss.item()
            scheduler.step()

            val_dice = evaluate(model, val_loader, device)
            elapsed = time.time() - start
            print(f"Epoch {epoch+1:2d}/{EPOCHS}  loss={running/len(train_loader):.4f}  "
                  f"val DSC per class = [{', '.join(f'{d:.4f}' for d in val_dice)}]  "
                  f"mean={val_dice.mean():.4f}  ({elapsed:.0f}s)")

        # Save before plotting so trained weights survive a visualization failure.
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
        print(f"Saved {args.checkpoint}")

    # Both modes evaluate the test split and save MRI / truth / prediction panels.
    test_dice = evaluate(model, test_loader, device)
    print("\n--- Test set DSC per class ---")
    class_names = ["background", "label 1", "label 2", "label 3"]
    for name, d in zip(class_names, test_dice):
        print(f"  {name:12s}: DSC = {d:.4f}  {'PASS' if d > 0.9 else 'FAIL'}")
    print(f"  mean DSC : {test_dice.mean():.4f}")
    print(f"  All labels > 0.9: {bool((test_dice > 0.9).all())}")

    # ----------------------------------------------------------------------
    # 6. Visualise some segmentation results
    # ----------------------------------------------------------------------
    model.eval()
    xb, yb = next(iter(test_loader))
    xb = xb.to(device)
    with torch.no_grad():
        preds = model(xb).argmax(1).cpu().numpy()
    xb = xb.cpu().numpy()
    yb = yb.numpy()

    n_show = min(4, len(xb))
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show), squeeze=False)
    for i in range(n_show):
        axes[i, 0].imshow(xb[i, 0], cmap="gray"); axes[i, 0].set_title("MRI")
        axes[i, 1].imshow(yb[i], cmap="viridis", vmin=0, vmax=3); axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(preds[i], cmap="viridis", vmin=0, vmax=3); axes[i, 2].set_title("Prediction")
        for j in range(3):
            axes[i, j].axis("off")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=120)
    plt.close()
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
