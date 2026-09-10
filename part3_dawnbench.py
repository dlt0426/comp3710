"""
COMP3710 Lab 2 - Part 3.2: DAWNBench CIFAR-10 fast training (PyTorch)

Trains a DAWNBench-style residual network on CIFAR-10 to >=94% test
accuracy as fast as possible.

Key speed technique (this version): the whole CIFAR-10 dataset is
pre-loaded into GPU memory once, and augmentation (random crop + flip)
is done on the GPU. This removes the DataLoader / CPU->GPU transfer
bottleneck, so each epoch is only a few seconds on an A100.

Other ingredients:
  - custom ResNet (built from scratch, NOT a torchvision pre-built model)
  - mixed precision (torch.cuda.amp) for Tensor Core speedups
  - OneCycle learning-rate schedule (few-epoch convergence)

Method based on David Page's cifar10-fast (94% in ~79s on a V100).

Run on Rangpur with sbatch (A100).
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as T

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(0)

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616])

# ----------------------------------------------------------------------
# 1. Load CIFAR-10 fully into GPU memory (normalised, as tensors)
# ----------------------------------------------------------------------
def load_split_to_gpu(train):
    ds = torchvision.datasets.CIFAR10(root="./data", train=train, download=True)
    x = torch.tensor(ds.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # (N,3,32,32)
    y = torch.tensor(ds.targets, dtype=torch.long)
    # normalise
    x = (x - CIFAR_MEAN.view(1, 3, 1, 1)) / CIFAR_STD.view(1, 3, 1, 1)
    return x.to(device), y.to(device)


train_x, train_y = load_split_to_gpu(train=True)
test_x, test_y = load_split_to_gpu(train=False)
# pre-pad the training images by 4 on each side so random crop is a cheap slice
train_x_pad = F.pad(train_x, (4, 4, 4, 4), mode="reflect")
print("Train:", tuple(train_x.shape), " Test:", tuple(test_x.shape))

N_TRAIN = train_x.shape[0]
BATCH = 512


def get_train_batches():
    """Yield GPU batches with random-crop + horizontal-flip augmentation."""
    perm = torch.randperm(N_TRAIN, device=device)
    for i in range(0, N_TRAIN - BATCH + 1, BATCH):
        idx = perm[i:i + BATCH]
        xb = train_x_pad[idx]                          # (B,3,40,40)
        # random crop 32x32
        ox = torch.randint(0, 9, (1,)).item()
        oy = torch.randint(0, 9, (1,)).item()
        xb = xb[:, :, oy:oy + 32, ox:ox + 32]
        # random horizontal flip (whole batch)
        if torch.rand(1).item() < 0.5:
            xb = torch.flip(xb, dims=[3])
        yield xb.contiguous(memory_format=torch.channels_last), train_y[idx]


def get_test_batches():
    for i in range(0, test_x.shape[0], 1024):
        xb = test_x[i:i + 1024].contiguous(memory_format=torch.channels_last)
        yield xb, test_y[i:i + 1024]


steps_per_epoch = N_TRAIN // BATCH

# ----------------------------------------------------------------------
# 2. DAWNBench-style ResNet (built from scratch)
# ----------------------------------------------------------------------
def conv_bn(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = conv_bn(c, c)
        self.conv2 = conv_bn(c, c)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class FastResNet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.prep = conv_bn(3, 64)
        self.layer1 = conv_bn(64, 128)
        self.pool1 = nn.MaxPool2d(2)
        self.res1 = Residual(128)
        self.layer2 = conv_bn(128, 256)
        self.pool2 = nn.MaxPool2d(2)
        self.layer3 = conv_bn(256, 512)
        self.pool3 = nn.MaxPool2d(2)
        self.res3 = Residual(512)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(512, n_classes)
        self.scale = 0.125

    def forward(self, x):
        x = self.prep(x)
        x = self.res1(self.pool1(self.layer1(x)))
        x = self.pool2(self.layer2(x))
        x = self.res3(self.pool3(self.layer3(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x) * self.scale


model = FastResNet().to(device).to(memory_format=torch.channels_last)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ----------------------------------------------------------------------
# 3. Loss, optimiser, schedule, AMP
# ----------------------------------------------------------------------
EPOCHS = 26
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0, momentum=0.9,
                            weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.5, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
    pct_start=0.25, div_factor=8, final_div_factor=200)
scaler = GradScaler()


def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in get_test_batches():
            with autocast():
                out = model(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total


# ----------------------------------------------------------------------
# 4. Train
# ----------------------------------------------------------------------
EVAL_FROM = EPOCHS - 4
print(f"\nTraining for {EPOCHS} epochs, batch size {BATCH}...")
train_start = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for xb, yb in get_train_batches():
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            out = model(xb)
            loss = criterion(out, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item()

    torch.cuda.synchronize()
    elapsed = time.time() - train_start
    if epoch + 1 > EVAL_FROM:
        acc = evaluate()
        print(f"Epoch {epoch+1:2d}/{EPOCHS}  loss={running_loss/steps_per_epoch:.3f}  "
              f"test_acc={acc*100:.2f}%  elapsed={elapsed:.1f}s")
    else:
        print(f"Epoch {epoch+1:2d}/{EPOCHS}  loss={running_loss/steps_per_epoch:.3f}  "
              f"(eval skipped)  elapsed={elapsed:.1f}s")

total_time = time.time() - train_start
final_acc = evaluate()

print("\n--- DAWNBench Result ---")
print(f"Final test accuracy: {final_acc*100:.2f}%")
print(f"Total training time : {total_time:.1f} s")
print(f"Reached 90%+ : {final_acc >= 0.90}")
print(f"Reached 94%+ : {final_acc >= 0.94}")
print(f"Under 360s   : {total_time <= 360}")

# ----------------------------------------------------------------------
# 5. Single inference pass demo
# ----------------------------------------------------------------------
model.eval()
xb, yb = next(iter(get_test_batches()))
with torch.no_grad(), autocast():
    preds = model(xb).argmax(1).cpu()
print("\nSample inference on one test batch:")
print("  predicted:", preds[:10].tolist())
print("  ground truth:", yb[:10].cpu().tolist())
