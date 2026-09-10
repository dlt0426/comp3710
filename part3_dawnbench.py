"""
COMP3710 Lab 2 - Part 3.2: DAWNBench CIFAR-10 fast training (PyTorch)

Trains a DAWNBench-style residual network on CIFAR-10 to >=94% test
accuracy as fast as possible, using:
  - a custom ResNet (built from scratch, NOT a torchvision pre-built model)
  - mixed precision (torch.cuda.amp) for Tensor Core speedups on the A100
  - OneCycle learning-rate schedule (the key to few-epoch convergence)
  - standard CIFAR augmentation (random crop + horizontal flip) + normalisation

Method is based on David Page's cifar10-fast (the DAWNBench reference that
reaches 94% in ~79s on a V100). Shakes' JAX solution ports the same idea.

Run on Rangpur with sbatch (A100). Reports total training time and final
test accuracy so you can check the DAWNBench targets:
  - >90% accuracy, fast                (requirement 1)
  - runs train + inference on cluster  (requirement 2)
  - >=94% accuracy, <=360s (V100 ref)  (requirement 3)
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True          # autotune conv algorithms
    torch.backends.cuda.matmul.allow_tf32 = True   # allow TF32 on A100
    torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(0)

# ----------------------------------------------------------------------
# 1. Data: CIFAR-10 with augmentation + normalisation
# ----------------------------------------------------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])
test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# Download to a local ./data folder. On the cluster this needs one-off
# internet access on the login node (see notes), or a pre-downloaded copy.
train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_tf)
test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_tf)

BATCH = 512
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False,
                         num_workers=4, pin_memory=True)

# ----------------------------------------------------------------------
# 2. DAWNBench-style ResNet (built from scratch)
#    A compact residual net tuned for 32x32 CIFAR images.
# ----------------------------------------------------------------------
def conv_bn(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Residual(nn.Module):
    """Two 3x3 conv-bn-relu layers with a skip connection (identity add)."""
    def __init__(self, c):
        super().__init__()
        self.conv1 = conv_bn(c, c)
        self.conv2 = conv_bn(c, c)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class FastResNet(nn.Module):
    """
    prep -> layer1(+res) -> layer2 -> layer3(+res) -> pool -> linear.
    This is the David Page 'ResNet9' topology used for DAWNBench.
    """
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
        self.scale = 0.125   # logit scaling, helps this net train stably

    def forward(self, x):
        x = self.prep(x)
        x = self.res1(self.pool1(self.layer1(x)))
        x = self.pool2(self.layer2(x))
        x = self.res3(self.pool3(self.layer3(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x) * self.scale


model = FastResNet().to(device)
if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)  # faster convs on Tensor Cores
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params/1e6:.2f}M")

# ----------------------------------------------------------------------
# 3. Loss, optimiser, OneCycle schedule, AMP scaler
# ----------------------------------------------------------------------
EPOCHS = 24
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0, momentum=0.9,
                            weight_decay=5e-4, nesterov=True)

steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.4, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
    pct_start=0.3, div_factor=10, final_div_factor=100)

scaler = GradScaler()

# ----------------------------------------------------------------------
# 4. Train
# ----------------------------------------------------------------------
def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast():
                out = model(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total


print(f"\nTraining for {EPOCHS} epochs, batch size {BATCH}...")
train_start = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():                       # mixed precision forward
            out = model(xb)
            loss = criterion(out, yb)
        scaler.scale(loss).backward()          # scaled backward for fp16 stability
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item()

    if device.type == "cuda":
        torch.cuda.synchronize()
    acc = evaluate()
    elapsed = time.time() - train_start
    print(f"Epoch {epoch+1:2d}/{EPOCHS}  "
          f"loss={running_loss/steps_per_epoch:.3f}  "
          f"test_acc={acc*100:.2f}%  elapsed={elapsed:.1f}s")

total_time = time.time() - train_start
final_acc = evaluate()

print("\n--- DAWNBench Result ---")
print(f"Final test accuracy: {final_acc*100:.2f}%")
print(f"Total training time : {total_time:.1f} s")
print(f"Reached 90%+ : {final_acc >= 0.90}")
print(f"Reached 94%+ : {final_acc >= 0.94}")
print(f"Under 360s   : {total_time <= 360}")

# ----------------------------------------------------------------------
# 5. Single inference pass demo (requirement 2)
# ----------------------------------------------------------------------
model.eval()
xb, yb = next(iter(test_loader))
xb = xb.to(device, memory_format=torch.channels_last)
with torch.no_grad(), autocast():
    preds = model(xb).argmax(1).cpu()
print("\nSample inference on one test batch:")
print("  predicted:", preds[:10].tolist())
print("  ground truth:", yb[:10].tolist())
