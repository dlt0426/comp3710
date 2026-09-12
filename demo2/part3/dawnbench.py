"""Part 3.2: CIFAR-10 with a from-scratch ResNet-18.

Train:     python dawnbench.py --mode train
Inference: python dawnbench.py --mode inference
Demo:      python dawnbench.py --mode demo
Demo loads a checkpoint, runs inference, then trains one complete epoch.
Accuracy and speed targets must be verified on Rangpur; they are not guaranteed.
"""
import argparse
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
import torchvision


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """[2, 2, 2, 2] BasicBlocks; CIFAR stem: 3x3, stride 1, no max pool.

    The 18 counted layers are the stem, 16 block convolutions, and classifier.
    Projection shortcuts are not included in the conventional depth count.
    """
    def __init__(self, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer1 = self._stage(64, 64, 1)
        self.layer2 = self._stage(64, 128, 2)
        self.layer3 = self._stage(128, 256, 2)
        self.layer4 = self._stage(256, 512, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, n_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    @staticmethod
    def _stage(in_channels, channels, stride):
        return nn.Sequential(BasicBlock(in_channels, channels, stride),
                             BasicBlock(channels, channels))

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
        return self.fc(self.pool(x).flatten(1))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_data(root, train, device):
    ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
    x = torch.tensor(ds.data, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255
    mean = x.new_tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = x.new_tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    return (x - mean) / std, torch.tensor(ds.targets, dtype=torch.long, device=device), ds.classes


def batches(x, y, batch_size, train=False):
    indices = torch.randperm(len(y), device=y.device) if train else torch.arange(len(y), device=y.device)
    for start in range(0, len(y), batch_size):
        idx = indices[start:start + batch_size]
        xb = x[idx]
        if train:
            # Each image gets its own random crop and horizontal flip.
            padded = F.pad(xb, (4, 4, 4, 4), mode="reflect")
            count = len(idx)
            offsets = torch.randint(9, (count, 2), device=x.device)
            rows = offsets[:, 0, None, None] + torch.arange(32, device=x.device)[None, :, None]
            cols = offsets[:, 1, None, None] + torch.arange(32, device=x.device)[None, None, :]
            xb = padded.permute(0, 2, 3, 1)[torch.arange(count, device=x.device)[:, None, None], rows, cols].permute(0, 3, 1, 2)
            flip = torch.rand(count, 1, 1, 1, device=x.device) < 0.5
            xb = torch.where(flip, xb.flip(3), xb)
        yield xb.contiguous(memory_format=torch.channels_last), y[idx]


def train_epoch(model, x, y, optimizer, scaler, scheduler, args, device):
    model.train()
    total_loss = torch.zeros((), device=device)
    sync(device)
    start = time.perf_counter()
    for xb, yb in batches(x, y, args.batch_size, train=True):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp):
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.detach() * len(yb)
    sync(device)
    return total_loss.item() / len(y), time.perf_counter() - start


@torch.no_grad()
def evaluate(model, x, y, classes, args, device, show=False):
    model.eval()
    correct = torch.zeros((), device=device, dtype=torch.long)
    examples = None
    sync(device)
    start = time.perf_counter()
    for xb, yb in batches(x, y, args.batch_size):
        with torch.autocast(device_type=device.type, enabled=args.amp):
            preds = model(xb).argmax(1)
        correct += (preds == yb).sum()
        if examples is None:
            examples = (preds[:10], yb[:10])
    sync(device)
    duration = time.perf_counter() - start
    if show:
        for pred, truth in zip(*(a.cpu().tolist() for a in examples)):
            print(f"Predicted: {classes[pred]:10s} Ground truth: {classes[truth]}")
    return correct.item() / len(y), duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "inference", "demo"], default="train")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).with_name("resnet18_cifar10.pt"))
    parser.add_argument("--data-dir", default=str(Path(__file__).with_name("data")))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--demo-lr", type=float, default=0.001)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = device.type == "cuda" and not args.no_amp
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}; mixed precision: {args.amp}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    if args.mode != "train" and not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Run --mode train first.")
    model = ResNet18().to(device=device, memory_format=torch.channels_last)
    if args.mode != "train":
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    test_x, test_y, classes = load_data(args.data_dir, False, device)
    if args.mode != "train":
        acc, seconds = evaluate(model, test_x, test_y, classes, args, device, show=True)
        print(f"Inference: test accuracy={acc:.2%}, time={seconds:.2f}s")
        if args.mode == "inference":
            return
    train_x, train_y, _ = load_data(args.data_dir, True, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.demo_lr if args.mode == "demo" else args.lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    if args.mode == "demo":
        loss, seconds = train_epoch(model, train_x, train_y, optimizer, scaler, None, args, device)
        print(f"Demo: one complete training epoch ({len(train_y)} images), loss={loss:.4f}, time={seconds:.2f}s")
        print("Demo finished; saved checkpoint was not modified.")
        return
    steps = (len(train_y) + args.batch_size - 1) // args.batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
        epochs=args.epochs, steps_per_epoch=steps, pct_start=0.2)
    sync(device)
    start = time.perf_counter()
    training_seconds = 0.0
    for epoch in range(1, args.epochs + 1):
        loss, seconds = train_epoch(model, train_x, train_y, optimizer, scaler, scheduler, args, device)
        training_seconds += seconds
        print(f"Epoch {epoch}/{args.epochs}: loss={loss:.4f}, time={seconds:.2f}s", flush=True)
    acc, inference_seconds = evaluate(model, test_x, test_y, classes, args, device, show=True)
    elapsed = time.perf_counter() - start
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "architecture": "ResNet18-CIFAR-stem", "test_accuracy": acc,
                "training_seconds": training_seconds, "train_and_eval_seconds": elapsed}, args.checkpoint)
    print(f"Final test accuracy: {acc:.2%}")
    print(f"Training only: {training_seconds:.2f}s; training + final evaluation: {elapsed:.2f}s")
    print("Timings exclude dataset loading and checkpoint saving.")
    print(f">90%: {acc > 0.90}; >=94%: {acc >= 0.94}; train + evaluation <=360s: {elapsed <= 360}")
    print(f"Saved: {args.checkpoint}")


if __name__ == "__main__":
    main()
