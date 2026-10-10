import os, glob, random
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

CLASSES = {"NC": 0, "AD": 1}

def patient_id(path):
    """'.../388206_78.jpeg' -> '388206'"""
    return os.path.basename(path).split("_")[0]

def collect(root):
    items = []
    for name, label in CLASSES.items():
        for p in sorted(glob.glob(os.path.join(root, name, "*.jpeg"))):
            items.append((p, label))
    return items

def split_by_patient(items, val_frac=0.15, seed=42):
    by_patient = defaultdict(list)
    for p, y in items:
        by_patient[(patient_id(p), y)].append((p, y))

    rng = random.Random(seed)
    train, val = [], []
    for label in CLASSES.values():
        pids = sorted(k for k in by_patient if k[1] == label)
        rng.shuffle(pids)
        n_val = int(len(pids) * val_frac)
        for i, k in enumerate(pids):
            (val if i < n_val else train).extend(by_patient[k])
    return train, val

class ADNIDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items, self.transform = items, transform
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, y, patient_id(path)
