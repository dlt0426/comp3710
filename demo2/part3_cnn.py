"""
COMP3710 Lab 2 - Part 3.1: CNN classifier for the LFW face dataset (PyTorch)

Reuses the LFW loading / train-test split from Part 2, but instead of
PCA + Random Forest, trains a small CNN end-to-end:
    - two 3x3 convolution layers, 32 filters each
    - dense (fully connected) layers for classification
    - Adam optimiser + cross-entropy loss (sparse categorical cross entropy)

The goal is to beat the Eigenfaces (PCA + Random Forest) accuracy from Part 2.

Run on Rangpur with sbatch (A100).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

torch.manual_seed(42)
np.random.seed(42)

# ----------------------------------------------------------------------
# 1. Load LFW (reused from Part 2). Use .images to keep the 2D structure
#    that a CNN needs, NOT the flattened .data used for PCA.
# ----------------------------------------------------------------------
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

X = lfw_people.images          # shape (n_samples, h, w), values already in [0, 1]
Y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
n_samples, h, w = lfw_people.images.shape

print("Total dataset size:")
print("  n_samples :", n_samples)
print("  image size: %d x %d" % (h, w))
print("  n_classes :", n_classes)

# Check the value range. If already in [0, 1] no normalisation is needed.
print("X_min:", X.min(), " X_max:", X.max())

# ----------------------------------------------------------------------
# 2. Train / test split (same split as Part 2)
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42
)

# Convolution layers expect 4D tensors: (N, channels, H, W).
# Faces are greyscale, so channels = 1. np.newaxis inserts that axis.
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
print("X_train shape:", X_train.shape)   # (N, 1, h, w)
print("X_test  shape:", X_test.shape)

# ----------------------------------------------------------------------
# 3. Wrap in tensors / DataLoaders
# ----------------------------------------------------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False
)

# ----------------------------------------------------------------------
# 4. CNN model: two 3x3 conv layers (32 filters each) + dense layers
# ----------------------------------------------------------------------
class FaceCNN(nn.Module):
    def __init__(self, h, w, n_classes):
        super().__init__()
        # Conv block 1: 1 -> 32 channels, 3x3 kernel, padding keeps H,W
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Conv block 2: 32 -> 32 channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)          # halves H and W each time

        # After two pool ops the spatial size is roughly h/4 x w/4.
        # Compute it exactly so the dense layer input is correct.
        h_out = h // 2 // 2
        w_out = w // 2 // 2
        self.flat_dim = 32 * h_out * w_out

        self.fc1 = nn.Linear(self.flat_dim, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # -> (N, 32, h/2, w/2)
        x = self.pool(self.relu(self.conv2(x)))   # -> (N, 32, h/4, w/4)
        x = x.view(x.size(0), -1)                 # flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                           # raw logits (no softmax here)
        return x


model = FaceCNN(h, w, n_classes).to(device)
print(model)

# ----------------------------------------------------------------------
# 5. Loss + optimiser
#    CrossEntropyLoss == (sparse) categorical cross entropy: it takes
#    integer class labels and applies softmax internally.
# ----------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------------------------------------------------
# 6. Training loop
# ----------------------------------------------------------------------
EPOCHS = 30
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1:2d}/{EPOCHS}  loss = {epoch_loss:.4f}")

# ----------------------------------------------------------------------
# 7. Evaluate on the test set
# ----------------------------------------------------------------------
model.eval()
all_preds = []
all_true = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = model(xb)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.numpy())

all_preds = np.array(all_preds)
all_true = np.array(all_true)
accuracy = np.mean(all_preds == all_true)

print("\n--- CNN Test Performance ---")
print("Total testing:", len(all_true))
print("Total correct:", int(np.sum(all_preds == all_true)))
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(all_true, all_preds, target_names=target_names))

# ----------------------------------------------------------------------
# 8. Save the training loss curve
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.title("CNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("cnn_loss.png", dpi=120)
plt.close()
print("Saved cnn_loss.png")
