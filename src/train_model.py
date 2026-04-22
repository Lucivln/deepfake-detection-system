import os
import random
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "data/final"
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)


# ----------------------------
# TRANSFORMS
# ----------------------------

# 🔥 TRAIN (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
])

# ✅ VALIDATION (clean)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ----------------------------
# LOAD FILE PATHS
# ----------------------------
samples = []

for label, class_name in enumerate(["fake", "real"]):
    folder = os.path.join(DATA_DIR, class_name)
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        samples.append((path, label))


# ----------------------------
# GROUP BY VIDEO ID
# ----------------------------
groups = defaultdict(list)

for path, label in samples:
    filename = os.path.basename(path)

    parts = filename.split("_")

    # Extract video ID
    if len(parts) >= 3:
        video_id = "_".join(parts[1:-1])
    else:
        video_id = filename

    groups[video_id].append((path, label))


# ----------------------------
# SPLIT GROUPS (NO LEAKAGE)
# ----------------------------
group_keys = list(groups.keys())
random.shuffle(group_keys)

split_idx = int(0.8 * len(group_keys))

train_keys = group_keys[:split_idx]
val_keys = group_keys[split_idx:]

train_samples = []
val_samples = []

for key in train_keys:
    train_samples.extend(groups[key])

for key in val_keys:
    val_samples.extend(groups[key])

print(f"Train samples: {len(train_samples)}")
print(f"Val samples: {len(val_samples)}")


# ----------------------------
# CUSTOM DATASET
# ----------------------------
class DeepfakeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ----------------------------
# DATA LOADERS
# ----------------------------
train_dataset = DeepfakeDataset(train_samples, train_transform)
val_dataset = DeepfakeDataset(val_samples, val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0
)

# ----------------------------
# MODEL
# ----------------------------
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(DEVICE)


# ----------------------------
# LOSS + OPTIMIZER
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# ----------------------------
# TRAIN LOOP
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"\nEpoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # ----------------------------
    # VALIDATION
    # ----------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")


# ----------------------------
# SAVE MODEL
# ----------------------------
torch.save(model.state_dict(), "deepfake_model_augmented.pth")
print("\nModel saved!")