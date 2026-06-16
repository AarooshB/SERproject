import os
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Config
# -------------------------

MFCC_DIR = Path("features_mfcc_norm")
BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12
NUM_CLASSES = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RAVDESS emotion labels: filename field 3
EMOTION_MAP = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fearful
    "07": 6,  # disgust
    "08": 7,  # surprised
}

IDX_TO_EMOTION = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised",
}

TRAIN_ACTORS = set(range(1, 19))
VAL_ACTORS = set(range(19, 22))
TEST_ACTORS = set(range(22, 25))


# -------------------------
# Reproducibility
# -------------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything()


# -------------------------
# Dataset
# -------------------------

def parse_ravdess_filename(path):
    stem = Path(path).stem
    parts = stem.split("-")

    emotion_code = parts[2]
    actor_id = int(parts[6])

    label = EMOTION_MAP[emotion_code]
    return label, actor_id


def fix_time_length(x, target_frames=130):
    """
    Force every feature file to have same time length.
    Expected input shape: [features, time]
    """
    if x.shape[1] < target_frames:
        pad_width = target_frames - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad_width)), mode="constant")
    else:
        x = x[:, :target_frames]

    return x


class RavdessMFCCDataset(Dataset):
    def __init__(self, files, augment=False, target_frames=130):
        self.files = files
        self.augment = augment
        self.target_frames = target_frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        x = np.load(path).astype(np.float32)

        # Make sure shape is [features, time]
        if x.shape[0] < x.shape[1]:
            pass
        else:
            # If saved as [time, features], transpose
            if x.shape[1] == 120 or x.shape[1] == 40:
                x = x.T

        x = fix_time_length(x, self.target_frames)

        if self.augment:
            x = self.augment_features(x)

        label, actor_id = parse_ravdess_filename(path)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def augment_features(self, x):
        # Small feature noise. Do not overdo it.
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.02, size=x.shape).astype(np.float32)
            x = x + noise

        # Time masking
        if random.random() < 0.4:
            t = x.shape[1]
            mask_len = random.randint(4, 12)
            start = random.randint(0, max(0, t - mask_len))
            x[:, start:start + mask_len] = 0

        # Feature masking
        if random.random() < 0.3:
            f = x.shape[0]
            mask_len = random.randint(4, 12)
            start = random.randint(0, max(0, f - mask_len))
            x[start:start + mask_len, :] = 0

        return x


# -------------------------
# Model
# -------------------------

class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super().__init__()

        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class TinySERNet(nn.Module):
    def __init__(self, input_channels=120, num_classes=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),

            SeparableConv1d(64, 96, kernel_size=5, padding=2),
            nn.MaxPool1d(kernel_size=2),

            SeparableConv1d(96, 128, kernel_size=5, padding=2),
            nn.Dropout(0.20),

            SeparableConv1d(128, 128, kernel_size=3, padding=1),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x


# -------------------------
# Train helpers
# -------------------------

def get_split_files():
    all_files = sorted(MFCC_DIR.glob("*.npy"))

    train_files = []
    val_files = []
    test_files = []

    for f in all_files:
        label, actor_id = parse_ravdess_filename(f)

        if actor_id in TRAIN_ACTORS:
            train_files.append(f)
        elif actor_id in VAL_ACTORS:
            val_files.append(f)
        elif actor_id in TEST_ACTORS:
            test_files.append(f)

    return train_files, val_files, test_files


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    preds = []
    labels = []

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()

            total_loss += loss.item() * x.size(0)

            pred = logits.argmax(dim=1)
            preds.extend(pred.detach().cpu().numpy())
            labels.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    return avg_loss, acc, macro_f1


def evaluate(model, loader):
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.numpy())

    names = [IDX_TO_EMOTION[i] for i in range(NUM_CLASSES)]

    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=names, digits=4))

    print("\nConfusion matrix:")
    print(confusion_matrix(labels, preds))


# -------------------------
# Main
# -------------------------

def main():
    train_files, val_files, test_files = get_split_files()

    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Test files:  {len(test_files)}")
    print(f"Device:      {DEVICE}")

    sample = np.load(train_files[0])
    print(f"Sample shape before fix: {sample.shape}")

    train_labels = [parse_ravdess_filename(f)[0] for f in train_files]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=np.array(train_labels),
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    train_ds = RavdessMFCCDataset(train_files, augment=True)
    val_ds = RavdessMFCCDataset(val_files, augment=False)
    test_ds = RavdessMFCCDataset(test_files, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = TinySERNet(input_channels=120, num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )

    best_val_f1 = 0
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion)

        scheduler.step(val_f1)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            bad_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "idx_to_emotion": IDX_TO_EMOTION,
                    "input_channels": 120,
                    "num_classes": NUM_CLASSES,
                },
                "tiny_ser_mfcc_best.pt",
            )

            print("Saved best model.")
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("Early stopping.")
            break

    checkpoint = torch.load("tiny_ser_mfcc_best.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nFinal test evaluation:")
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()