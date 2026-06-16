import os
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# -------------------------
# Config
# -------------------------

MFCC_DIR = Path("features_mfcc_norm")
BATCH_SIZE = 32
EPOCHS = 80
LR = 3e-4
WEIGHT_DECAY = 2e-4
PATIENCE = 14
NUM_CLASSES = 6
TARGET_FRAMES = 130
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 6-class setup:
# neutral + calm merged, fearful dropped
# RAVDESS emotion code is filename field 3
EMOTION_MAP = {
    "01": 0,  # neutral -> neutral_calm
    "02": 0,  # calm -> neutral_calm
    "03": 1,  # happy
    "04": 2,  # sad
    "05": 3,  # angry
    # "06": dropped, fearful
    "07": 4,  # disgust
    "08": 5,  # surprised
}

IDX_TO_EMOTION = {
    0: "neutral_calm",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "disgust",
    5: "surprised",
}

TRAIN_ACTORS = set(range(1, 19))
VAL_ACTORS = set(range(19, 22))
TEST_ACTORS = set(range(22, 25))


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything()


def parse_ravdess_filename(path):
    stem = Path(path).stem
    parts = stem.split("-")
    emotion_code = parts[2]
    actor_id = int(parts[6])

    if emotion_code not in EMOTION_MAP:
        return None, actor_id

    return EMOTION_MAP[emotion_code], actor_id


def fix_time_length(x, target_frames=TARGET_FRAMES):
    # Expected shape: [features, time]
    if x.shape[1] < target_frames:
        pad_width = target_frames - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad_width)), mode="constant")
    else:
        x = x[:, :target_frames]
    return x


class RavdessMFCCDataset(Dataset):
    def __init__(self, files, augment=False, target_frames=TARGET_FRAMES):
        self.files = files
        self.augment = augment
        self.target_frames = target_frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        x = np.load(path).astype(np.float32)

        # Convert [time, features] to [features, time] if needed
        if x.shape[0] != 120 and x.shape[1] == 120:
            x = x.T

        x = fix_time_length(x, self.target_frames)

        if self.augment:
            x = self.augment_features(x)

        label, _ = parse_ravdess_filename(path)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def augment_features(self, x):
        # Mild noise
        if random.random() < 0.55:
            x = x + np.random.normal(0, 0.025, size=x.shape).astype(np.float32)

        # Random time shift
        if random.random() < 0.55:
            shift = random.randint(-8, 8)
            x = np.roll(x, shift, axis=1)
            if shift > 0:
                x[:, :shift] = 0
            elif shift < 0:
                x[:, shift:] = 0

        # Time masking
        if random.random() < 0.45:
            t = x.shape[1]
            mask_len = random.randint(5, 16)
            start = random.randint(0, max(0, t - mask_len))
            x[:, start:start + mask_len] = 0

        # Feature masking
        if random.random() < 0.35:
            f = x.shape[0]
            mask_len = random.randint(5, 14)
            start = random.randint(0, max(0, f - mask_len))
            x[start:start + mask_len, :] = 0

        return x


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
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class TinySERNet6(nn.Module):
    def __init__(self, input_channels=120, num_classes=6):
        super().__init__()

        # Smaller than previous 64 -> 96 -> 128 model.
        # This reduces memorization and is faster on Jetson Nano.
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.30),

            SeparableConv1d(32, 48, kernel_size=5, padding=2),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.35),

            SeparableConv1d(48, 64, kernel_size=5, padding=2),
            nn.Dropout(0.35),

            SeparableConv1d(64, 64, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_split_files():
    all_files = sorted(MFCC_DIR.glob("*.npy"))
    train_files, val_files, test_files = [], [], []
    skipped = 0

    for f in all_files:
        label, actor_id = parse_ravdess_filename(f)
        if label is None:
            skipped += 1
            continue

        if actor_id in TRAIN_ACTORS:
            train_files.append(f)
        elif actor_id in VAL_ACTORS:
            val_files.append(f)
        elif actor_id in TEST_ACTORS:
            test_files.append(f)

    print(f"Skipped fearful files: {skipped}")
    return train_files, val_files, test_files


def make_weighted_sampler(files):
    labels = [parse_ravdess_filename(f)[0] for f in files]
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights_per_class = 1.0 / np.maximum(counts, 1)
    sample_weights = [weights_per_class[y] for y in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0
    preds, labels = [], []

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
            preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
            labels.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return avg_loss, acc, macro_f1


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(y.numpy())

    names = [IDX_TO_EMOTION[i] for i in range(NUM_CLASSES)]
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=names, digits=4))
    print("\nConfusion matrix:")
    print(confusion_matrix(labels, preds))


def main():
    train_files, val_files, test_files = get_split_files()

    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Test files:  {len(test_files)}")
    print(f"Device:      {DEVICE}")

    if not train_files:
        raise RuntimeError(f"No .npy files found in {MFCC_DIR.resolve()}")

    sample = np.load(train_files[0])
    print(f"Sample shape before fix: {sample.shape}")

    train_labels = np.array([parse_ravdess_filename(f)[0] for f in train_files])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=train_labels,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    train_ds = RavdessMFCCDataset(train_files, augment=True)
    val_ds = RavdessMFCCDataset(val_files, augment=False)
    test_ds = RavdessMFCCDataset(test_files, augment=False)

    train_sampler = make_weighted_sampler(train_files)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = TinySERNet6(input_channels=120, num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.08)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )

    best_val_f1 = 0.0
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
                    "target_frames": TARGET_FRAMES,
                    "dropped": "fearful",
                    "merged": "neutral+calm",
                },
                "tiny_ser_mfcc_6class_best.pt",
            )
            print("Saved best model.")
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("Early stopping.")
            break

    checkpoint = torch.load("tiny_ser_mfcc_6class_best.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nFinal test evaluation:")
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()
