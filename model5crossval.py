"""
Model 5 with leave-actors-out cross-validation.

RAVDESS's 24 actors are split into 8 folds of 3 actors. For each fold k:
  test = fold k (3 actors)
  val  = fold k+1 (3 actors, used only for early stopping / model selection)
  train = remaining 18 RAVDESS actors + ALL of CREMA-D

Reports per-fold macro F1, mean +/- std across folds, and a pooled
confusion matrix over all 24 actors (every actor appears in test exactly once).

Run AFTER extract_features_v5.py. Expect ~8x the training time of one run.
"""

import random
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Config
# -------------------------

FEAT_DIR = Path("features_v5_norm")
BATCH_SIZE = 64
EPOCHS = 60
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10
NUM_CLASSES = 6
INPUT_CHANNELS = 123
TARGET_FRAMES = 130

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IDX_TO_EMOTION = {
    0: "neutral_calm", 1: "happy", 2: "sad",
    3: "angry", 4: "disgust", 5: "surprised",
}

# 8 folds of 3 RAVDESS actors each
FOLDS = [[f"rav{a:02d}" for a in range(start, start + 3)] for start in range(1, 25, 3)]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_feature_filename(path):
    parts = Path(path).stem.split("-")
    return int(parts[1]), parts[2]  # label, speaker


def fix_shape_and_length(x, target_frames=TARGET_FRAMES):
    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {x.shape}")
    if x.shape[1] == INPUT_CHANNELS and x.shape[0] != INPUT_CHANNELS:
        x = x.T
    if x.shape[0] != INPUT_CHANNELS:
        raise ValueError(f"Expected {INPUT_CHANNELS} channels, got shape {x.shape}")
    if x.shape[1] < target_frames:
        x = np.pad(x, ((0, 0), (0, target_frames - x.shape[1])), mode="constant")
    else:
        x = x[:, :target_frames]
    return x.astype(np.float32)


class SERFeatureDataset(Dataset):
    def __init__(self, files, augment=False):
        self.files = files
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        x = np.load(path).astype(np.float32)
        x = fix_shape_and_length(x)
        if self.augment:
            x = self.augment_features(x)
        y, _ = parse_feature_filename(path)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def augment_features(self, x):
        x = x.copy()
        if random.random() < 0.4:
            x = x + np.random.normal(0, 0.015, size=x.shape).astype(np.float32)
        if random.random() < 0.4:
            x = np.roll(x, random.randint(-5, 5), axis=1)
        if random.random() < 0.5:
            t = random.randint(5, 18)
            t0 = random.randint(0, max(0, x.shape[1] - t))
            x[:, t0:t0 + t] = 0.0
        if random.random() < 0.5:
            f = random.randint(3, 10)
            f0_idx = random.randint(0, max(0, 120 - f))
            x[f0_idx:f0_idx + f, :] = 0.0
        return x


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class AttentiveStatsPooling(nn.Module):
    def __init__(self, channels, attn_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, attn_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attn_dim, channels, kernel_size=1),
        )

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=2)
        mean = torch.sum(x * w, dim=2)
        var = torch.sum((x ** 2) * w, dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-6))
        return torch.cat([mean, std], dim=1)


class SERNetV5(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 48, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(48), nn.ReLU(), nn.Dropout(0.20),
            SeparableConv1d(48, 64), nn.MaxPool1d(2), nn.Dropout(0.25),
            SeparableConv1d(64, 96), nn.Dropout(0.25),
            SeparableConv1d(96, 96, kernel_size=3, padding=1),
        )
        self.gru = nn.GRU(96, 64, batch_first=True, bidirectional=True)
        self.pool = AttentiveStatsPooling(128)
        self.classifier = nn.Sequential(
            nn.Linear(256, 96), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        return self.classifier(x)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, preds, labels = 0.0, [], []

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds.extend(logits.argmax(1).detach().cpu().numpy())
            labels.extend(y.detach().cpu().numpy())

    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), f1


def predict(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    return np.array(labels), np.array(preds)


def train_one_fold(fold_idx, train_files, val_files, test_files):
    seed_everything(42 + fold_idx)

    train_labels = [parse_feature_filename(f)[0] for f in train_files]
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(NUM_CLASSES), y=np.array(train_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(SERFeatureDataset(train_files, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(SERFeatureDataset(val_files),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(SERFeatureDataset(test_files),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SERNetV5().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4)

    best_val_f1, bad_epochs = 0.0, 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer)
        _, val_f1 = run_epoch(model, val_loader, criterion)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break

    model.load_state_dict(best_state)
    labels, preds = predict(model, test_loader)
    fold_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"Fold {fold_idx + 1}/8 | test actors {FOLDS[fold_idx]} | "
          f"best val F1 {best_val_f1:.4f} | test macro F1 {fold_f1:.4f}")
    return labels, preds, fold_f1


def main():
    all_files = sorted(FEAT_DIR.glob("*.npy"))
    if not all_files:
        raise RuntimeError(f"No .npy files found in {FEAT_DIR.resolve()}")
    print(f"Total feature files: {len(all_files)} | Device: {DEVICE}\n")

    all_labels, all_preds, fold_f1s = [], [], []

    for k in range(len(FOLDS)):
        test_actors = set(FOLDS[k])
        val_actors = set(FOLDS[(k + 1) % len(FOLDS)])

        train_files, val_files, test_files = [], [], []
        for f in all_files:
            _, speaker = parse_feature_filename(f)
            if speaker in test_actors:
                test_files.append(f)
            elif speaker in val_actors:
                val_files.append(f)
            else:
                train_files.append(f)  # remaining RAVDESS + all CREMA-D

        labels, preds, fold_f1 = train_one_fold(k, train_files, val_files, test_files)
        all_labels.extend(labels)
        all_preds.extend(preds)
        fold_f1s.append(fold_f1)

    fold_f1s = np.array(fold_f1s)
    names = [IDX_TO_EMOTION[i] for i in range(NUM_CLASSES)]

    print("\n================ CROSS-VALIDATION SUMMARY ================")
    print(f"Per-fold macro F1: {np.round(fold_f1s, 4).tolist()}")
    print(f"Mean macro F1: {fold_f1s.mean():.4f} +/- {fold_f1s.std():.4f}")
    print("\nPooled classification report (all 24 actors as test):")
    print(classification_report(all_labels, all_preds, target_names=names,
                                digits=4, zero_division=0))
    print("Pooled confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()