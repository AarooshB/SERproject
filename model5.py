"""
Model 5: same architecture as Model 4 (conv trunk + BiGRU + attentive stats
pooling), but trained on RAVDESS + CREMA-D with 123-channel features
(MFCC + deltas + pitch + voicing + energy) from extract_features_v5.py.

Splits (so results stay comparable to models 3/4):
  train: RAVDESS actors 1-18  +  ALL of CREMA-D
  val:   RAVDESS actors 19-21
  test:  RAVDESS actors 22-24

Feature filenames: {dataset}-{label}-{speaker}-{counter}.npy
"""

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

FEAT_DIR = Path("features_v5_norm")
BATCH_SIZE = 64            # bigger dataset -> bigger batch is fine
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 12
NUM_CLASSES = 6
INPUT_CHANNELS = 123
TARGET_FRAMES = 130

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IDX_TO_EMOTION = {
    0: "neutral_calm",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "disgust",
    5: "surprised",
}

VAL_ACTORS = {f"rav{a:02d}" for a in range(19, 22)}
TEST_ACTORS = {f"rav{a:02d}" for a in range(22, 25)}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything()


def parse_feature_filename(path):
    # {dataset}-{label}-{speaker}-{counter}.npy
    parts = Path(path).stem.split("-")
    label = int(parts[1])
    speaker = parts[2]
    return label, speaker


def fix_shape_and_length(x, target_frames=TARGET_FRAMES):
    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {x.shape}")

    if x.shape[1] == INPUT_CHANNELS and x.shape[0] != INPUT_CHANNELS:
        x = x.T

    if x.shape[0] != INPUT_CHANNELS:
        raise ValueError(f"Expected {INPUT_CHANNELS} channels, got shape {x.shape}")

    if x.shape[1] < target_frames:
        pad_width = target_frames - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad_width)), mode="constant")
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
            noise = np.random.normal(0, 0.015, size=x.shape).astype(np.float32)
            x = x + noise

        if random.random() < 0.4:
            shift = random.randint(-5, 5)
            x = np.roll(x, shift, axis=1)

        # SpecAugment-style time masking
        if random.random() < 0.5:
            t = random.randint(5, 18)
            t0 = random.randint(0, max(0, x.shape[1] - t))
            x[:, t0:t0 + t] = 0.0

        # Channel masking — restricted to the 120 MFCC channels so the
        # 3 prosody channels (F0, voicing, RMS) are never masked out.
        if random.random() < 0.5:
            f = random.randint(3, 10)
            f0_idx = random.randint(0, max(0, 120 - f))
            x[f0_idx:f0_idx + f, :] = 0.0

        return x


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


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
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.20),

            SeparableConv1d(48, 64, kernel_size=5, padding=2),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25),

            SeparableConv1d(64, 96, kernel_size=5, padding=2),
            nn.Dropout(0.25),

            SeparableConv1d(96, 96, kernel_size=3, padding=1),
        )

        self.gru = nn.GRU(
            input_size=96, hidden_size=64, num_layers=1,
            batch_first=True, bidirectional=True,
        )

        self.pool = AttentiveStatsPooling(channels=128)

        self.classifier = nn.Sequential(
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def get_split_files():
    all_files = sorted(FEAT_DIR.glob("*.npy"))
    train_files, val_files, test_files = [], [], []

    for f in all_files:
        _, speaker = parse_feature_filename(f)
        if speaker in TEST_ACTORS:
            test_files.append(f)
        elif speaker in VAL_ACTORS:
            val_files.append(f)
        else:
            train_files.append(f)  # RAVDESS 1-18 + all CREMA-D

    return train_files, val_files, test_files


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
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
            pred = logits.argmax(dim=1)
            preds.extend(pred.detach().cpu().numpy())
            labels.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


def evaluate(model, loader, header):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.numpy())

    names = [IDX_TO_EMOTION[i] for i in range(NUM_CLASSES)]
    print(f"\n===== {header} =====")
    print(classification_report(labels, preds, target_names=names, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))


def main():
    train_files, val_files, test_files = get_split_files()

    if not train_files:
        raise RuntimeError(f"No .npy files found in {FEAT_DIR.resolve()}")

    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Test files:  {len(test_files)}")
    print(f"Device:      {DEVICE}")

    train_labels = [parse_feature_filename(f)[0] for f in train_files]
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    print(f"Train class counts: {dict(zip(IDX_TO_EMOTION.values(), counts))}")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=np.array(train_labels),
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {class_weights.detach().cpu().numpy()}")

    train_ds = SERFeatureDataset(train_files, augment=True)
    val_ds = SERFeatureDataset(val_files, augment=False)
    test_ds = SERFeatureDataset(test_files, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SERNetV5().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4,
    )

    best_val_f1 = 0.0
    bad_epochs = 0
    save_path = "model5_ser_combined_best.pt"

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
                    "input_channels": INPUT_CHANNELS,
                    "num_classes": NUM_CLASSES,
                    "target_frames": TARGET_FRAMES,
                },
                save_path,
            )
            print("Saved best model.")
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("Early stopping.")
            break

    checkpoint = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Print BOTH val and test confusion matrices (val helps diagnose
    # whether the tiny 3-actor test set is just hard).
    evaluate(model, val_loader, "VALIDATION (RAVDESS actors 19-21)")
    evaluate(model, test_loader, "TEST (RAVDESS actors 22-24)")


if __name__ == "__main__":
    main()