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
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 16
NUM_CLASSES = 6
TARGET_FRAMES = 130

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTION_MAP = {
    "01": 0,  # neutral_calm
    "02": 0,  # neutral_calm
    "03": 1,  # happy
    "04": 2,  # sad
    "05": 3,  # angry
    # "06" fearful is dropped
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
    return emotion_code, actor_id


def get_label(path):
    emotion_code, actor_id = parse_ravdess_filename(path)
    if emotion_code not in EMOTION_MAP:
        return None
    return EMOTION_MAP[emotion_code]


def fix_shape_and_length(x, target_frames=TARGET_FRAMES):
    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {x.shape}")

    if x.shape[1] == 120 and x.shape[0] != 120:
        x = x.T

    if x.shape[0] != 120:
        raise ValueError(f"Expected 120 feature channels, got shape {x.shape}")

    if x.shape[1] < target_frames:
        pad_width = target_frames - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad_width)), mode="constant")
    else:
        x = x[:, :target_frames]

    return x.astype(np.float32)


class RavdessMFCCDataset(Dataset):
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

        y = get_label(path)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def augment_features(self, x):
        # x: [120, T]
        x = x.copy()

        # Small Gaussian noise (kept from model 3)
        if random.random() < 0.4:
            noise = np.random.normal(0, 0.015, size=x.shape).astype(np.float32)
            x = x + noise

        # Small time shift (kept from model 3)
        if random.random() < 0.4:
            shift = random.randint(-5, 5)
            x = np.roll(x, shift, axis=1)

        # NEW: SpecAugment-style time masking (zero out a random chunk of frames)
        if random.random() < 0.5:
            t = random.randint(5, 18)
            t0 = random.randint(0, max(0, x.shape[1] - t))
            x[:, t0:t0 + t] = 0.0

        # NEW: SpecAugment-style channel/frequency masking
        if random.random() < 0.5:
            f = random.randint(3, 10)
            f0 = random.randint(0, max(0, x.shape[0] - f))
            x[f0:f0 + f, :] = 0.0

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
        x = self.act(x)
        return x


class AttentiveStatsPooling(nn.Module):
    """
    Learns per-frame attention weights, then summarizes the sequence
    with an attention-weighted mean AND std. Output dim = 2 * channels.

    Compared to plain global average pooling, this lets the model focus on
    the emotionally informative frames and keep information about how much
    the features VARY over time (key for sad vs neutral, happy vs surprised).
    """
    def __init__(self, channels, attn_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, attn_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attn_dim, channels, kernel_size=1),
        )

    def forward(self, x):
        # x: [B, C, T]
        w = torch.softmax(self.attn(x), dim=2)          # [B, C, T]
        mean = torch.sum(x * w, dim=2)                  # [B, C]
        var = torch.sum((x ** 2) * w, dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-6))           # [B, C]
        return torch.cat([mean, std], dim=1)            # [B, 2C]


class SERNetV4(nn.Module):
    """
    Model 4.
    Same conv trunk as model 3 (48 -> 64 -> 96), but:
      - a small BiGRU on top of the conv features to model temporal order
      - attentive statistics pooling instead of global average pooling
    """
    def __init__(self, input_channels=120, num_classes=6):
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
            input_size=96,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.pool = AttentiveStatsPooling(channels=128)  # BiGRU output = 2*64

        self.classifier = nn.Sequential(
            nn.Linear(256, 96),   # pooled dim = 2 * 128
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.features(x)          # [B, 96, T']
        x = x.transpose(1, 2)         # [B, T', 96]
        x, _ = self.gru(x)            # [B, T', 128]
        x = x.transpose(1, 2)         # [B, 128, T']
        x = self.pool(x)              # [B, 256]
        x = self.classifier(x)
        return x


def get_split_files():
    all_files = sorted(MFCC_DIR.glob("*.npy"))
    train_files, val_files, test_files = [], [], []

    for f in all_files:
        emotion_code, actor_id = parse_ravdess_filename(f)

        if emotion_code not in EMOTION_MAP:
            continue

        if actor_id in TRAIN_ACTORS:
            train_files.append(f)
        elif actor_id in VAL_ACTORS:
            val_files.append(f)
        elif actor_id in TEST_ACTORS:
            test_files.append(f)

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


def evaluate(model, loader):
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
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=names, digits=4, zero_division=0))
    print("\nConfusion matrix:")
    print(confusion_matrix(labels, preds))


def main():
    train_files, val_files, test_files = get_split_files()

    if not train_files:
        raise RuntimeError(f"No .npy files found in {MFCC_DIR.resolve()}")

    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Test files:  {len(test_files)}")
    print(f"Device:      {DEVICE}")
    print(f"Sample shape before fix: {np.load(train_files[0]).shape}")

    train_labels = [get_label(f) for f in train_files]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=np.array(train_labels),
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {class_weights.detach().cpu().numpy()}")

    train_ds = RavdessMFCCDataset(train_files, augment=True)
    val_ds = RavdessMFCCDataset(val_files, augment=False)
    test_ds = RavdessMFCCDataset(test_files, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SERNetV4(input_channels=120, num_classes=NUM_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )

    best_val_f1 = 0.0
    bad_epochs = 0
    save_path = "model4_ser_mfcc_6class_best.pt"

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

    print("\nFinal test evaluation:")
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()