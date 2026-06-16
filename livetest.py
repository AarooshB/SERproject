import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import librosa
import sounddevice as sd


# =====================
# CONFIG
# =====================

CHECKPOINT_PATH = "model4_ser_mfcc_6class_best.pt"

MIC_SAMPLE_RATE = 44100
SAMPLE_RATE = 16000

INPUT_DEVICE = 0

WINDOW_SECONDS = 3
N_MFCC = 40
TARGET_FRAMES = 94

SMOOTHING_WINDOW = 5

CLASS_NAMES = [
    "neutral_calm",
    "happy",
    "sad",
    "angry",
    "disgust",
    "surprised",
]


# =====================
# MODEL
# =====================

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


class SERNetMid(nn.Module):
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
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =====================
# FEATURE EXTRACTION
# =====================

def fix_frames(x, target_frames=TARGET_FRAMES):
    if x.shape[1] < target_frames:
        pad_width = target_frames - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad_width)), mode="constant")
    else:
        x = x[:, :target_frames]

    return x


def extract_features(audio):
    audio = audio.astype(np.float32)

    y = librosa.util.fix_length(
        audio,
        size=int(SAMPLE_RATE * 3)
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])

    # per-channel normalization
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True) + 1e-8

    features = (features - mean) / std

    features = fix_frames(features)

    return features.astype(np.float32)


# =====================
# LOAD MODEL
# =====================

def load_model(device):
    model = SERNetMid(
        input_channels=120,
        num_classes=6
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    return model


# =====================
# PREDICTION
# =====================

def predict(model, features, device):
    x = torch.from_numpy(features).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1
        ).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    return pred_idx, confidence, probs


# =====================
# MAIN LOOP
# =====================

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Loading model: {CHECKPOINT_PATH}")

    model = load_model(device)

    print("Model loaded successfully.")
    print("Starting live SER...\n")

    history = deque(maxlen=SMOOTHING_WINDOW)

    try:
        while True:
            total_start = time.time()

            print("Listening...")

            audio = sd.rec(
                int(WINDOW_SECONDS * MIC_SAMPLE_RATE),
                samplerate=MIC_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=INPUT_DEVICE,
            )

            sd.wait()

            audio = audio.squeeze()

            print("Audio max:", np.max(np.abs(audio)))
            print("Audio std:", np.std(audio))

            # resample mic audio -> model sample rate
            audio = librosa.resample(
                audio,
                orig_sr=MIC_SAMPLE_RATE,
                target_sr=SAMPLE_RATE
            )

            feat_start = time.time()

            features = extract_features(audio)

            feat_ms = (
                time.time() - feat_start
            ) * 1000

            infer_start = time.time()

            pred_idx, confidence, probs = predict(
                model,
                features,
                device
            )

            infer_ms = (
                time.time() - infer_start
            ) * 1000

            history.append(pred_idx)

            smoothed_idx = max(
                set(history),
                key=list(history).count
            )

            total_ms = (
                time.time() - total_start
            ) * 1000

            print("--------------------------------")
            print(f"Prediction:      {CLASS_NAMES[pred_idx]}")
            print(f"Confidence:      {confidence:.2f}")
            print(f"Smoothed:        {CLASS_NAMES[smoothed_idx]}")
            print(f"Probabilities:   {np.round(probs, 3)}")
            print(f"Feature time:    {feat_ms:.1f} ms")
            print(f"Inference time:  {infer_ms:.1f} ms")
            print(f"Total time:      {total_ms:.1f} ms")
            print("--------------------------------\n")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()