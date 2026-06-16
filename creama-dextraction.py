"""
Feature extraction for Model 5.

Processes BOTH datasets with identical settings so they are fully uniform:
  - RAVDESS  (48kHz wavs)  -> loaded at 16kHz
  - CREMA-D  (16kHz wavs)  -> loaded at 16kHz

Per file, computes 123 channels on the same time grid (hop = 512 @ 16kHz):
  40 MFCC + 40 delta + 40 delta-delta   (same as your old pipeline)
  + 1 F0 / pitch contour (pyin, 0 where unvoiced)
  + 1 voiced probability
  + 1 RMS energy

Then per-speaker, per-channel normalization (same scheme as your old script).

Output filenames are unified:  {dataset}-{label}-{speaker}-{counter}.npy
  e.g.  rav-2-rav14-0531.npy   cre-3-cre1042-2210.npy
so the training script parses every file the same way.

Label map (6-class, fearful dropped):
  0 neutral_calm   1 happy   2 sad   3 angry   4 disgust   5 surprised
CREMA-D has no "surprised"; RAVDESS provides it.
"""

import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa

# -------------------------
# Config
# -------------------------

RAVDESS_DIR = Path("all_audio")      # your existing RAVDESS wavs
CREMA_DIR = Path("CREMA-D/AudioWAV") # adjust to wherever you unzip CREMA-D
SAVE_DIR = Path("features_v5_norm")

SAMPLE_RATE = 16000
N_MFCC = 40
HOP_LENGTH = 512        # explicit now (was librosa default before — same value)
N_FFT = 2048
FMIN_PITCH = 65.0       # ~C2, low end of speech
FMAX_PITCH = 400.0      # high end of typical speech F0

# RAVDESS emotion codes -> 6-class label
RAVDESS_MAP = {
    "01": 0,  # neutral
    "02": 0,  # calm -> neutral_calm
    "03": 1,  # happy
    "04": 2,  # sad
    "05": 3,  # angry
    # "06" fearful dropped
    "07": 4,  # disgust
    "08": 5,  # surprised
}

# CREMA-D emotion codes -> 6-class label
CREMA_MAP = {
    "NEU": 0,
    "HAP": 1,
    "SAD": 2,
    "ANG": 3,
    "DIS": 4,
    # "FEA" fearful dropped; CREMA-D has no surprised
}

SAVE_DIR.mkdir(exist_ok=True)


def extract_features(y, sr=SAMPLE_RATE):
    """
    Shared feature function. Use this EXACT function in the live pipeline
    later so training and deployment can never drift apart.
    Input: 1D float audio array at 16kHz. Output: [123, T] float32.
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=FMIN_PITCH,
        fmax=FMAX_PITCH,
        sr=sr,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )
    f0 = np.nan_to_num(f0, nan=0.0)            # unvoiced frames -> 0
    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)

    rms = librosa.feature.rms(
        y=y, frame_length=N_FFT, hop_length=HOP_LENGTH
    )[0]

    # All tracks share the hop, but lengths can differ by a frame; trim to min.
    T = min(mfcc.shape[1], len(f0), len(rms))
    feats = np.vstack([
        mfcc[:, :T],
        d1[:, :T],
        d2[:, :T],
        f0[None, :T],
        voiced_prob[None, :T],
        rms[None, :T],
    ])
    return feats.astype(np.float32)  # [123, T]


def collect_files():
    """Yield (wav_path, label, speaker_id, dataset_tag) for both datasets."""
    items = []

    # RAVDESS: 03-01-EMO-..-..-..-ACTOR.wav
    for f in sorted(RAVDESS_DIR.glob("*.wav")):
        parts = f.stem.split("-")
        if len(parts) < 7:
            continue
        emo = parts[2]
        if emo not in RAVDESS_MAP:
            continue  # drops fearful
        actor = int(parts[6])
        items.append((f, RAVDESS_MAP[emo], f"rav{actor:02d}", "rav"))

    # CREMA-D: 1001_DFA_ANG_XX.wav  ->  ActorID_Sentence_Emotion_Level
    if CREMA_DIR.exists():
        for f in sorted(CREMA_DIR.glob("*.wav")):
            parts = f.stem.split("_")
            if len(parts) < 4:
                continue
            emo = parts[2]
            if emo not in CREMA_MAP:
                continue  # drops FEA
            items.append((f, CREMA_MAP[emo], f"cre{parts[0]}", "cre"))
    else:
        print(f"WARNING: {CREMA_DIR} not found — extracting RAVDESS only.")

    return items


def main():
    items = collect_files()
    print(f"Total files to process: {len(items)}")

    # ---- Pass 1: extract raw features, group by speaker ----
    speaker_feats = defaultdict(list)   # speaker -> list of (out_name, feats)

    for i, (wav, label, speaker, tag) in enumerate(items):
        try:
            y, _ = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
            feats = extract_features(y)
        except Exception as e:
            print(f"Skipping {wav.name}: {e}")
            continue

        out_name = f"{tag}-{label}-{speaker}-{i:05d}.npy"
        speaker_feats[speaker].append((out_name, feats))

        if (i + 1) % 200 == 0:
            print(f"  extracted {i + 1}/{len(items)}")

    # ---- Pass 2: per-speaker, per-channel normalization (same as before) ----
    for speaker, entries in speaker_feats.items():
        stacked = np.hstack([f for _, f in entries])      # [123, total_T]
        mean = np.mean(stacked, axis=1, keepdims=True)
        std = np.std(stacked, axis=1, keepdims=True) + 1e-8

        for out_name, feats in entries:
            norm = (feats - mean) / std
            np.save(SAVE_DIR / out_name, norm.astype(np.float32))

    n_saved = len(list(SAVE_DIR.glob("*.npy")))
    print(f"Done. Saved {n_saved} normalized feature files to {SAVE_DIR}/")
    print("Channels: 123 = 40 MFCC + 40 delta + 40 delta2 + F0 + voiced_prob + RMS")


if __name__ == "__main__":
    main()