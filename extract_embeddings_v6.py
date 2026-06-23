"""
extract_embeddings_v6.py  (Week 6)
==================================
Pre-generate DistilHuBERT embeddings ONCE, including deployment-relevant
augmented copies, and cache to disk. Training then runs in seconds.

WHY pre-baked (not on-the-fly):
  The backbone is frozen. On-the-fly audio aug would re-run the 94M-param
  backbone every epoch (~50-100x slower) only to train a small head. We get
  the same regularization benefit far cheaper by pre-generating a few
  augmented embedding copies once.

WHICH augmentations (and why only these):
  We bake ONLY the augmentations that mimic the live-Jetson deployment gap
  between clean studio audio and a real mic in a room:
    * additive Gaussian noise  (mic/room noise)
    * random gain              (mic distance / level)
  Pitch/speed shift are NOT baked: pitch IS part of the emotion signal
  (your Model 5 leaned on F0), so shifting it risks corrupting labels for
  little gain. They're available behind --risky_aug if you want to experiment.

OUTPUT (single .npz):
  X       : (N, 768) float32  embeddings  (clean + augmented)
  y       : (N,) int64        labels
  speaker : (N,) int64        speaker id (RAVDESS actor or CREMA-D speaker)
  is_aug  : (N,) bool         True for augmented rows (so eval can exclude them)
  source  : (N,) str          'ravdess' or 'cremad'

USAGE
-----
RAVDESS only, clean + 1 noise + 1 gain copy per clip:
  python extract_embeddings_v6.py --ravdess_dir /path/RAVDESS --n_aug 2

RAVDESS + CREMA-D:
  python extract_embeddings_v6.py --ravdess_dir /path/RAVDESS \
      --include_cremad --cremad_dir /home/vinod/SERproject/CREMA-D --n_aug 2
"""

import argparse
import glob
import os

import numpy as np
import torch
import librosa

from transformers import AutoFeatureExtractor
from model import DISTILHUBERT_NAME, SAMPLE_RATE
from transformers import HubertModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTIONS = ["neutral_calm", "happy", "sad", "angry", "disgust", "surprised"]
EMO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

# RAVDESS filename field 3 -> class idx (fearful 06 dropped, calm 02 -> neutral)
RAVDESS_CODE_TO_IDX = {
    "01": 0, "02": 0, "03": 1, "04": 2, "05": 3, "07": 4, "08": 5,
}

# CREMA-D emotion code (in filename) -> class idx. FEA (fear) excluded.
# CREMA-D filenames: 1001_DFA_ANG_XX.wav  -> field[2] is the emotion code.
CREMAD_CODE_TO_IDX = {
    "NEU": 0, "HAP": 1, "SAD": 2, "ANG": 3, "DIS": 4,
    # CREMA-D has no 'surprised'; 'FEA' (fear) intentionally excluded.
}
# Speaker-id offset so CREMA-D speakers never collide with RAVDESS actors 1..24
CREMAD_SPK_OFFSET = 1000


# ---------------------------------------------------------------------------
# Augmentations (waveform level). Applied BEFORE the feature extractor.
# ---------------------------------------------------------------------------
def aug_gaussian_noise(wav, snr_db_range=(15, 30), rng=None):
    rng = rng or np.random
    snr_db = rng.uniform(*snr_db_range)
    sig_power = np.mean(wav ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)
    return wav + noise


def aug_random_gain(wav, gain_db_range=(-6, 6), rng=None):
    rng = rng or np.random
    gain = 10 ** (rng.uniform(*gain_db_range) / 20)
    return (wav * gain).astype(np.float32)


def aug_time_shift(wav, max_frac=0.1, rng=None):
    rng = rng or np.random
    shift = int(rng.uniform(-max_frac, max_frac) * len(wav))
    return np.roll(wav, shift).astype(np.float32)


def aug_risky_pitch_speed(wav, sr, rng=None):
    """Off by default. Pitch/speed can corrupt emotion labels."""
    rng = rng or np.random
    if rng.random() < 0.5:
        steps = rng.uniform(-1.5, 1.5)
        wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=steps)
    else:
        rate = rng.uniform(0.92, 1.08)
        wav = librosa.effects.time_stretch(wav, rate=rate)
    return wav.astype(np.float32)


def make_aug(wav, sr, risky=False, rng=None):
    """One random augmented copy using deployment-relevant transforms."""
    rng = rng or np.random
    out = wav.copy()
    out = aug_time_shift(out, rng=rng)
    if rng.random() < 0.8:
        out = aug_gaussian_noise(out, rng=rng)
    if rng.random() < 0.8:
        out = aug_random_gain(out, rng=rng)
    if risky:
        out = aug_risky_pitch_speed(out, sr, rng=rng)
    return out


# ---------------------------------------------------------------------------
# Dataset indexing
# ---------------------------------------------------------------------------
def index_ravdess(data_dir):
    items = []
    for p in glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True):
        f = os.path.basename(p).replace(".wav", "").split("-")
        if len(f) != 7:
            continue
        if f[2] not in RAVDESS_CODE_TO_IDX:
            continue
        items.append((p, RAVDESS_CODE_TO_IDX[f[2]], int(f[6]), "ravdess"))
    return items


def index_cremad(data_dir):
    items = []
    # CREMA-D audio usually lives in an AudioWAV/ subfolder
    search = os.path.join(data_dir, "**", "*.wav")
    for p in glob.glob(search, recursive=True):
        name = os.path.basename(p).replace(".wav", "")
        parts = name.split("_")
        if len(parts) < 3:
            continue
        spk = parts[0]          # e.g. 1001
        emo = parts[2]          # e.g. ANG
        if emo not in CREMAD_CODE_TO_IDX:
            continue            # skips FEA (fear)
        try:
            spk_id = CREMAD_SPK_OFFSET + int(spk)
        except ValueError:
            continue
        items.append((p, CREMAD_CODE_TO_IDX[emo], spk_id, "cremad"))
    return items


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_one(backbone, fe, wav, max_seconds=3.0):
    max_len = int(max_seconds * SAMPLE_RATE)
    if len(wav) > max_len:
        start = (len(wav) - max_len) // 2
        wav = wav[start:start + max_len]
    inputs = fe(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    iv = inputs["input_values"].to(DEVICE)
    with torch.no_grad():
        hs = backbone(iv).last_hidden_state
        return hs.mean(dim=1).squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess_dir", required=True)
    ap.add_argument("--include_cremad", action="store_true")
    ap.add_argument("--cremad_dir", default="/home/vinod/SERproject/CREMA-D")
    ap.add_argument("--n_aug", type=int, default=2,
                    help="augmented copies per TRAINING clip (0 = clean only)")
    ap.add_argument("--risky_aug", action="store_true",
                    help="also apply pitch/speed shift (may corrupt labels)")
    ap.add_argument("--out", default="ravdess_embeddings_v6.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    items = index_ravdess(args.ravdess_dir)
    print(f"RAVDESS: {len(items)} clips")
    if args.include_cremad:
        if not os.path.isdir(args.cremad_dir):
            raise FileNotFoundError(f"CREMA-D not found at {args.cremad_dir}")
        c = index_cremad(args.cremad_dir)
        print(f"CREMA-D: {len(c)} clips (fear excluded; no 'surprised' in CREMA-D)")
        items += c
    print(f"TOTAL source clips: {len(items)}")

    fe = AutoFeatureExtractor.from_pretrained(DISTILHUBERT_NAME)
    backbone = HubertModel.from_pretrained(DISTILHUBERT_NAME).to(DEVICE).eval()

    X, y, spk, is_aug, source = [], [], [], [], []
    for i, (path, label, speaker, src) in enumerate(items):
        wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

        # clean embedding (always)
        X.append(embed_one(backbone, fe, wav)); y.append(label)
        spk.append(speaker); is_aug.append(False); source.append(src)

        # augmented embeddings (marked is_aug so eval can drop them)
        for _ in range(args.n_aug):
            aw = make_aug(wav, SAMPLE_RATE, risky=args.risky_aug, rng=rng)
            X.append(embed_one(backbone, fe, aw)); y.append(label)
            spk.append(speaker); is_aug.append(True); source.append(src)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)} clips embedded")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    spk = np.array(spk, dtype=np.int64)
    is_aug = np.array(is_aug, dtype=bool)
    source = np.array(source)
    np.savez(args.out, X=X, y=y, speaker=spk, is_aug=is_aug, source=source)
    print(f"\nSaved {args.out}: X={X.shape}, "
          f"{(~is_aug).sum()} clean + {is_aug.sum()} augmented rows")


if __name__ == "__main__":
    main()