"""
train_ravdess.py
================
Train the DistilHuBERT + mean-pool + linear classifier on RAVDESS, 6 classes.

Matches the methodology in your existing work:
  - 6 classes: neutral_calm, happy, sad, angry, disgust, surprised
  - 'calm' merged into 'neutral_calm'
  - ACTOR-INDEPENDENT splits (no actor appears in both train and test)
  - weighted cross-entropy (handles the neutral_calm imbalance)
  - reports per-class precision/recall/F1 + confusion matrix
  - optional actor-independent K-fold CV -> mean +/- std macro F1 (headline metric)

DistilHuBERT replaces your MFCC+CNN front end. We precompute one pooled
768-d embedding per clip (backbone frozen), then train just the linear head.
This is fast and runs even on a modest GPU.

USAGE
-----
# 1) single actor-independent split (quick):
python train_ravdess.py --data_dir /path/to/RAVDESS --mode split

# 2) full actor-independent K-fold CV (headline number for resume/labs):
python train_ravdess.py --data_dir /path/to/RAVDESS --mode cv --folds 6

RAVDESS layout expected (the standard zip):
  RAVDESS/Actor_01/03-01-05-01-01-01-01.wav
  ...
Filename field 3 (1-indexed) is the emotion code:
  01 neutral 02 calm 03 happy 04 sad 05 angry 06 fearful 07 disgust 08 surprised
We KEEP 01,02,03,04,05,07,08 and DROP 06 (fearful) to stay at your 6 classes.
Field 7 is the actor id.
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

import librosa
from transformers import AutoFeatureExtractor

from model import (
    DistilHubertClassifier, EMOTIONS, NUM_CLASSES,
    DISTILHUBERT_NAME, SAMPLE_RATE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RAVDESS emotion-code (filename field 3) -> our class index.
# fearful (06) is intentionally absent => those files are skipped.
RAVDESS_CODE_TO_IDX = {
    "01": 0,  # neutral  -> neutral_calm
    "02": 0,  # calm     -> neutral_calm
    "03": 1,  # happy
    "04": 2,  # sad
    "05": 3,  # angry
    "07": 4,  # disgust
    "08": 5,  # surprised
}


# ---------------------------------------------------------------------------
# 1. Index the dataset: list (wav_path, label, actor_id)
# ---------------------------------------------------------------------------
def index_ravdess(data_dir):
    paths = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    if not paths:
        raise FileNotFoundError(
            f"No .wav files found under {data_dir}. "
            f"Expected RAVDESS/Actor_xx/*.wav"
        )
    items = []
    for p in paths:
        name = os.path.basename(p).replace(".wav", "")
        f = name.split("-")
        if len(f) != 7:
            continue
        emo_code = f[2]
        actor_id = int(f[6])
        if emo_code not in RAVDESS_CODE_TO_IDX:
            continue  # skips fearful
        items.append((p, RAVDESS_CODE_TO_IDX[emo_code], actor_id))
    print(f"Indexed {len(items)} clips across "
          f"{len(set(a for _, _, a in items))} actors.")
    return items


# ---------------------------------------------------------------------------
# 2. Precompute pooled DistilHuBERT embeddings (frozen backbone).
#    One 768-d vector per clip. Cached to disk so re-runs are instant.
# ---------------------------------------------------------------------------
def compute_embeddings(items, cache_path="ravdess_embeddings.npz",
                       max_seconds=3.0):
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return d["X"], d["y"], d["actors"]

    fe = AutoFeatureExtractor.from_pretrained(DISTILHUBERT_NAME)
    backbone = DistilHubertClassifier(freeze_backbone=True).backbone
    backbone.to(DEVICE).eval()

    X, y, actors = [], [], []
    max_len = int(max_seconds * SAMPLE_RATE)

    for i, (path, label, actor) in enumerate(items):
        wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        # center-crop / pad to a fixed max so memory is bounded
        if len(wav) > max_len:
            start = (len(wav) - max_len) // 2
            wav = wav[start:start + max_len]
        inputs = fe(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        iv = inputs["input_values"].to(DEVICE)
        with torch.no_grad():
            hs = backbone(iv).last_hidden_state      # (1, T, 768)
            emb = hs.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)
        X.append(emb)
        y.append(label)
        actors.append(actor)
        if (i + 1) % 100 == 0:
            print(f"  embedded {i + 1}/{len(items)}")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    actors = np.array(actors, dtype=np.int64)
    np.savez(cache_path, X=X, y=y, actors=actors)
    print(f"Saved embeddings -> {cache_path}  X={X.shape}")
    return X, y, actors


# ---------------------------------------------------------------------------
# 3. A linear head trainer that works on cached embeddings (fast).
#    This is the SAME head as model.py's DistilHubertClassifier.head, trained
#    standalone on frozen features. We export it so live_infer.py can load it
#    into the full model.
# ---------------------------------------------------------------------------
def train_head(X_tr, y_tr, X_te, y_te, epochs=120, lr=3e-4, wd=1e-4):
    feat_dim = X_tr.shape[1]
    head = nn.Linear(feat_dim, NUM_CLASSES).to(DEVICE)

    # weighted cross-entropy: weight_c = N / (K * count_c)
    counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(y_tr) / (NUM_CLASSES * counts)
    weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)

    Xtr = torch.tensor(X_tr, device=DEVICE)
    ytr = torch.tensor(y_tr, device=DEVICE)
    Xte = torch.tensor(X_te, device=DEVICE)

    head.train()
    for ep in range(epochs):
        opt.zero_grad()
        logits = head(Xtr)
        loss = crit(logits, ytr)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        pred = head(Xte).argmax(dim=1).cpu().numpy()
    return head, pred


def report(y_true, y_pred, header=""):
    print(f"\n===== {header} =====")
    print(classification_report(
        y_true, y_pred, target_names=EMOTIONS, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ---------------------------------------------------------------------------
# 4. Modes
# ---------------------------------------------------------------------------
def run_split(X, y, actors, holdout_actors=(22, 23, 24)):
    """Single actor-independent split (mirrors your test on actors 22-24)."""
    te_mask = np.isin(actors, holdout_actors)
    tr_mask = ~te_mask
    head, pred = train_head(X[tr_mask], y[tr_mask], X[te_mask], y[te_mask])
    f1 = report(y[te_mask], pred, f"TEST (actors {holdout_actors})")
    print(f"\nMacro F1: {f1:.4f}")
    save_head(head)
    return f1


def run_cv(X, y, actors, folds=6):
    """Actor-independent K-fold CV -> pooled report + mean +/- std macro F1."""
    gkf = GroupKFold(n_splits=folds)
    fold_f1s = []
    pooled_true, pooled_pred = [], []
    best_head, best_f1 = None, -1.0

    for k, (tr, te) in enumerate(gkf.split(X, y, groups=actors)):
        head, pred = train_head(X[tr], y[tr], X[te], y[te])
        f1 = f1_score(y[te], pred, average="macro", zero_division=0)
        test_actors = sorted(set(actors[te].tolist()))
        print(f"Fold {k + 1}/{folds}  actors={test_actors}  macroF1={f1:.4f}")
        fold_f1s.append(f1)
        pooled_true.extend(y[te].tolist())
        pooled_pred.extend(pred.tolist())
        if f1 > best_f1:
            best_f1, best_head = f1, head

    report(np.array(pooled_true), np.array(pooled_pred),
           "POOLED (all actors as test once)")
    arr = np.array(fold_f1s)
    print(f"\nFold macro F1s: {[f'{x:.4f}' for x in fold_f1s]}")
    print(f"HEADLINE: CV macro F1 = {arr.mean():.4f} +/- {arr.std():.4f} "
          f"over {folds} actor-independent folds")
    save_head(best_head)  # save best fold's head for deployment


# ---------------------------------------------------------------------------
# 5. Save the trained head into a full deployable checkpoint.
#    We instantiate the FULL model (backbone + head) and copy head weights in,
#    so live_infer.py loads one clean state_dict.
# ---------------------------------------------------------------------------
def save_head(head, out_path="distilhubert_ser.pt"):
    full = DistilHubertClassifier(freeze_backbone=True)
    # copy the standalone linear head weights into the model's head
    with torch.no_grad():
        full.head.weight.copy_(head.weight.detach().cpu())
        full.head.bias.copy_(head.bias.detach().cpu())
    torch.save(
        {"state_dict": full.state_dict(), "emotions": EMOTIONS},
        out_path,
    )
    print(f"\nSaved deployable checkpoint -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="path to RAVDESS root")
    ap.add_argument("--mode", choices=["split", "cv"], default="cv")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--cache", default="ravdess_embeddings.npz")
    args = ap.parse_args()

    items = index_ravdess(args.data_dir)
    X, y, actors = compute_embeddings(items, cache_path=args.cache)

    if args.mode == "split":
        run_split(X, y, actors)
    else:
        run_cv(X, y, actors, folds=args.folds)


if __name__ == "__main__":
    main()