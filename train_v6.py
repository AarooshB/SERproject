"""
train_v6.py  (Week 6)
=====================
Train an improved head on the cached v6 embeddings with speaker-independent
cross-validation. Saves a full artifact bundle.

KEY CORRECTNESS RULES (don't break these):
  1. Speaker-independent folds: no speaker (RAVDESS actor OR CREMA-D speaker)
     appears in both train and val. GroupKFold on the `speaker` array enforces it.
  2. Augmented rows are TRAIN-ONLY. Validation uses CLEAN rows only, so metrics
     reflect real performance, not augmented copies. We mask is_aug in val.
  3. Macro F1 is the headline metric (not accuracy).

Outputs (into --out_dir):
  best_model.pt              head checkpoint + metadata (loadable by live_infer_v6)
  class_mapping.json         index -> emotion
  train_config.json          all hyperparams for reproducibility
  fold_metrics.csv           per-fold accuracy + macro F1
  pooled_report.txt          pooled classification report
  confusion_matrix.png       pooled confusion matrix image
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
from sklearn.model_selection import GroupKFold

from model_v6 import EmbeddingHead, FocalLoss

EMOTIONS = ["neutral_calm", "happy", "sad", "angry", "disgust", "surprised"]
NUM_CLASSES = len(EMOTIONS)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return (d["X"], d["y"], d["speaker"], d["is_aug"],
            d["source"] if "source" in d else np.array(["?"] * len(d["y"])))


def class_weights(y):
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    counts[counts == 0] = 1.0
    w = len(y) / (NUM_CLASSES * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def train_head(X_tr, y_tr, X_va, head_kind, loss_kind, epochs, lr, wd,
               dropout, feat_dropout, gamma):
    head = EmbeddingHead(head_kind, NUM_CLASSES, dropout, feat_dropout).to(DEVICE)
    w = class_weights(y_tr)
    if loss_kind == "focal":
        crit = FocalLoss(weight=w, gamma=gamma)
    else:
        crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)

    Xtr = torch.tensor(X_tr, device=DEVICE)
    ytr = torch.tensor(y_tr, device=DEVICE)
    Xva = torch.tensor(X_va, device=DEVICE)

    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(head(Xtr), ytr)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        pred = head(Xva).argmax(dim=1).cpu().numpy()
    return head, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="ravdess_embeddings_v6.npz")
    ap.add_argument("--head", choices=["linear", "mlp1", "mlp2"], default="mlp1")
    ap.add_argument("--loss", choices=["wce", "focal"], default="wce")
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--feat_dropout", type=float, default=0.1)
    ap.add_argument("--out_dir", default="week6_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    X, y, spk, is_aug, source = load_data(args.emb)
    print(f"Loaded {X.shape[0]} rows ({(~is_aug).sum()} clean, {is_aug.sum()} aug)")

    gkf = GroupKFold(n_splits=args.folds)
    # Group on speaker so folds are speaker-independent. We split on CLEAN rows
    # only (so every speaker is representable), then add that speaker's aug rows
    # to TRAIN.
    clean_idx = np.where(~is_aug)[0]
    Xc, yc, spkc = X[clean_idx], y[clean_idx], spk[clean_idx]

    fold_rows, pooled_true, pooled_pred = [], [], []
    best_f1, best_head = -1.0, None

    for k, (tr_c, va_c) in enumerate(gkf.split(Xc, yc, groups=spkc)):
        train_speakers = set(spkc[tr_c].tolist())
        val_speakers = sorted(set(spkc[va_c].tolist()))

        # TRAIN = all rows (clean+aug) whose speaker is in train_speakers
        tr_mask = np.isin(spk, list(train_speakers))
        # VAL = CLEAN rows whose speaker is in val_speakers (no aug in val!)
        va_mask = (~is_aug) & np.isin(spk, val_speakers)

        head, pred = train_head(
            X[tr_mask], y[tr_mask], X[va_mask],
            args.head, args.loss, args.epochs, args.lr, args.wd,
            args.dropout, args.feat_dropout, args.gamma)

        yt = y[va_mask]
        f1 = f1_score(yt, pred, average="macro", zero_division=0)
        acc = accuracy_score(yt, pred)
        print(f"Fold {k+1}/{args.folds} val_speakers={val_speakers} "
              f"acc={acc:.4f} macroF1={f1:.4f}")
        fold_rows.append({"fold": k + 1, "accuracy": round(acc, 4),
                          "macro_f1": round(f1, 4),
                          "val_speakers": " ".join(map(str, val_speakers))})
        pooled_true.extend(yt.tolist())
        pooled_pred.extend(pred.tolist())
        if f1 > best_f1:
            best_f1, best_head = f1, head

    # ---- pooled metrics ----
    pt, pp = np.array(pooled_true), np.array(pooled_pred)
    report = classification_report(pt, pp, target_names=EMOTIONS,
                                   digits=4, zero_division=0)
    cm = confusion_matrix(pt, pp)
    f1s = np.array([r["macro_f1"] for r in fold_rows])
    headline = (f"CV macro F1 = {f1s.mean():.4f} +/- {f1s.std():.4f} "
                f"over {args.folds} speaker-independent folds")
    print("\n===== POOLED =====\n" + report)
    print("Confusion matrix:\n", cm)
    print("\nHEADLINE:", headline)

    # ---- save artifacts ----
    torch.save({"head_state": best_head.state_dict(),
                "head_kind": args.head, "emotions": EMOTIONS},
               os.path.join(args.out_dir, "best_model.pt"))
    with open(os.path.join(args.out_dir, "class_mapping.json"), "w") as f:
        json.dump({i: e for i, e in enumerate(EMOTIONS)}, f, indent=2)
    with open(os.path.join(args.out_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.out_dir, "fold_metrics.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        wr.writeheader(); wr.writerows(fold_rows)
    with open(os.path.join(args.out_dir, "pooled_report.txt"), "w") as f:
        f.write(report + "\n\n" + headline + "\n\nConfusion matrix:\n" + str(cm))

    # confusion matrix image (matplotlib optional; skip cleanly if missing)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")
        ax.set_yticklabels(EMOTIONS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Pooled CM  (macroF1={f1s.mean():.3f})")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=120)
        print("Saved confusion_matrix.png")
    except ImportError:
        print("(matplotlib not installed; skipped confusion_matrix.png)")

    print(f"\nAll artifacts saved to {args.out_dir}/")


if __name__ == "__main__":
    main()