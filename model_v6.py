"""
model_v6.py  (Week 6)
=====================
Improved classifier heads that operate on the 768-d pooled DistilHuBERT
embedding. Backbone stays FROZEN and identical to model.py, so:
  - your cached embeddings are still valid
  - inference cost is unchanged (same backbone forward, just a bigger head)
  - nothing in model.py / live_infer*.py is overwritten

Heads available (pick via --head):
  linear : 768 -> 6                              (your current baseline)
  mlp1   : 768 -> LN -> 256 -> drop -> 6
  mlp2   : 768 -> LN -> 256 -> 128 -> drop -> 6

"SpecAugment" decision (spec item #4):
  Spectrogram SpecAugment does NOT fit a frozen-backbone embedding pipeline --
  by the time we have the 768-d vector, there is no spectrogram to mask.
  The clean, pipeline-appropriate analog is FEATURE DROPOUT on the embedding
  (randomly zero whole feature dimensions during training). That is the
  embedding-level equivalent of channel masking and costs nothing at inference.
  It's built into the heads below via `feat_dropout`.
"""

import torch
import torch.nn as nn

EMBED_DIM = 768


class EmbeddingHead(nn.Module):
    """Classifier head over a precomputed 768-d embedding."""

    def __init__(self, kind="mlp1", num_classes=6,
                 dropout=0.3, feat_dropout=0.1):
        super().__init__()
        self.kind = kind
        # feature dropout = "channel masking" on the embedding (SpecAugment analog)
        self.feat_dropout = nn.Dropout(feat_dropout) if feat_dropout > 0 else nn.Identity()

        if kind == "linear":
            self.net = nn.Linear(EMBED_DIM, num_classes)
        elif kind == "mlp1":
            self.net = nn.Sequential(
                nn.LayerNorm(EMBED_DIM),
                nn.Linear(EMBED_DIM, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )
        elif kind == "mlp2":
            self.net = nn.Sequential(
                nn.LayerNorm(EMBED_DIM),
                nn.Linear(EMBED_DIM, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )
        else:
            raise ValueError(f"unknown head kind: {kind}")

    def forward(self, x):
        # x: (B, 768)
        x = self.feat_dropout(x)
        return self.net(x)


class FocalLoss(nn.Module):
    """
    Focal loss for class imbalance (spec item #7).
    Down-weights easy examples so the head focuses on hard ones
    (your sad/happy confusions). gamma=0 reduces to weighted CE.
    """

    def __init__(self, weight=None, gamma=1.5):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(
            logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)            # prob of the true class
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


def build_full_model_state(head, backbone_state=None):
    """
    Wrap a trained EmbeddingHead back into the full-model layout that
    live_infer can load. We keep it simple: live_infer_v6 loads the head
    separately and reuses the frozen backbone from HF, so we just save the
    head state_dict + metadata. (No need to duplicate 94M backbone params.)
    """
    return {"head_state": head.state_dict(), "head_kind": head.kind}