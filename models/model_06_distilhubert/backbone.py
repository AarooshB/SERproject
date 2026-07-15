"""
Shared model definition for live speech emotion recognition.

DistilHuBERT (frozen or fine-tuned) -> mean pool over time -> linear classifier.

Both train_ravdess.py and live_infer.py import from here so the architecture
can NEVER drift between training and inference. If you change the head here,
both scripts stay in sync automatically.
"""

import torch
import torch.nn as nn
from transformers import HubertModel

# ---------------------------------------------------------------------------
# 6-class label map. Order MUST stay fixed: the classifier's output neuron i
# corresponds to EMOTIONS[i]. This matches your existing RAVDESS 6-class setup.
# RAVDESS 'calm' is merged into 'neutral_calm'.
# ---------------------------------------------------------------------------
EMOTIONS = [
    "neutral_calm",  # 0
    "happy",         # 1
    "sad",           # 2
    "angry",         # 3
    "disgust",       # 4
    "surprised",     # 5
]
NUM_CLASSES = len(EMOTIONS)

DISTILHUBERT_NAME = "ntu-spml/distilhubert"
SAMPLE_RATE = 16000  # DistilHuBERT expects 16 kHz mono


class DistilHubertClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=True,
                 hidden_dim=None):
        """
        num_classes    : number of emotion classes (6)
        freeze_backbone: if True, DistilHuBERT weights are frozen and only the
                         head trains. Start with True (fast, low-RAM). Set False
                         later to fine-tune the top layers for more accuracy.
        hidden_dim     : if set, uses a 2-layer MLP head instead of a single
                         linear layer. None = pure linear (your spec).
        """
        super().__init__()
        self.backbone = HubertModel.from_pretrained(DISTILHUBERT_NAME)
        feat_dim = self.backbone.config.hidden_size  # 768 for distilhubert

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()  # disable dropout in backbone

        if hidden_dim is None:
            self.head = nn.Linear(feat_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes),
            )

    @staticmethod
    def _masked_mean_pool(hidden_states, attention_mask=None):
        """
        Mean-pool hidden states over the time axis.
        hidden_states: (B, T, H)
        attention_mask (optional): (B, T) 1 for real frames, 0 for padding.
        Using a masked mean means padded batches don't dilute the average.
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).float()        # (B, T, 1)
        summed = (hidden_states * mask).sum(dim=1)          # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-6)            # (B, 1)
        return summed / counts

    def forward(self, input_values, attention_mask=None):
        """
        input_values  : (B, num_samples) float waveform at 16 kHz, normalized
        attention_mask: (B, num_samples) optional, from the feature extractor
        returns        : (B, num_classes) logits
        """
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone(input_values, attention_mask=attention_mask)
        else:
            out = self.backbone(input_values, attention_mask=attention_mask)

        hidden = out.last_hidden_state  # (B, T, H)

        # The backbone downsamples time, so the waveform-level attention_mask
        # doesn't line up with T. HubertModel exposes a helper to convert it.
        feat_mask = None
        if attention_mask is not None:
            feat_len = self.backbone._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1)
            ).to(torch.long)
            feat_mask = torch.zeros(
                hidden.shape[0], hidden.shape[1],
                device=hidden.device, dtype=torch.long
            )
            for i, L in enumerate(feat_len):
                feat_mask[i, :L] = 1

        pooled = self._masked_mean_pool(hidden, feat_mask)  # (B, H)
        return self.head(pooled)