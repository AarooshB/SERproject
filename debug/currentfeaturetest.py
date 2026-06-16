import os
import numpy as np
import torch

from livetest import SERNetMid, CLASS_NAMES, CHECKPOINT_PATH

FEATURE_DIR = "features_mfcc_norm"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SERNetMid(input_channels=120, num_classes=6).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".npy")]

for file in files[:20]:
    path = os.path.join(FEATURE_DIR, file)
    x = np.load(path).astype(np.float32)

    if x.shape[1] < 94:
        x = np.pad(x, ((0, 0), (0, 94 - x.shape[1])), mode="constant")
    else:
        x = x[:, :94]

    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))

    print(file)
    print("Prediction:", CLASS_NAMES[pred])
    print("Probs:", np.round(probs, 3))
    print()