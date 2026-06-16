import os
import numpy as np
import librosa

# ===== CONFIG =====
AUDIO_DIR = "all_audio"
SAVE_DIR = "audio_windows"

SAMPLE_RATE = 16000
TARGET_DURATION = 3

WINDOW_SEC = 2.0
HOP_SEC = 0.5
# ==================

os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_LENGTH = int(SAMPLE_RATE * TARGET_DURATION)
WIN_LEN = int(SAMPLE_RATE * WINDOW_SEC)
HOP_LEN = int(SAMPLE_RATE * HOP_SEC)


def sliding_windows(y):
    windows = []
    for start in range(0, len(y) - WIN_LEN + 1, HOP_LEN):
        end = start + WIN_LEN
        windows.append(y[start:end])
    return windows


for file in os.listdir(AUDIO_DIR):
    if not file.endswith(".wav"):
        continue

    path = os.path.join(AUDIO_DIR, file)

    # load + fix to 3 sec
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    y = librosa.util.fix_length(y, size=TARGET_LENGTH)

    windows = sliding_windows(y)

    for i, w in enumerate(windows):
        save_name = file.replace(".wav", f"_w{i}.npy")
        np.save(os.path.join(SAVE_DIR, save_name), w)


print("Done.")