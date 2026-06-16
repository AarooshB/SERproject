import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
WINDOW_SECONDS = 3
N_MFCC = 40

print("Recording 3 seconds. Talk clearly.")

audio = sd.rec(
    int(WINDOW_SECONDS * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
)

sd.wait()

audio = audio.squeeze()

# =====================
# SAVE AUDIO
# =====================

audio_int16 = np.int16(audio * 32767)
write("debug_recording.wav", SAMPLE_RATE, audio_int16)

print("\nSaved audio: debug_recording.wav")

# =====================
# AUDIO STATS
# =====================

print("\nAUDIO STATS")
print("max:", np.max(np.abs(audio)))
print("mean:", np.mean(audio))
print("std:", np.std(audio))

# =====================
# FEATURE EXTRACTION
# =====================

y = librosa.util.fix_length(audio, size=int(SAMPLE_RATE * 3))

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

# =====================
# FEATURE STATS
# =====================

print("\nLIVE FEATURE STATS")
print("shape:", features.shape)
print("mean:", features.mean())
print("std:", features.std())
print("min:", features.min())
print("max:", features.max())

np.save("debug_live_feature.npy", features)

print("\nSaved: debug_live_feature.npy")