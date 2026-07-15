import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

device_id = 0
fs = 44100
seconds = 3

print("Recording from device", device_id)

audio = sd.rec(
    int(seconds * fs),
    samplerate=fs,
    channels=1,
    dtype="float32",
    device=device_id
)
sd.wait()

print("max:", np.max(np.abs(audio)))
print("std:", np.std(audio))

write("test_audio.wav", fs, np.int16(audio * 32767))
print("Saved test_audio.wav")