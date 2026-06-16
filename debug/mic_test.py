import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

fs = 16000
seconds = 3

print(sd.query_devices())

device_id = int(input("Enter input device ID: "))

print("Recording...")
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

sd.play(audio, fs)
sd.wait()