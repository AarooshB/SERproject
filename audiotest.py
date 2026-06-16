import sounddevice as sd
from scipy.io.wavfile import write
sd.default.device = (24, None)
fs = 44100  # sample rate
seconds = 5

print("Recording...")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
print("Recording complete")

# save file
filename = "test_audio.wav"
write(filename, fs, audio)
print(f"Saved to {filename}")

# playback
print("Playing back...")
sd.play(audio, fs)
sd.wait()
print("Done")