import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
model.eval()

audio, sr = librosa.load("test.wav", sr=16000)

inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt",
    padding=True
)

inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)

print(embeddings.shape)