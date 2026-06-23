"""
live_infer_v6.py

Fixed live mic inference for DistilHuBERT SER.

Changes:
- Records mic at native sample rate, usually 44100 Hz
- Resamples to 16000 Hz for DistilHuBERT
- Prints mic peak/RMS/std
- Skips weak silence windows
- Normalizes live mic volume before inference
- Resets smoothing during silence so it does not get stuck on sad

Run:
  python live_infer_v6.py --source mic --ckpt week6_out/best_model.pt --device 0

Sim:
  python live_infer_v6.py --source sim --ckpt week6_out/best_model.pt --wav sample.wav
"""

import argparse
import collections
import queue
import sys
import time

import numpy as np
import torch

from transformers import AutoFeatureExtractor, HubertModel
from model import DISTILHUBERT_NAME, SAMPLE_RATE
from model_v6 import EmbeddingHead


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTIONS = [
    "neutral_calm",
    "happy",
    "sad",
    "angry",
    "disgust",
    "surprised",
]

WINDOW_SEC = 2.5
HOP_SEC = 1.0

SMOOTH_N = 5


def load(ckpt_path):
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {ckpt_path}")

    fe = AutoFeatureExtractor.from_pretrained(DISTILHUBERT_NAME)
    backbone = HubertModel.from_pretrained(DISTILHUBERT_NAME).to(DEVICE).eval()

    ck = torch.load(ckpt_path, map_location="cpu")

    head_kind = ck.get("head_kind", "mlp1")

    # Your current model_v6.py expects positional args.
    head = EmbeddingHead(head_kind, len(EMOTIONS)).to(DEVICE)
    head.load_state_dict(ck["head_state"])
    head.eval()

    if DEVICE == "cuda":
        backbone.half()
        head.half()

    print("Model loaded.")
    return fe, backbone, head


def infer_window(fe, backbone, head, wav):
    timing = {}

    t0 = time.time()

    inputs = fe(
        wav,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    input_values = inputs["input_values"].to(DEVICE)

    if DEVICE == "cuda":
        input_values = input_values.half()

    timing["preprocess"] = (time.time() - t0) * 1000

    t0 = time.time()

    with torch.inference_mode():
        hidden = backbone(input_values).last_hidden_state
        emb = hidden.mean(dim=1)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    timing["backbone"] = (time.time() - t0) * 1000

    t0 = time.time()

    with torch.inference_mode():
        logits = head(emb)
        probs = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    timing["head"] = (time.time() - t0) * 1000
    timing["total"] = timing["preprocess"] + timing["backbone"] + timing["head"]

    return probs, timing


def smoother():
    ema = np.ones(len(EMOTIONS), dtype=np.float32) / len(EMOTIONS)
    votes = collections.deque(maxlen=SMOOTH_N)

    def update(probs, alpha=0.6):
        nonlocal ema

        ema = alpha * ema + (1.0 - alpha) * probs

        raw_idx = int(probs.argmax())
        votes.append(raw_idx)

        maj_idx = collections.Counter(votes).most_common(1)[0][0]
        ema_idx = int(ema.argmax())

        return raw_idx, ema_idx, maj_idx, ema

    return update


def normalize_live_audio(wav, target_rms):
    wav = wav.astype(np.float32)

    # Remove DC offset.
    wav = wav - np.mean(wav)

    rms = np.sqrt(np.mean(wav ** 2)) + 1e-8

    # Normalize quiet mic audio so it does not look like sad/low-energy speech.
    wav = wav * (target_rms / rms)

    # Avoid clipping.
    wav = np.clip(wav, -1.0, 1.0)

    return wav.astype(np.float32)


def run_mic(fe, backbone, head, args):
    import sounddevice as sd
    import librosa

    update = smoother()
    audio_q = queue.Queue()

    device_info = sd.query_devices(args.device, "input")
    mic_sr = int(device_info["default_samplerate"])

    win_mic = int(WINDOW_SEC * mic_sr)
    hop_mic = int(HOP_SEC * mic_sr)

    win_model = int(WINDOW_SEC * SAMPLE_RATE)

    ring = np.zeros(win_mic, dtype=np.float32)

    print("\nAvailable audio devices:")
    print(sd.query_devices())

    print("\nMic config:")
    print(f"Input device: {args.device}")
    print(f"Device name: {device_info['name']}")
    print(f"Mic sample rate: {mic_sr}")
    print(f"Model sample rate: {SAMPLE_RATE}")
    print(f"Window: {WINDOW_SEC}s")
    print(f"Hop: {HOP_SEC}s")
    print(f"RMS silence threshold: {args.rms_silence}")
    print(f"Target normalized RMS: {args.target_rms}")
    print()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Sounddevice status: {status}", file=sys.stderr)

        audio_q.put(indata[:, 0].copy())

    # Warm up model.
    infer_window(fe, backbone, head, np.zeros(win_model, dtype=np.float32))

    print("Listening... Ctrl+C to stop\n")

    with sd.InputStream(
        samplerate=mic_sr,
        channels=1,
        dtype="float32",
        blocksize=hop_mic,
        callback=callback,
        device=args.device,
    ):
        while True:
            block = audio_q.get()

            ring = np.roll(ring, -len(block))
            ring[-len(block):] = block

            peak = float(np.max(np.abs(ring)))
            rms = float(np.sqrt(np.mean(ring ** 2)))
            std = float(np.std(ring))

            print(f"[mic] peak={peak:.5f} rms={rms:.5f} std={std:.5f}")

            if rms < args.rms_silence:
                print(f"[silence rms={rms:.5f}] resetting smoother\n")
                update = smoother()
                continue

            t0 = time.time()

            wav16 = librosa.resample(
                ring,
                orig_sr=mic_sr,
                target_sr=SAMPLE_RATE,
            ).astype(np.float32)

            if len(wav16) < win_model:
                wav16 = np.pad(wav16, (0, win_model - len(wav16)))
            else:
                wav16 = wav16[:win_model]

            wav16 = normalize_live_audio(wav16, args.target_rms)

            resample_norm_ms = (time.time() - t0) * 1000

            probs, timing = infer_window(fe, backbone, head, wav16)

            raw_idx, ema_idx, maj_idx, ema = update(probs)

            print(f"Raw:       {EMOTIONS[raw_idx]:12s} conf={probs[raw_idx]:.2f}")
            print(f"Smoothed:  {EMOTIONS[maj_idx]:12s} conf={ema[maj_idx]:.2f}")
            print(f"Probs:     {np.round(probs, 3)}")
            print(
                f"Latency:   audio {resample_norm_ms:.1f} ms | "
                f"pre {timing['preprocess']:.1f} ms | "
                f"backbone {timing['backbone']:.1f} ms | "
                f"head {timing['head']:.1f} ms | "
                f"total {timing['total'] + resample_norm_ms:.1f} ms"
            )
            print("-" * 60)


def run_sim(fe, backbone, head, args):
    import librosa

    update = smoother()

    wav, _ = librosa.load(
        args.wav,
        sr=SAMPLE_RATE,
        mono=True,
    )

    win = int(WINDOW_SEC * SAMPLE_RATE)
    hop = int(HOP_SEC * SAMPLE_RATE)

    infer_window(fe, backbone, head, np.zeros(win, dtype=np.float32))

    print(f"Simulating rolling inference over {args.wav}")
    print(f"Audio length: {len(wav) / SAMPLE_RATE:.1f}s\n")

    for start in range(0, max(1, len(wav) - win + 1), hop):
        window = wav[start:start + win]

        if len(window) < win:
            window = np.pad(window, (0, win - len(window)))

        rms = float(np.sqrt(np.mean(window ** 2)))
        tstamp = start / SAMPLE_RATE

        if rms < args.rms_silence:
            print(f"t={tstamp:5.1f}s [silence rms={rms:.5f}]")
            update = smoother()
            continue

        window = normalize_live_audio(window, args.target_rms)

        probs, timing = infer_window(fe, backbone, head, window)

        raw_idx, ema_idx, maj_idx, ema = update(probs)

        print(
            f"t={tstamp:5.1f}s "
            f"raw={EMOTIONS[raw_idx]:12s} "
            f"smooth={EMOTIONS[maj_idx]:12s} "
            f"conf={ema[maj_idx]:.2f} "
            f"| total {timing['total']:.1f} ms"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        default="week6_out/best_model.pt",
    )

    parser.add_argument(
        "--source",
        choices=["mic", "sim"],
        default="mic",
    )

    parser.add_argument(
        "--wav",
        help="WAV file for --source sim",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Mic device index. Use 0 for your USB PnP Sound Device.",
    )

    parser.add_argument(
        "--rms_silence",
        type=float,
        default=0.006,
        help="Skip windows below this mic RMS.",
    )

    parser.add_argument(
        "--target_rms",
        type=float,
        default=0.03,
        help="Normalize live speech to this RMS before inference.",
    )

    args = parser.parse_args()

    fe, backbone, head = load(args.ckpt)

    if args.source == "mic":
        run_mic(fe, backbone, head, args)
    else:
        if not args.wav:
            parser.error("--source sim requires --wav")

        run_sim(fe, backbone, head, args)


if __name__ == "__main__":
    main()
