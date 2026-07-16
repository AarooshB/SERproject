# Real-Time Speech Emotion Recognition on Edge Devices

Real-time Speech Emotion Recognition (SER) using deep learning and pretrained speech transformers, designed for efficient deployment on edge hardware.

This project investigates how modern speech foundation models can be adapted for real-time emotion recognition while remaining lightweight enough for deployment on embedded devices such as the NVIDIA Jetson series. Throughout development, multiple CNN and hybrid architectures were evaluated before converging on a frozen DistilHuBERT encoder with a lightweight MLP classifier, providing strong accuracy with low inference cost.

**Author:** Aaroosh Balakrishnan
**University:** University of California, Los Angeles (UCLA)

## Demo

**Video Demo**

https://drive.google.com/drive/u/0/folders/1DE4-ThQO9VLnjqozcJXxqJOTWTJzuThR

---

# Features

* Real-time speech emotion recognition from a microphone
* Speaker-independent evaluation
* DistilHuBERT transformer embeddings
* Lightweight MLP classifier for efficient inference
* Six emotion classification
* Live command line inference
* Optimized for future NVIDIA Jetson deployment
* Modular training and inference pipeline

---

# Supported Emotions

The model predicts one of the following six emotions.

| Label | Emotion        |
| ----: | -------------- |
|     0 | Neutral / Calm |
|     1 | Happy          |
|     2 | Sad            |
|     3 | Angry          |
|     4 | Disgust        |
|     5 | Surprised      |

---

# Project Pipeline

```text
Live Microphone
      │
      ▼
Audio Preprocessing
(Resample + Normalize)
      │
      ▼
Frozen DistilHuBERT Encoder
      │
      ▼
768-D Embedding
      │
      ▼
MLP Classifier
      │
      ▼
Emotion Prediction
      │
      ▼
Temporal Smoothing
      │
      ▼
Live Terminal Output
```

---

# Results

## Final Model Performance

| Metric                    |                                       Value |
| ------------------------- | ------------------------------------------: |
| Accuracy                  |                                   **74.7%** |
| Pooled Macro F1           |                                  **0.7500** |
| Cross Validation Macro F1 |                         **0.7480 ± 0.0691** |
| Evaluation                | 6-Fold Speaker Independent Cross Validation |

## Per-Class Performance

| Emotion        | Precision | Recall |     F1 |
| -------------- | --------: | -----: | -----: |
| Neutral / Calm |    0.8405 | 0.6771 | 0.7500 |
| Happy          |    0.6147 | 0.6979 | 0.6537 |
| Sad            |    0.5702 | 0.6771 | 0.6190 |
| Angry          |    0.8182 | 0.8438 | 0.8308 |
| Disgust        |    0.8764 | 0.8125 | 0.8432 |
| Surprised      |    0.7990 | 0.8073 | 0.8031 |

The final DistilHuBERT model substantially outperformed every previous architecture while remaining computationally efficient enough for real-time inference.

---

# Dataset

## RAVDESS

Primary benchmark dataset.

Features:

* 24 professional actors
* Balanced emotion classes
* High-quality recordings
* Speaker-independent evaluation

## CREMA-D

Used during later development to improve robustness and increase dataset diversity.

---

# Model Development

The project evolved through six major iterations.

## Model 1

* MFCC features
* Lightweight 1D CNN
* Batch Normalization
* Dropout
* Global Average Pooling

Result:

Initial working baseline.

---

## Model 2

Changes:

* Smaller CNN architecture

Result:

Reduced overfitting but decreased overall accuracy.

---

## Model 3

Improvements:

* Combined Neutral and Calm
* Improved class balancing
* Weighted Cross Entropy
* Better augmentation

Result:

Improved overall classification performance.

---

## Model 4

Added:

* BiGRU
* Attentive Statistics Pooling
* SpecAugment
* Time masking
* Feature masking

Result:

Minimal improvement. Continued confusion between Happy, Sad, and Neutral.

---

## Model 5

Added:

* CREMA-D
* Pitch features
* Speaker-independent validation

Result:

Improved validation accuracy but limited unseen speaker generalization.

---

## Model 6 (Current)

Major improvements:

* Frozen DistilHuBERT encoder
* LayerNorm MLP classifier
* Feature Dropout
* Speaker-aware cross validation
* Focal Loss
* Weighted Cross Entropy
* Improved augmentation
* Live inference pipeline

Result:

Current production model with the highest overall performance.

---

# Feature Extraction

## Earlier Models

* MFCC
* Delta MFCC
* Delta-Delta MFCC
* Feature normalization

## Current Model

The latest architecture replaces handcrafted features with pretrained DistilHuBERT embeddings extracted directly from raw speech.

Advantages:

* Strong pretrained speech representations
* Faster than Wav2Vec2
* Lightweight inference
* Better transfer learning
* Suitable for edge deployment

---

# Training Strategy

Training uses:

* Speaker-independent cross validation
* Weighted Cross Entropy
* Focal Loss
* Early stopping
* Learning rate scheduling

Augmentation includes:

* Noise injection
* Random gain

Validation data is kept clean to avoid leakage.

---

# Installation

Clone the repository.

```bash
git clone https://github.com/AarooshB/SERproject.git

cd SERproject
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Quick Start

If a trained checkpoint is available, launch live inference directly.

```bash
python models/model_06_distilhubert/live_infer.py \
    --source mic \
    --ckpt models/model_06_distilhubert/results/best_model.pt
```

The application will:

1. Listen through the microphone.
2. Extract DistilHuBERT embeddings.
3. Predict the current emotion.
4. Display predictions in real time.

Example output:

```text
Listening...

Prediction: Happy
Confidence: 0.91

Prediction: Angry
Confidence: 0.88
```

---

# Running on an Audio File

```bash
python models/model_06_distilhubert/live_infer.py \
    --source sim \
    --wav path/to/audio.wav \
    --ckpt models/model_06_distilhubert/results/best_model.pt
```

---

# Training From Scratch

## Step 1

Extract embeddings.

```bash
python models/model_06_distilhubert/extract_embeddings.py \
    --ravdess_dir data/ravdess \
    --include_cremad \
    --cremad_dir data/cremad \
    --n_aug 2
```

---

## Step 2

Train the classifier.

```bash
python models/model_06_distilhubert/train.py \
    --head mlp2 \
    --loss focal \
    --gamma 1.5 \
    --folds 6
```

Training checkpoints are saved to:

```text
models/model_06_distilhubert/results/
```

---

## Step 3

Run live inference.

```bash
python models/model_06_distilhubert/live_infer.py \
    --source mic \
    --ckpt models/model_06_distilhubert/results/best_model.pt
```

---

# Command Line Options

## live_infer.py

| Argument       | Description           |
| -------------- | --------------------- |
| `--source mic` | Live microphone input |
| `--source sim` | Audio file inference  |
| `--wav`        | Path to WAV file      |
| `--ckpt`       | Model checkpoint      |

---

# Repository Structure

```text
SERproject/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── ravdess/
│   ├── all_audio/
│   └── cremad/
│
├── features/
│   ├── mfcc/
│   ├── mfcc_norm/
│   ├── mel/
│   └── v5_norm/
│
├── models/
│   ├── model_01_mfcc_cnn/
│   ├── model_02_mfcc_cnn_narrow/
│   ├── model_03_balanced/
│   ├── model_04_bigru/
│   ├── model_05_cremad/
│   └── model_06_distilhubert/
│       ├── backbone.py
│       ├── model.py
│       ├── extract_embeddings.py
│       ├── train.py
│       ├── live_infer.py
│       ├── embeddings/
│       ├── checkpoints/
│       └── results/
│
├── notebooks/
├── scripts/
└── debug/
```

---

# Technologies

* Python
* PyTorch
* Hugging Face Transformers
* DistilHuBERT
* Librosa
* NumPy
* Scikit-learn
* Matplotlib

---

# Future Work

* Fine-tune the DistilHuBERT backbone
* Improve Happy emotion recognition
* Reduce Neutral and Sad confusion
* Expand training with additional datasets
* TensorRT optimization
* Model quantization for embedded deployment
* Real-time facial emotion fusion
* Full NVIDIA Jetson deployment

---

# Acknowledgments

* RAVDESS Dataset
* CREMA-D Dataset
* Hugging Face Transformers
* PyTorch
* NVIDIA Jetson Platform

---

# License

This project is intended for research and educational purposes.

---

# Contact

**Aaroosh Balakrishnan**

GitHub: https://github.com/AarooshB

Project Repository:

https://github.com/AarooshB/SERproject
