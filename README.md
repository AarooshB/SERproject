# SERproject

Real-time Speech Emotion Recognition (SER) using deep learning and transformer-based speech models.

This project explores automatic speech emotion recognition from audio recordings using both traditional deep learning architectures and pretrained speech transformers. The primary goal is to build a model that can classify human emotions from speech while remaining lightweight enough for real-time deployment on NVIDIA Jetson devices.

---

## Highlights

* Speaker-independent emotion recognition using the RAVDESS dataset
* Evaluated multiple CNN and transformer-based architectures
* Achieved **74.8% Macro F1** using DistilHuBERT embeddings with an MLP classifier
* Built a real-time microphone inference pipeline
* Optimized for deployment on NVIDIA Jetson edge devices

---

## Project Pipeline

```text
Audio Input
     │
     ▼
Preprocessing
(Resample, Normalize)
     │
     ▼
Feature Extraction
(MFCC / DistilHuBERT)
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
Live Emotion Output
```

---

## Dataset

### RAVDESS

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) is used as the primary benchmark dataset.

**Features**

* 24 professional actors
* High-quality speech recordings
* Balanced emotion classes
* Widely used benchmark for Speech Emotion Recognition

**Emotions Classified**

* Neutral / Calm
* Happy
* Sad
* Angry
* Disgust
* Surprised

---

## Model Evolution

This project was developed through several iterations, with each version addressing weaknesses identified in the previous model.

### Model 1

* MFCC features
* Lightweight 1D CNN
* Batch Normalization
* Dropout
* Global Average Pooling

**Result:** Established the initial baseline.

---

### Model 2

Changes:

* Reduced model width to decrease overfitting

**Result:** Performance decreased.

---

### Model 3

Improvements:

* Combined Neutral and Calm into one class
* Better class balancing
* Weighted Cross Entropy loss
* Improved data augmentation

**Result:** Improved overall performance.

---

### Model 4

Added:

* BiGRU temporal modeling
* Attentive Statistics Pooling
* SpecAugment
* Time masking
* Feature masking

**Result**

* No significant improvement
* Continued confusion between Happy, Sad, and Neutral emotions

---

### Model 5

Added:

* CREMA-D dataset
* Pitch features
* Speaker-independent cross validation

**Result**

* Improved validation performance
* Better recognition of Sad emotion
* Limited generalization on unseen speakers

---

### Week 6 (Current Model)

Major improvements:

* DistilHuBERT pretrained embeddings
* Stronger MLP classifier
* LayerNorm
* Dropout
* Feature Dropout
* Speaker-aware cross validation
* Focal Loss
* Weighted Cross Entropy
* Improved augmentation pipeline
* Real-time inference support

**Result:** Best-performing model to date.

---

## Feature Extraction

### CNN Models

* MFCCs
* Delta MFCCs
* Delta-Delta MFCCs
* Normalization

### DistilHuBERT Models

Instead of handcrafted acoustic features, the latest models use frozen DistilHuBERT embeddings extracted from raw speech.

**Advantages**

* Rich pretrained speech representations
* Faster than Wav2Vec2
* Suitable for edge deployment
* Strong transfer learning performance

---

## Training Strategy

Training includes:

* Speaker-independent cross validation
* Weighted Cross Entropy
* Focal Loss
* Early stopping
* Learning rate scheduling
* Data augmentation

**Augmentations**

* Noise injection
* Gain augmentation

Validation data is kept completely clean to prevent data leakage.

---

## Results

### Final Week 6 Results

| Metric                    |                                       Value |
| :------------------------ | ------------------------------------------: |
| Accuracy                  |                                   **74.7%** |
| Pooled Macro F1           |                                  **0.7500** |
| Cross Validation Macro F1 |                         **0.7480 ± 0.0691** |
| Evaluation                | 6-fold Speaker-Independent Cross Validation |

### Per-Class Performance

| Emotion        | Precision | Recall | F1 Score |
| :------------- | --------: | -----: | -------: |
| Neutral / Calm |    0.8405 | 0.6771 |   0.7500 |
| Happy          |    0.6147 | 0.6979 |   0.6537 |
| Sad            |    0.5702 | 0.6771 |   0.6190 |
| Angry          |    0.8182 | 0.8438 |   0.8308 |
| Disgust        |    0.8764 | 0.8125 |   0.8432 |
| Surprised      |    0.7990 | 0.8073 |   0.8031 |

The DistilHuBERT model significantly outperformed the earlier CNN-based approaches while maintaining low inference cost suitable for real-time deployment.

---

## Repository Structure

```text
SERproject/
│
├── extract_embeddings_v6.py
├── train_v6.py
├── model_v6.py
├── live_infer_v6.py
├── week6_out/
├── README.md
└── requirements.txt
```

---

## Running the Project

### 1. Extract DistilHuBERT Embeddings

```bash
python extract_embeddings_v6.py
```

### 2. Train the Model

```bash
python train_v6.py \
    --emb ravdess_embeddings_v6.npz \
    --head mlp2 \
    --loss focal \
    --gamma 1.5 \
    --folds 6
```

### 3. Run Live Inference

```bash
python live_infer_v6.py \
    --source mic \
    --ckpt week6_out/best_model.pt
```

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* DistilHuBERT
* Librosa
* NumPy
* Scikit-learn
* Matplotlib

---

## Future Work

* Improve Happy emotion recognition
* Reduce confusion between Neutral and Sad
* Fine-tune the DistilHuBERT backbone
* Incorporate additional speech emotion datasets
* Optimize inference using TensorRT
* Quantize the model for faster edge deployment
* Extend the system to multimodal emotion recognition using audio and facial expressions

---

## Author

**Aaroosh Balakrishnan**

University of California, Los Angeles (UCLA)

GitHub: https://github.com/AarooshB/SERproject
