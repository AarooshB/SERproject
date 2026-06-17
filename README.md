# SERproject

Speech Emotion Recognition (SER) using deep learning and transformer-based speech models.

## Overview

This project explores automatic speech emotion recognition from audio recordings. The goal is to classify speech into emotional categories such as:

* Neutral
* Happy
* Sad
* Angry
* Disgust
* Surprised

The project evaluates both lightweight neural networks and modern pretrained speech models to compare accuracy, computational cost, and suitability for real-time deployment.

## Dataset

### RAVDESS

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) is used for training and evaluation.

Features:

* Professional actors
* Multiple emotional categories
* Balanced dataset
* Widely used SER benchmark

## Models Explored

### CNN-Based Models

* 1D Convolutional Neural Networks
* MFCC feature inputs
* Batch normalization
* Dropout regularization
* Global average pooling

### DistilHuBERT

A lightweight transformer model pretrained on large-scale speech data.

Advantages:

* Smaller than Wav2Vec2
* Faster inference
* Suitable for edge devices
* Strong speech representations

### Wav2Vec2

Large pretrained speech representation model.

Advantages:

* State-of-the-art speech features
* Strong performance on SER tasks
* Transfer learning from large speech corpora

## Feature Extraction

Audio preprocessing includes:

* Resampling
* Normalization
* MFCC extraction
* Data augmentation

  * Noise injection
  * Time shifting

## Training

Experiments include:

* Cross-validation
* Weighted cross-entropy loss
* Class balancing
* Hyperparameter tuning
* Model comparison

## Results

Metrics evaluated:

* Accuracy
* Confusion matrix
* Precision
* Recall
* F1 score

Special attention is given to confusion between neutral and calm emotions and the effect of combining those classes.

## Deployment Goals

Target platform:

* NVIDIA Jetson Nano

Requirements:

* Real-time inference
* Low memory usage
* Low latency
* Edge deployment capability

## Future Work

* Real-time microphone emotion recognition
* TensorRT optimization
* Larger emotion datasets
* Multimodal emotion recognition
* Personalized speaker adaptation

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Librosa
* NumPy
* Scikit-learn
* Matplotlib

## Author

Aaroosh Balakrishnan

University of California, Los Angeles

