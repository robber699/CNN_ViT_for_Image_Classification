
# Deep Learning for Visual Computing - Assignment 1

This repository contains the implementation of Assignment 1 for the "Deep Learning for Visual Computing" course. The assignment focuses on deep learning techniques for image classification using the CIFAR-10 dataset, implemented with PyTorch.

## Overview
The project is organized into several parts, each covering key aspects of a deep learning pipeline, from dataset handling to model training and evaluation. Below is a brief description of the key components:

### 1. Dataset Preparation
The CIFAR-10 dataset is used for training, validation, and testing. The dataset is organized into training (40,000 images), validation (10,000 images), and test (10,000 images) subsets. The `CIFAR10Dataset` class handles loading and preprocessing of the images, ensuring they are in the correct format and order (RGB channels).

### 2. Metrics Implementation
The `Accuracy` class is implemented to track both overall and per-class accuracy during model training and evaluation. This is crucial for assessing model performance, especially in scenarios involving imbalanced datasets.

### 3. Model - ResNet18
The ResNet18 model from PyTorch's `torchvision` library is used as the baseline for image classification. The project includes a custom wrapper class, `DeepClassifier`, which handles model loading, saving, and evaluation.

### 4. Training Loop
The `ImgClassificationTrainer` class encapsulates the training process, including the management of training epochs, validation, and checkpointing of the best-performing model. Various metrics are logged during training to monitor progress.

### 5. Experimentation and Results
Experiments were conducted using the ResNet18 model, with different hyperparameters such as learning rate, optimizer (AdamW), and learning rate scheduler (ExponentialLR). Data augmentation techniques like random cropping and mirroring, as well as regularization methods like weight decay and dropout, were explored to reduce overfitting.

### 6. Custom Models
In addition to ResNet18, two custom models were implemented:
- A simple CNN model in `cnn.py`.
- A Vision Transformer (ViT) model in `vit.py`, which was adapted from existing implementations.

## Report
The repository also includes a report summarizing the findings from the experiments. The report discusses the impact of different techniques on model performance, compares CNNs and ViTs, and analyzes the results obtained from various model configurations.
