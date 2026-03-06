# 🌸 Flower Image Classifier — PyTorch Deep Learning Project

> A production-style image classification pipeline built with **PyTorch** and **transfer learning** (VGG16/VGG13).  
> Trains on 102 flower categories and ships as a ready-to-use **command-line application**.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Udacity](https://img.shields.io/badge/Udacity-AI%20Programming%20with%20Python-02b3e4)](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training-a-model)
  - [Prediction](#predicting-a-flower)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Rubric & Checklist](#rubric--checklist)
- [License](#license)

---

## Overview

This project is part of Udacity's **AI Programming with Python Nanodegree (ND089)**. It demonstrates how to:

1. Build an **image classifier** in a Jupyter Notebook using PyTorch
2. Convert that classifier into a **reusable command-line app** (`train.py` + `predict.py`)

The model uses **transfer learning** — a pretrained VGG network whose convolutional layers are frozen, while a custom fully-connected classifier is trained on the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).

**Key skills demonstrated:**
- Transfer learning with `torchvision.models`
- Data augmentation and normalization pipelines
- GPU-accelerated training via CUDA
- Model checkpointing and loading
- Argparse-based CLI design
- PIL image preprocessing for inference

---

## Project Structure

```
flower-image-classifier/
│
├── train.py                  # CLI script: train a new model
├── predict.py                # CLI script: classify an image
├── model_utils.py            # Shared utilities: data loading, model building, checkpointing
├── cat_to_name.json          # Mapping of category indices → flower names (102 classes)
├── Image_Classifier_Project.ipynb  # Part 1: full development notebook
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## Features

| Feature | Detail |
|---|---|
| **Transfer learning** | VGG16 or VGG13 backbone (frozen weights) |
| **Custom classifier** | Configurable hidden units, ReLU, Dropout, LogSoftmax |
| **Data augmentation** | Random rotation, random crop, horizontal flip |
| **GPU support** | `--gpu` flag for CUDA acceleration in both training and inference |
| **Checkpoint system** | Save and load full model state + `class_to_idx` mapping |
| **Top-K predictions** | Return the top K most probable classes with probabilities |
| **Category name mapping** | JSON file maps numeric labels to human-readable flower names |

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/flower-image-classifier.git
cd flower-image-classifier
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the flower dataset**
```bash
# The dataset is NOT included in this repo (per project instructions).
# Download from:
# https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
wget https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
tar -xzf flower_data.tar.gz
# Expected layout: flowers/train/, flowers/valid/, flowers/test/
```

---

## Usage

### Training a Model

```bash
python train.py <data_directory> [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `data_dir` | positional | — | Path to dataset root (must contain `train/`, `valid/`, `test/`) |
| `--save_dir` | str | `.` | Directory to save the checkpoint |
| `--arch` | str | `vgg16` | Model architecture: `vgg16` or `vgg13` |
| `--learning_rate` | float | `0.001` | Optimizer learning rate |
| `--hidden_units` | int | `1024` | Units in the hidden layer of the classifier |
| `--epochs` | int | `5` | Number of training epochs |
| `--gpu` | flag | off | Enable GPU (CUDA) training |

**Examples:**
```bash
# Basic training on CPU
python train.py flowers/

# Train with VGG13, more epochs, saved to checkpoints/
python train.py flowers/ --arch vgg13 --epochs 10 --save_dir checkpoints/

# Train with GPU acceleration
python train.py flowers/ --gpu --learning_rate 0.0003 --hidden_units 512
```

---

### Predicting a Flower

```bash
python predict.py <image_path> <checkpoint> [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `input` | positional | — | Path to the input image |
| `checkpoint` | positional | — | Path to the saved `.pth` checkpoint |
| `--top_k` | int | `1` | Return top K most likely classes |
| `--category_names` | str | None | Path to JSON file mapping categories to names |
| `--gpu` | flag | off | Use GPU for inference |

**Examples:**
```bash
# Predict the top class
python predict.py flowers/test/1/image_06743.jpg checkpoint.pth

# Return top 5 predictions with flower names
python predict.py flowers/test/1/image_06743.jpg checkpoint.pth \
    --top_k 5 \
    --category_names cat_to_name.json

# GPU inference
python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --gpu --top_k 3
```

**Example output:**
```
Predictions for flowers/test/1/image_06743.jpg:
pink primrose: 87.43%
hard-leaved pocket orchid: 4.12%
globe-flower: 3.01%
```

---

## Model Architecture

```
VGG16 backbone (pretrained on ImageNet, weights frozen)
    └── Custom Classifier:
        ├── Linear(25088 → hidden_units)
        ├── ReLU
        ├── Dropout(p=0.2)
        ├── Linear(hidden_units → 102)
        └── LogSoftmax(dim=1)
```

- **Loss function:** Negative Log Likelihood Loss (`NLLLoss`)  
- **Optimizer:** Adam (applied only to classifier parameters)  
- **Input normalization:** ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`

---

## Dataset

**[102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)** by Nilsback & Zisserman (Oxford VGG Group)

- 102 flower categories
- ~8,000 training images
- Split into `train/`, `valid/`, and `test/` folders
- Each subfolder is named by numeric category ID (mapped to names via `cat_to_name.json`)

---

## Rubric & Checklist

### Part 1 — Development Notebook

| Criteria | Status |
|---|---|
| All packages imported in first cell | ✅ |
| Training data augmented (rotation, crop, flip) | ✅ |
| All data normalized with ImageNet stats | ✅ |
| Data loaded with `ImageFolder` | ✅ |
| DataLoaders created for all splits | ✅ |
| Pretrained network loaded with frozen parameters | ✅ |
| Custom feedforward classifier defined | ✅ |
| Classifier trained, feature network frozen | ✅ |
| Validation loss and accuracy displayed during training | ✅ |
| Test accuracy measured | ✅ |
| Model saved as checkpoint with `class_to_idx` | ✅ |
| Checkpoint loading function implemented | ✅ |
| `process_image` converts PIL image to model-ready tensor | ✅ |
| `predict` returns top-K classes and probabilities | ✅ |
| Sanity check: matplotlib figure with image + top 5 classes | ✅ |

### Part 2 — Command Line Application

| Criteria | Status |
|---|---|
| `train.py` trains a network on a dataset | ✅ |
| Training/validation loss and accuracy printed | ✅ |
| `--arch` allows choosing architecture (vgg16 / vgg13) | ✅ |
| `--learning_rate`, `--hidden_units`, `--epochs` flags | ✅ |
| `--gpu` flag for GPU training | ✅ |
| `predict.py` reads image + checkpoint, prints result | ✅ |
| `--top_k` flag for top K predictions | ✅ |
| `--category_names` flag for human-readable names | ✅ |
| `--gpu` flag for GPU inference | ✅ |

---

## License

This project is submitted as part of Udacity's **AI Programming with Python Nanodegree**. The starter code structure is provided by [Udacity](https://github.com/udacity/aipnd-project); the implementation is original student work.

Released under the [MIT License](LICENSE).
