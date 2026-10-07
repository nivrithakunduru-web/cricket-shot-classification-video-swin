# Cricket Shot Classification using Video Swin Transformer

A deep learning project for classifying cricket batting shots from video clips using a pretrained **Video Swin Transformer (Swin3D-T)**.

This repository experimentally evaluates different strategies for cricket shot classification:

1. Baseline Video Swin Transformer
2. Spatial Augmentation
3. Temporal Augmentation
4. Regularization

Each experiment is maintained separately so that its effect on model performance can be evaluated clearly.

---

## Project Overview

The objective is to classify cricket batting videos into **10 shot categories** using a Video Swin Transformer pretrained on Kinetics-400.

### Experiment Status

| Experiment | Description | Status |
|---|---|---|
| 01 | Video Swin Baseline | ✅ Completed |
| 02 | Spatial Augmentation | ✅ Completed |
| 03 | Temporal Augmentation | ✅ Completed |
| 04 | Regularization | 🔄 To be added |

---

# Dataset

The dataset contains **1,888 cricket shot videos** belonging to 10 classes.

## Shot Classes

1. Cover
2. Defense
3. Flick
4. Hook
5. Late Cut
6. Lofted
7. Pull
8. Square Cut
9. Straight
10. Sweep

## Dataset Split

| Split | Videos |
|---|---:|
| Train | 1,321 |
| Validation | 283 |
| Test | 284 |
| **Total** | **1,888** |

The video dataset itself is not stored in this repository because of its size.

Expected dataset organization:

```text
dataset/
├── train/
│   ├── cover/
│   ├── defense/
│   ├── flick/
│   ├── hook/
│   ├── late_cut/
│   ├── lofted/
│   ├── pull/
│   ├── square_cut/
│   ├── straight/
│   └── sweep/
├── val/
└── test/
```

---

# Model Architecture

All current experiments use **Swin3D-T** from Torchvision.

| Component | Configuration |
|---|---|
| Architecture | Swin3D-T |
| Pretrained Weights | Kinetics-400 |
| Number of Classes | 10 |
| Frames per Video | 32 |
| Input Resolution | 224 × 224 |
| Fine-tuning | Full model |
| Framework | PyTorch / Torchvision |

The original Kinetics-400 classification head predicts 400 action classes.

It is replaced with a new classification layer:

```text
Original:
Linear(768 → 400)

Modified:
Linear(768 → 10)
```

This allows the pretrained Video Swin Transformer to classify the 10 cricket shot categories.

---

# Common Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Loss Function | Cross Entropy Loss |
| Batch Size | 2 |
| Initial Backbone LR | 1e-5 |
| Initial Classification Head LR | 1e-4 |
| Weight Decay | 1e-4 |
| Mixed Precision | Enabled |
| Fine-tuning Strategy | Full fine-tuning |
| Pretraining Dataset | Kinetics-400 |

Differential learning rates are used during fine-tuning.

The pretrained backbone receives a smaller learning rate, while the newly initialized classification head receives a larger learning rate.

---

# Experiment 01 — Baseline Video Swin

The baseline establishes the reference performance of the pretrained Video Swin Transformer.

No custom spatial or temporal augmentation is applied.

## Baseline Preprocessing

```text
Input Video
    ↓
Read complete video
    ↓
Uniformly sample 32 frames
    ↓
Resize
    ↓
Center Crop
    ↓
224 × 224
    ↓
Normalize
    ↓
Tensor [3, 32, 224, 224]
    ↓
Video Swin Transformer
```

## Baseline Training Results

The baseline was trained for **8 epochs**.

| Epoch | Train Accuracy | Validation Accuracy | Train Loss | Validation Loss |
|---:|---:|---:|---:|---:|
| 1 | 38.53% | 63.96% | 1.7358 | 1.0256 |
| 2 | 73.96% | 74.91% | 0.7897 | 0.6610 |
| 3 | 89.17% | 77.39% | 0.3863 | 0.7084 |
| 4 | 94.25% | 82.69% | 0.2290 | 0.5118 |
| 5 | 97.50% | 83.75% | 0.1174 | 0.4456 |
| **6** | **98.41%** | **87.28%** | **0.0837** | **0.4279** |
| 7 | 98.41% | 85.87% | 0.0640 | 0.4607 |
| 8 | 99.02% | 86.57% | 0.0500 | 0.5066 |

The highest validation accuracy occurred at **Epoch 6**.

Therefore, the Epoch 6 checkpoint was selected for final testing.

## Baseline Final Results

| Metric | Result |
|---|---:|
| Best Epoch | **6** |
| Best Validation Accuracy | **87.28%** |
| Test Loss | **0.3554** |
| Test Accuracy | **88.03%** |
| Correct Predictions | **250 / 284** |
| Wrong Predictions | 34 |
| Macro F1 | **0.8798** |
| Weighted F1 | **0.8804** |

### Accuracy Curve

![Baseline Accuracy Curve](results/baseline/accuracy_curve.png)

### Loss Curve

![Baseline Loss Curve](results/baseline/loss_curve.png)

### Confusion Matrix

![Baseline Confusion Matrix](results/baseline/confusion_matrix.png)

Detailed metrics:

```text
results/baseline/metrics.txt
```

---

# Experiment 02 — Spatial Augmentation

This experiment investigates whether introducing spatial variation during training improves generalization.

Spatial augmentation is applied **only to training videos**.

Validation and test preprocessing remain deterministic.

## Spatial Augmentation Strategy

| Augmentation | Configuration |
|---|---|
| Random Resized Crop | Scale 0.80–1.00 |
| Horizontal Flip | Probability 0.5 |
| Brightness | ±0.15 |
| Contrast | ±0.15 |
| Saturation | ±0.10 |

The same randomly generated spatial transformation parameters are applied consistently across all frames of a clip.

This preserves temporal consistency while modifying spatial appearance.

## Spatial Training Results

The model was trained for **7 epochs**.

| Epoch | Train Accuracy | Validation Accuracy | Train Loss | Validation Loss |
|---:|---:|---:|---:|---:|
| 1 | 44.28% | 57.60% | 1.6124 | 1.1599 |
| 2 | 72.07% | 67.49% | 0.8402 | 0.8609 |
| 3 | 83.72% | 67.49% | 0.5387 | 0.8979 |
| 4 | 86.90% | 76.33% | 0.3931 | 0.6973 |
| 5 | 90.16% | 71.02% | 0.2860 | 0.9015 |
| **6** | **94.85%** | **77.39%** | **0.1927** | **0.6222** |
| 7 | 95.08% | 74.91% | 0.1757 | 0.7836 |

Epoch 6 achieved the highest validation accuracy and was selected for final testing.

## Spatial Augmentation Final Results

| Metric | Result |
|---|---:|
| Best Epoch | **6** |
| Best Validation Accuracy | **77.39%** |
| Test Loss | **0.5284** |
| Test Accuracy | **80.99%** |
| Correct Predictions | **230 / 284** |
| Wrong Predictions | 54 |
| Macro F1 | **0.8082** |
| Weighted F1 | **0.8092** |

### Accuracy Curve

![Spatial Accuracy Curve](results/spatial_augmentation/accuracy_curve.png)

### Loss Curve

![Spatial Loss Curve](results/spatial_augmentation/loss_curve.png)

### Confusion Matrix

![Spatial Confusion Matrix](results/spatial_augmentation/confusion_matrix.png)

Detailed metrics:

```text
results/spatial_augmentation/metrics.txt
```

---

# Experiment 03 — Temporal Augmentation

The temporal augmentation experiment introduces variation in **which frames are selected from each training video**.

Instead of always selecting the same deterministic frame positions, the training video timeline is divided into **32 temporal segments**.

One frame is randomly selected from each segment.

## Temporal Training Sampling

```text
Complete Video Timeline
        ↓
Divide timeline into 32 segments
        ↓
Randomly choose one frame
from each segment
        ↓
Preserve chronological order
        ↓
32-frame training clip
        ↓
Video Swin Transformer
```

Therefore, the same training video can produce different 32-frame clips on different accesses.

This provides temporal variation while preserving the overall sequence of the cricket shot.

## Validation and Test Sampling

Temporal randomness is **not** applied during validation or testing.

```text
Validation / Test Video
        ↓
Deterministic uniform sampling
        ↓
32 frames
        ↓
Video Swin Transformer
```

This ensures reproducible evaluation.

No custom spatial augmentation is introduced in this experiment.

---

## Temporal Training Results

The temporal augmentation model was trained for **10 epochs**.

| Epoch | Train Accuracy | Validation Accuracy | Train Loss | Validation Loss |
|---:|---:|---:|---:|---:|
| 1 | 40.42% | 56.89% | 1.7163 | 1.1819 |
| 2 | 74.34% | 74.56% | 0.8072 | 0.7263 |
| 3 | 86.37% | 80.21% | 0.4328 | 0.5246 |
| 4 | 92.58% | 83.75% | 0.2847 | 0.5198 |
| 5 | 95.53% | 85.87% | 0.1771 | 0.4200 |
| 6 | 97.12% | 86.57% | 0.1120 | 0.4087 |
| 7 | 96.82% | 80.57% | 0.1033 | 0.7537 |
| 8 | 98.56% | 88.69% | 0.0709 | 0.3673 |
| **9** | **99.62%** | **89.05%** | **0.0440** | **0.3642** |
| 10 | 99.17% | 85.51% | 0.0404 | 0.4932 |

Before Epoch 8, the learning rates were reduced:

```text
Backbone:
1e-5 → 5e-6

Classification Head:
1e-4 → 5e-5
```

Epoch 9 achieved the highest validation accuracy of **89.05%** and was selected as the final checkpoint.

An intermediate test was performed after Epoch 8, but training subsequently continued. Therefore, that intermediate result is not used as the final reported result.

---

## Temporal Augmentation Final Results

| Metric | Result |
|---|---:|
| Best Epoch | **9** |
| Best Validation Accuracy | **89.05%** |
| Test Loss | **0.3736** |
| Test Accuracy | **89.44%** |
| Correct Predictions | **254 / 284** |
| Wrong Predictions | 30 |
| Macro F1 | **0.8941** |
| Weighted F1 | **0.8940** |

### Accuracy Curve

![Temporal Accuracy Curve](results/temporal_augmentation/accuracy_curve.png)

### Loss Curve

![Temporal Loss Curve](results/temporal_augmentation/loss_curve.png)

### Confusion Matrix

![Temporal Confusion Matrix](results/temporal_augmentation/confusion_matrix.png)

Detailed metrics:

```text
results/temporal_augmentation/metrics.txt
```

---

# Experiment Comparison

| Experiment | Best Epoch | Best Validation Accuracy | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 6 | 87.28% | 88.03% | 0.8798 | 0.8804 |
| Spatial Augmentation | 6 | 77.39% | 80.99% | 0.8082 | 0.8092 |
| **Temporal Augmentation** | **9** | **89.05%** | **89.44%** | **0.8941** | **0.8940** |
| Regularization | — | — | — | — | — |

---

# Current Findings

## Baseline → Spatial Augmentation

Spatial augmentation reduced performance:

```text
Validation Accuracy
87.28% → 77.39%

Test Accuracy
88.03% → 80.99%
```

The chosen spatial transformations therefore did not improve performance under this experiment configuration.

---

## Baseline → Temporal Augmentation

Temporal augmentation improved performance:

```text
Validation Accuracy
87.28% → 89.05%
Improvement: +1.77 percentage points

Test Accuracy
88.03% → 89.44%
Improvement: +1.41 percentage points
```

Macro F1 also improved:

```text
0.8798 → 0.8941
```

Among the experiments completed so far, **temporal augmentation provides the strongest performance**.

This suggests that introducing variation in temporal frame selection is beneficial for this video-based cricket shot classification task.

---

# Repository Structure

```text
cricket-shot-classification-video-swin/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_video_swin_baseline.ipynb
│   ├── 02_video_swin_spatial_augmentation.ipynb
│   ├── 03_video_swin_temporal_augmentation.ipynb
│   └── 04_video_swin_regularization.ipynb
│
└── results/
    │
    ├── baseline/
    │   ├── accuracy_curve.png
    │   ├── loss_curve.png
    │   ├── confusion_matrix.png
    │   └── metrics.txt
    │
    ├── spatial_augmentation/
    │   ├── accuracy_curve.png
    │   ├── loss_curve.png
    │   ├── confusion_matrix.png
    │   └── metrics.txt
    │
    ├── temporal_augmentation/
    │   ├── accuracy_curve.png
    │   ├── loss_curve.png
    │   ├── confusion_matrix.png
    │   └── metrics.txt
    │
    └── regularization/
```

---

# Installation

Install the required packages using:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- PyTorch
- Torchvision
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

---

# Hardware

The experiments were performed using an **NVIDIA Tesla T4 GPU** in Kaggle.

Automatic Mixed Precision (AMP) was used during training to reduce GPU memory usage and improve computational efficiency.

---

# Next Experiment

## Experiment 04 — Regularization

The next experiment investigates regularization strategies designed to reduce overfitting and improve model generalization.

After the regularization experiment is completed, all four experiments will be compared using:

- Validation accuracy
- Test accuracy
- Test loss
- Macro F1
- Weighted F1
- Training and validation curves
- Class-wise prediction behavior
