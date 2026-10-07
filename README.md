# Cricket Shot Classification using Video Swin Transformer

A deep learning project for classifying cricket batting shots from video clips using a pretrained **Video Swin Transformer (Swin3D-T)**.

This repository studies different strategies for improving video-based cricket shot classification. The experiments begin with a baseline Video Swin Transformer and are progressively extended with spatial augmentation, temporal augmentation, and regularization.

---

## Project Overview

The objective of this project is to classify cricket batting videos into **10 shot categories** using a Video Swin Transformer.

The experiments are organized sequentially so that the effect of each modification can be evaluated independently.

| Experiment | Description | Status |
|---|---|---|
| 01 | Video Swin Baseline | ✅ Completed |
| 02 | Spatial Augmentation | 🔄 To be added |
| 03 | Temporal Augmentation | 🔄 To be added |
| 04 | Regularization | 🔄 To be added |

---

## Dataset

The dataset contains **1,888 cricket shot videos** belonging to 10 classes.

### Shot Classes

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

### Dataset Split

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

Baseline Model
The baseline experiment uses Swin3D-T from Torchvision.
Component	        Configuration
Architecture	        Swin3D-T
Pretrained weights	Kinetics-400
Number of classes	10
Frames per video	32
Input resolution	224 × 224
Fine-tuning	        Full model
Framework	        PyTorch / Torchvision
The original Kinetics-400 classification layer predicts 400 action classes. It is replaced with a new classification layer for the 10 cricket shot classes.

Video Preprocessing
Each video is converted into a fixed-length clip before being passed to the network.
Input Video
    ↓
Read video frames
    ↓
Uniformly sample 32 frames
    ↓
Convert frames to tensor
    ↓
Resize
    ↓
Center crop to 224 × 224
    ↓
Normalize
    ↓
Video Tensor [C, T, H, W]
    ↓
[3, 32, 224, 224]
    ↓
Video Swin Transformer

Uniform temporal sampling allows videos with different numbers of frames to be converted into a consistent 32-frame input representation.

Training Configuration

Parameter	                       Value
Optimizer	                       AdamW
Loss Function	                  Cross Entropy Loss
Batch Size	                         2
Backbone Learning Rate                  1e-5
Classification Head Learning Rate	1e-4
Weight Decay	                        1e-4
Mixed Precision	                       Enabled
Fine-tuning Strategy	           Full fine-tuning
Pretraining Dataset	             Kinetics-400



Differential learning rates are used during fine-tuning.
The pretrained Video Swin backbone uses a smaller learning rate, while the newly initialized classification head uses a larger learning rate.

Baseline Training Results
The baseline model was trained for 8 epochs.

| Epoch | Train Accuracy | Validation Accuracy | Train Loss | Validation Loss |
|
| 1 | 38.53% | 63.96% | 1.7358 | 1.0256 |
| 2 | 73.96% | 74.91% | 0.7897 | 0.6610 |
| 3 | 89.17% | 77.39% | 0.3863 | 0.7084 |
| 4 | 94.25% | 82.69% | 0.2290 | 0.5118 |
| 5 | 97.50% | 83.75% | 0.1174 | 0.4456 |
| 6 | 98.41% | 87.28% | 0.0837 | 0.4279 |
| 7 | 98.41% | 85.87% | 0.0640 | 0.4607 |
| 8 | 99.02% | 86.57% | 0.0500 | 0.5066 |
The highest validation accuracy was obtained at Epoch 6.
After Epoch 6, training accuracy continued to remain very high while validation performance stopped improving, indicating increasing overfitting.
Therefore, the Epoch 6 checkpoint was selected as the final baseline model.

Final Baseline Test Results
The selected Epoch 6 checkpoint was evaluated on the test set containing 284 videos.
Metric	Result
Test Loss	0.3554
Test Accuracy	88.03%
Correct Predictions	250
Wrong Predictions	34
Total Test Videos	284
Macro F1 Score	0.8798
Weighted F1 Score	0.8804

Hardware
The baseline experiment was trained on an NVIDIA Tesla T4 GPU using Kaggle.
Automatic Mixed Precision (AMP) was used during training to reduce GPU memory usage and improve computational efficiency.

Experiments and Future Work
The baseline establishes the initial performance of the Video Swin Transformer.
The following experiments will extend this baseline:
02 — Spatial Augmentation
Introduce spatial transformations to improve robustness to variations in appearance and framing.
03 — Temporal Augmentation
Introduce variation in temporal sampling to improve robustness to differences in shot timing and motion.
04 — Regularization
Apply regularization strategies to reduce overfitting and improve generalization.
After completing all experiments, their validation and test performance will be compared to determine the most effective configuration.
