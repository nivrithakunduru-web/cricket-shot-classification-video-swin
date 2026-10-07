# Cricket Shot Classification using Video Swin Transformer

A deep learning project for classifying cricket batting shots from video clips using a pretrained **Video Swin Transformer (Swin3D-T)**.

The project investigates different training strategies for video-based cricket shot classification, starting with a baseline Video Swin Transformer and later extending it with spatial augmentation, temporal augmentation, and regularization.

---

## Shot Classes

The dataset contains 10 cricket shot classes:

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

Total videos: **1,888**

| Split | Videos |
|------|------:|
| Train | 1,321 |
| Validation | 283 |
| Test | 284 |
| **Total** | **1,888** |

---

## Model

The baseline model uses:

- **Architecture:** Swin3D-T
- **Framework:** PyTorch / Torchvision
- **Pretrained weights:** Kinetics-400
- **Input frames:** 32
- **Input resolution:** 224 × 224
- **Output classes:** 10

The original 400-class classification head of the Kinetics-400 pretrained model is replaced with a 10-class classification layer for cricket shot recognition.

---

## Video Preprocessing

Each video is converted into a fixed-length clip before being passed to the model.

Pipeline:

```text
Video
  ↓
Read frames
  ↓
Uniformly sample 32 frames
  ↓
Resize
  ↓
Center crop to 224 × 224
  ↓
Normalize
  ↓
[C, T, H, W]
  ↓
Video Swin Transformer

