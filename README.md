# SR (Super Resolution) - PyTorch Implementation

This project is a PyTorch implementation of an Super Resolution model. It aims to reconstruct high-resolution images from low-resolution inputs.

## 📌 Key Features

* Based on **SR architecture** with simplified residual blocks (no BatchNorm)
* **PixelShuffle-based upsampling** for 2x or 4x super-resolution
* Uses **L1 loss** for training
* Automatically selects available device (MPS, CUDA, or CPU)
* Saves model checkpoints every 10 epochs (`sr_epoch{n}.pth`)

---

## 🏗️ Model Architecture

```
Input (LR Image)
    ↓
Conv2D (head)
    ↓
[Residual Block × N]
    ↓
Conv2D (body_tail)
    ↓
Global Residual Connection (Skip)
    ↓
Upsampler (PixelShuffle)
    ↓
Conv2D (tail)
    ↓
Output (SR Image)
```

* **ResidualBlock**: Conv → ReLU → Conv → Residual
* **Upsampler**: Conv → PixelShuffle(2x) → ReLU (repeated for scale 4)

---

## 🧠 Training Setup

### ✅ Virtual Environment

```bash
# MAC / Linux
> python3 -m venv SR
> source SR/bin/activate
> pip install torch tqdm

# Windows
> python -m venv SR
> SR\Scripts\activate
> pip install torch tqdm
```

### ✅ Hyperparameters

```python
batch_size = 8
epochs = 100
lr = 1e-4
scale = 4
num_blocks = 16
img_size = 512  # HR image size
```

### ✅ Loss and Optimizer

```python
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

### ✅ Device Selection

Automatically selects MPS, CUDA, or CPU:

```python
device = torch.device("mps" if torch.backends.mps.is_available()
                    else "cuda" if torch.cuda.is_available()
                    else "cpu")
```

---

## 📁 Data Structure

```
data/
├── LR/        # Low-resolution images (already downsampled by 4x)
└── HR/        # High-resolution ground truth images
```

### ✅ Dataset Class

* HR images are resized to `img_size`
* LR images are used as-is (assumed pre-downsampled)

---

## 🚀 Training Execution

```bash
> python3 train.py
```

---

## 📦 Inference (Not tested yet!)

> An `inference.py` script will be added soon. It will load a trained model and perform super-resolution on a given LR image.

---
