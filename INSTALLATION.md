# Installation Guide

## Prerequisites

### 1. Install Python (Required)

**Windows:**
1. Go to https://www.python.org/downloads/
2. Download Python 3.8 or higher
3. Run installer
4. ✅ **IMPORTANT:** Check "Add Python to PATH"
5. Click "Install Now"

**Verify Installation:**
```bash
python --version
```
Should show: `Python 3.x.x`

---

## Setup Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- PyTorch (deep learning framework)
- torchvision (computer vision tools)
- Pillow (image processing)
- OpenCV (computer vision)
- NumPy (numerical computing)
- Matplotlib (visualization)
- scikit-learn (machine learning utilities)

### Step 2: Verify Installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"
```

---

## Quick Test

### Option 1: Run Demo (Recommended)
```bash
python demo.py
```

This will:
1. Create sample training data
2. Train a quick model (5 epochs)
3. Test prediction
4. Show results

### Option 2: Manual Setup
```bash
# Create directories
mkdir -p data/train/safe
mkdir -p data/train/unsafe
mkdir -p data/test

# Add your images to the folders
# Then train
python train.py

# Then predict
python predict.py --image data/test/your_image.jpg
```

### Option 3: Jupyter Notebook
```bash
jupyter notebook vehicle_load_detection.ipynb
```

---

## GPU Support (Optional but Faster)

### Check if CUDA is available:
```python
import torch
print(torch.cuda.is_available())
```

### Install CUDA-enabled PyTorch:
Visit: https://pytorch.org/get-started/locally/

Example for CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Troubleshooting

### Error: "python is not recognized"
- Python not in PATH
- Reinstall Python with "Add to PATH" checked
- Or use full path: `C:\Python39\python.exe`

### Error: "pip is not recognized"
```bash
python -m pip install -r requirements.txt
```

### Error: "No module named 'torch'"
```bash
pip install torch torchvision
```

### Error: "CUDA out of memory"
- Reduce batch_size in train.py
- Use CPU instead: Set `device = 'cpu'`

---

## File Checklist

After setup, you should have:
```
✅ model.py
✅ train.py
✅ predict.py
✅ demo.py
✅ requirements.txt
✅ README.md
✅ vehicle_load_detection.ipynb
✅ data/ (folder)
✅ models/ (folder, created during training)
```

---

## Ready to Go!

Run the demo:
```bash
python demo.py
```

Or start training with your own data:
```bash
python train.py
```

🎉 **You're all set!**
