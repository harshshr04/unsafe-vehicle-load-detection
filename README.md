# Unsafe Vehicle Load Detection

AI-powered system to detect unsafe vehicle loads using computer vision and deep learning.

## Features
- Image classification for safe/unsafe vehicle loads
- Pre-trained model support (ResNet, EfficientNet)
- Real-time inference
- Training pipeline with data augmentation

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train model
python train.py

# Run inference
python predict.py --image path/to/image.jpg

# Run Jupyter notebook
jupyter notebook vehicle_load_detection.ipynb
```

## Project Structure
- `train.py` - Model training script
- `predict.py` - Inference script
- `model.py` - Model architecture
- `data/` - Dataset directory
- `models/` - Saved models
- `vehicle_load_detection.ipynb` - Interactive notebook
