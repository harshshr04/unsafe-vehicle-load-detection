"""
Demo script to show the vehicle load detection system in action
"""
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from model import get_model
from torchvision import transforms
import random

def create_sample_images():
    """Create sample images for demonstration"""
    print("Creating sample training images...")
    
    # Create directories
    os.makedirs('data/train/safe', exist_ok=True)
    os.makedirs('data/train/unsafe', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    # Generate sample images
    for i in range(10):
        # Safe load images (balanced, centered)
        img = Image.new('RGB', (400, 300), color=(100, 150, 200))
        draw = ImageDraw.Draw(img)
        
        # Draw vehicle
        draw.rectangle([100, 150, 300, 250], fill=(80, 80, 80), outline=(0, 0, 0), width=2)
        # Draw balanced load
        draw.rectangle([120, 100, 280, 150], fill=(139, 69, 19), outline=(0, 0, 0), width=2)
        draw.text((150, 20), "SAFE LOAD", fill=(0, 255, 0))
        
        img.save(f'data/train/safe/safe_{i}.jpg')
    
    for i in range(10):
        # Unsafe load images (unbalanced, overloaded)
        img = Image.new('RGB', (400, 300), color=(100, 150, 200))
        draw = ImageDraw.Draw(img)
        
        # Draw vehicle
        draw.rectangle([100, 150, 300, 250], fill=(80, 80, 80), outline=(0, 0, 0), width=2)
        # Draw unbalanced/overloaded cargo
        draw.rectangle([120, 50, 320, 150], fill=(139, 69, 19), outline=(255, 0, 0), width=3)
        draw.polygon([(120, 50), (150, 30), (290, 30), (320, 50)], fill=(160, 82, 45))
        draw.text((140, 20), "UNSAFE LOAD", fill=(255, 0, 0))
        
        img.save(f'data/train/unsafe/unsafe_{i}.jpg')
    
    # Create test image
    img = Image.new('RGB', (400, 300), color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 150, 300, 250], fill=(80, 80, 80), outline=(0, 0, 0), width=2)
    draw.rectangle([120, 60, 310, 150], fill=(139, 69, 19), outline=(255, 0, 0), width=3)
    img.save('data/test/test_vehicle.jpg')
    
    print("✓ Sample images created!")
    print(f"  - 10 safe load images in data/train/safe/")
    print(f"  - 10 unsafe load images in data/train/unsafe/")
    print(f"  - 1 test image in data/test/")

def quick_train():
    """Quick training demonstration"""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    from train import train_model
    
    model, history = train_model(
        data_dir='data/train',
        epochs=5,  # Quick demo
        batch_size=4,
        lr=0.001
    )
    
    print("\n✓ Training completed!")
    print(f"  - Final Loss: {history['loss'][-1]:.4f}")
    print(f"  - Final Accuracy: {history['accuracy'][-1]:.2f}%")
    print(f"  - Model saved to: models/best_model.pth")
    
    return model, history

def test_prediction():
    """Test prediction on sample image"""
    print("\n" + "="*60)
    print("TESTING PREDICTION")
    print("="*60)
    
    from predict import predict_image
    
    test_image = 'data/test/test_vehicle.jpg'
    
    if not os.path.exists('models/best_model.pth'):
        print("Error: Model not found. Please train first.")
        return
    
    result = predict_image(test_image)
    
    print(f"\nTest Image: {test_image}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print("\nProbabilities:")
    print(f"  Safe Load: {result['probabilities']['safe']:.2f}%")
    print(f"  Unsafe Load: {result['probabilities']['unsafe']:.2f}%")
    
    return result

def show_project_structure():
    """Display project structure"""
    print("\n" + "="*60)
    print("PROJECT STRUCTURE")
    print("="*60)
    
    structure = """
vehicle-load-detection/
├── model.py                    # Model architecture (ResNet50)
├── train.py                    # Training script
├── predict.py                  # Inference script
├── demo.py                     # This demo file
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── vehicle_load_detection.ipynb # Jupyter notebook
├── data/
│   ├── train/
│   │   ├── safe/              # Safe load images
│   │   └── unsafe/            # Unsafe load images
│   └── test/                  # Test images
└── models/
    ├── best_model.pth         # Trained model
    └── training_history.json  # Training metrics
    """
    print(structure)

def main():
    print("\n" + "="*60)
    print("VEHICLE LOAD DETECTION - DEMO")
    print("="*60)
    
    # Show structure
    show_project_structure()
    
    # Step 1: Create sample data
    print("\n[Step 1/3] Creating sample dataset...")
    create_sample_images()
    
    # Step 2: Train model
    print("\n[Step 2/3] Training model (quick demo with 5 epochs)...")
    model, history = quick_train()
    
    # Step 3: Test prediction
    print("\n[Step 3/3] Testing prediction...")
    result = test_prediction()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Replace sample images with real vehicle images")
    print("2. Train with more epochs: python train.py")
    print("3. Test on your images: python predict.py --image path/to/image.jpg")
    print("4. Use Jupyter notebook: jupyter notebook vehicle_load_detection.ipynb")
    print("\n")

if __name__ == '__main__':
    main()
