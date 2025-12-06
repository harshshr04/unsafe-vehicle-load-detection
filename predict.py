import torch
from torchvision import transforms
from PIL import Image
import argparse
from model import get_model
import os

def predict_image(image_path, model_path='models/best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_model(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(1).item()
        confidence = probabilities[0][predicted_class].item()
    
    classes = ['Safe Load', 'Unsafe Load']
    result = {
        'prediction': classes[predicted_class],
        'confidence': confidence * 100,
        'probabilities': {
            'safe': probabilities[0][0].item() * 100,
            'unsafe': probabilities[0][1].item() * 100
        }
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Predict vehicle load safety')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Path to model')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f'Error: Image not found at {args.image}')
        return
    
    if not os.path.exists(args.model):
        print(f'Error: Model not found at {args.model}')
        return
    
    result = predict_image(args.image, args.model)
    
    print('\n' + '='*50)
    print(f'Prediction: {result["prediction"]}')
    print(f'Confidence: {result["confidence"]:.2f}%')
    print('\nProbabilities:')
    print(f'  Safe Load: {result["probabilities"]["safe"]:.2f}%')
    print(f'  Unsafe Load: {result["probabilities"]["unsafe"]:.2f}%')
    print('='*50 + '\n')

if __name__ == '__main__':
    main()
