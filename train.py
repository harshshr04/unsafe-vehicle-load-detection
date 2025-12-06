import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from model import get_model
import json

class VehicleLoadDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['safe', 'unsafe']
        self.images = []
        self.labels = []
        
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model(data_dir='data/train', epochs=20, batch_size=16, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset and dataloader
    train_dataset = VehicleLoadDataset(data_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f'Training samples: {len(train_dataset)}')
    
    # Model, loss, optimizer
    model = get_model(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # Training loop
    best_loss = float('inf')
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        
        print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%')
        
        scheduler.step(epoch_loss)
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
            print('Model saved!')
    
    # Save training history
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f)
    
    print('Training completed!')
    return model, history

if __name__ == '__main__':
    train_model()
