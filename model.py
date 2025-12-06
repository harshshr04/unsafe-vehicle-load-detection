import torch
import torch.nn as nn
from torchvision import models

class VehicleLoadClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(VehicleLoadClassifier, self).__init__()
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes=2, pretrained=True):
    """Create and return the model"""
    model = VehicleLoadClassifier(num_classes=num_classes, pretrained=pretrained)
    return model
