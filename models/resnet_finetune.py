import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights

class ResNet50FineTune(nn.Module):
    def __init__(self, num_classes=143):
        super(ResNet50FineTune, self).__init__()
        # Load a pre-trained ResNet50 model with specified weights
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all layers in the network
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
