import torch.nn as nn
from torchvision import models

class ResNet50FineTune(nn.Module):
    def __init__(self, num_classes=143):
        super(ResNet50FineTune, self).__init__()
        # Load a pre-trained ResNet50 model
        self.model = models.resnet50(pretrained=True)
        
        # Freeze all layers in the network
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Forward pass through the modified ResNet50
        return self.model(x)
