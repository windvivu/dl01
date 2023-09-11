import torch.nn as nn
import torch
import torchvision.models as models
from torchsummary import summary

class ResNetBase(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.backbone(x).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))    


if __name__ == '__main__':
    model = ResNetBase().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    summary(model, input_size=(3, 224, 224))