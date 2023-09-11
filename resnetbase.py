import torch.nn as nn
import torchvision.models as models

class ResNetBase(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.backbone(x)    


if __name__ == '__main__':
    model = ResNetBase()
    for name, param in model.named_parameters():
        print(name, param.shape)