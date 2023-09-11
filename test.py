import torch.nn as nn
import torch
import torchvision.models as models
from torchsummary import summary

class AlexNetBase(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Chuyển mô hình và đầu vào lên GPU (cuda) nếu có sẵn
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)#.to(self.device)
        self.backbone.classifier[6] = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        # Chuyển dữ liệu đầu vào lên GPU nếu có sẵn
        x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return self.backbone(x)

if __name__ == '__main__':
    model = AlexNetBase().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output.shape)
    summary(model, input_size=(3, 224, 224))
