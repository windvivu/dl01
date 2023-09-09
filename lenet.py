import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Convolutional layer 1
        # output size = (input_size - kernel_size + 2*padding)/stride + 1
        # output size = (32 - 5 + 2*0)/1 + 1 = 28
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        # output size = (input_size - kernel_size)/stride + 1
        # output size = (28 - 2)/2 + 1 = 14
        self.conv2 = nn.Conv2d(6, 16, 5)  # Convolutional layer 2
        # output size = (14 - 5 + 2*0)/1 + 1 = 10

        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))

        self.fc1 = nn.Linear(16 * 5  * 5 , 120)  # Fully connected layer 1
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer 2
        self.fc3 = nn.Linear(84, num_classes)  # Fully connected layer 3

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.shape[0], -1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    input = torch.randn(8, 3, 224, 224)
    lenet_model = LeNet(num_classes=10)
    output = lenet_model(input)
    print(output.shape)

