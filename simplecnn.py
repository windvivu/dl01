import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, size=224):
        super().__init__()
        self.sizex = [size, size,0]
       
        self.conv1 = self.make_block(in_channels=3, out_channels=8)
        self.conv2 = self.make_block(in_channels=8, out_channels=16)
        self.conv3 = self.make_block(in_channels=16, out_channels=32)
        self.conv4 = self.make_block(in_channels=32, out_channels=64)
        self.conv5 = self.make_block(in_channels=64, out_channels=128)

        self.in_features = int(self.sizex[0]*self.sizex[1]*self.sizex[2])

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.in_features, out_features=512),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes),
        )

    def make_block(self, in_channels, out_channels):
        self.sizex = [i/2 for i in self.sizex]
        self.sizex[2] = out_channels
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            # output size = (input_size - kernel_size + 2*padding)/stride + 1
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x) 
      
        x = x.view(x.shape[0], -1) # flatten 4d to 2d

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
    
if __name__ == "__main__":
    import torch
    model = SimpleCNN(size=32)
    input = torch.randn(8,3,32,32)
    output = model(input)