import torch
import torch.nn.functional as F
import torch.nn as nn


"""
I will use bottleneck defined in resnet
Main advantages of bottleneck is:
    increases number of layers
    decreases number of parameters

Bottleneck is essentially, a 2 times 3x3 conv2d blocks replaced by 1x1,3x3,1x1 blocks    
"""
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(kernel_size=3,in_channels=in_channels, out_channels=out_channels, stride=1,padding=(1,1))

def conv1x1(in_channels,out_channels):
    return nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1)

def conv2d_block(in_channels,out_channels,kernel_size=3,stride=1,padding=(1,1),bn=False):
    layers = [nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,out_channels=out_channels, stride=stride,padding=padding),
              nn.ReLU()
              ]
    if bn:
        layers.append(nn.BatchNorm2d(num_features=out_channels))

    return nn.Sequential(*layers)

class BottleNeck(nn.Module):

    def __init__(self, num_channels, width):
        # first 1x1 block
        super().__init__()
        self.conv1x1_1 = conv1x1(in_channels=num_channels, out_channels=width)
        self.bn1 = nn.BatchNorm2d(num_features=width)
        self.relu1 = nn.ReLU()

        # 3x3 block
        self.conv3x3 = conv3x3(width,width)
        self.bn2 = nn.BatchNorm2d(num_features=width)
        self.relu2 = nn.ReLU()

        # second 1x1 block
        self.conv1x1_2 = conv1x1(in_channels=width, out_channels=num_channels)
        self.bn3 = nn.BatchNorm2d(num_features=num_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        # save input for residual block
        inp = x
        # forward through first 1x1 block
        x = self.conv1x1_1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # forward through 3x3 block
        x = self.conv3x3(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # forward through second 1x1 block
        x = self.conv1x1_2(x)
        x = self.bn3(x)
        x+=inp
        out = self.relu3(x)
        return out


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.net = nn.Sequential(
            conv2d_block(in_channels=1, out_channels=10, kernel_size=3, padding=(1, 1)),
            conv2d_block(in_channels=10, out_channels=20, kernel_size=3, padding=(1, 1), bn=True),
            BottleNeck(20, 10),
            BottleNeck(20, 10),
            BottleNeck(20, 10),
            conv2d_block(in_channels=20, out_channels=40, kernel_size=3, padding=(1, 1), bn=True),
            conv2d_block(in_channels=40, out_channels=80, kernel_size=3, padding=(1, 1), bn=True),
            BottleNeck(80, 20),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(80 * 7 * 6, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 27),
            nn.ReLU(),
            nn.Softmax(dim=0)
        )

    def forward(self,x):
        x = self.net(x)
        return x