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
    return nn.Conv1d(kernel_size=3,in_channels=in_channels, out_channels=out_channels, stride=1,padding=(1,1))

def conv1x1(in_channels,out_channels):
    return nn.Conv1d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1)

def conv2d_block(in_channels,out_channels,kernel_size=3,stride=1,padding=(1,1)):
    layers = [nn.Conv1d(kernel_size=kernel_size, in_channels=in_channels,out_channels=out_channels, stride=stride,padding=padding),
              nn.BatchNorm1d(num_features=out_channels),
              nn.ReLU()
              ]
    return nn.Sequential(*layers)

class BottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, width):
        # first 1x1 block
        super().__init__()
        self.conv1x1_1 = conv1x1(in_channels=in_channels,out_channels=width)
        self.bn1 = nn.BatchNorm2d(num_features=width)
        self.relu1 = nn.ReLU()

        # 3x3 block
        self.conv3x3 = conv3x3(width,width)
        self.bn2 = nn.BatchNorm2d(num_features=width)
        self.relu2 = nn.ReLU()

        # second 1x1 block
        self.conv1x1_2 = conv1x1(in_channels=width, out_channels=out_channels)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
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
        self.fc1 = nn.Linear(in_features=14, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=100)
        self.fc4 = nn.Linear(in_features=100, out_features=100)
        self.fc5 = nn.Linear(in_features=100, out_features=55)
        self.out = nn.Linear(in_features=55, out_features=27)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.s_max = nn.Softmax()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.out(x)
        x = self.s_max(x)
        return x