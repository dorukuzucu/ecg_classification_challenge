import torch
import torch.nn as nn


"""
Following methods are created for ease of use.
    conv3x3: 2D convolutional layer with kernel_size=3x3, stride=1, padding=1
    conv1x1: 2D convolutional layer with kernel_size=1x1, stride=1
    conv2d_block: combination of following:
        Conv2d
        ReLU
        BatchNorm(optional. set via param 'bn'
"""
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(kernel_size=3,in_channels=in_channels, out_channels=out_channels, stride=1,padding=(1,1))

def conv1x1(in_channels,out_channels):
    return nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1)

def conv2d_block(in_channels,out_channels,kernel_size=3,stride=1,padding=(1,1),bn=False):
    layers = []
    layers.append(nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,out_channels=out_channels, stride=stride,padding=padding))
    if bn:
        layers.append(nn.BatchNorm2d(num_features=out_channels))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class BottleNeck(nn.Module):
    """
    Bottleneck is essentially, a 2 times 3x3 conv2d blocks replaced by 1x1,3x3,1x1 blocks
    """
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

class ECGNet(nn.Module):
    def __init__(self):
        super(ECGNet, self).__init__()
        self.conv_net = nn.Sequential(
            conv2d_block(in_channels=1, out_channels=10, kernel_size=3, padding=(1, 1)),
            conv2d_block(in_channels=10, out_channels=20, kernel_size=3, padding=(1, 1), bn=True),
            BottleNeck(20, 10),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten()
        )
        self.out_layers = nn.Sequential(
            nn.Linear(in_features=20*7*6, out_features=128, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=27, bias=True),
            nn.ReLU(),
            nn.Softmax()
        )


    def forward(self,x):
        x = self.conv_net(x)
        out = self.out_layers(x)
        return out


class Model_1(nn.Module):
    def __init__(self,num_bottle_neck):
        super().__init__()

        # create layers for convolutional part
        conv_layers = []
        conv_layers.append(conv2d_block(in_channels=1, out_channels=5, kernel_size=3, padding=(1, 1)))
        conv_layers.append(conv2d_block(in_channels=5, out_channels=24, kernel_size=3, padding=(1, 1), bn=True))
        bottle_necks = [BottleNeck(24,12) for _ in range(num_bottle_neck)]
        conv_layers = conv_layers+bottle_necks
        # first part of our net is conv net
        self.conv_net = nn.Sequential(*conv_layers)
        # we need to flatten convolutional output
        self.flat = nn.Flatten()
        # output layer
        self.out_layers = nn.Sequential(
            nn.Linear(in_features=4032, out_features=800, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(800),
            nn.Linear(in_features=800, out_features=164, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=164, out_features=27, bias=True)
        )

    def forward(self,x):
        x = self.conv_net(x)
        x = self.flat(x)
        out = self.out_layers(x)
        return out

class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.conv_1 = conv2d_block(in_channels=1, out_channels=3, kernel_size=3, padding=(1, 1),bn=True)
        self.conv_2 = conv2d_block(in_channels=3, out_channels=8, kernel_size=3, padding=(1, 1),bn=True)
        self.drop_out = nn.Dropout(p=0.1)
        self.b_neck = BottleNeck(8,4)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(8*14*12,124)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(124,27)
        self.softmax = nn.Softmax()

    def forward(self,x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.drop_out(x)
        x = self.b_neck(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.relu_1(x)
        x = self.fc_2(x)
        out = self.softmax(x)
        return out
