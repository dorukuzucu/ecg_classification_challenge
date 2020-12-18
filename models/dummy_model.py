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
    layers = [nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,out_channels=out_channels, stride=stride,padding=padding),
              nn.ReLU()
              ]
    if bn:
        layers.append(nn.BatchNorm2d(num_features=out_channels))

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


class OutLayer(nn.Module):
    """
    This class is used to construct output layers. It takes 5 parameters:
        input_features: number of features in incoming tensor
        output_features: required number of classes on output
        bn_flag: Flag to set batch normalization layer
        dropout_flag: Flag to set dropout layer
        dropout_value: possibility for a weight to be zeroed
    """
    def __init__(self,input_features,output_features=27,bn_flag=True,dropout_flag=True,dropout_value=0.15):
        super(OutLayer, self).__init__()
        layers = []
        # add required number of layers
        while (input_features//4)>output_features:
            layers.append(nn.Linear(input_features,input_features//4))
            layers.append(nn.ReLU())
            if bn_flag:
                layers.append(nn.BatchNorm1d(input_features//4))
            if dropout_flag:
                layers.append(nn.Dropout(p=dropout_value))
            input_features = input_features//4

        # add final layers
        layers.append(nn.Linear(input_features,output_features))
        layers.append(nn.Softmax(dim=0))
        # create network
        self.net = nn.Sequential(*layers)

    def forward(self,x):
        out = self.net(x)
        return out


class ECGNetMini(nn.Module):
    def __init__(self):
        super(ECGNetMini, self).__init__()
        self.conv_net = nn.Sequential(
            conv2d_block(in_channels=1, out_channels=10, kernel_size=3, padding=(1, 1)),
            conv2d_block(in_channels=10, out_channels=20, kernel_size=3, padding=(1, 1), bn=True),
            BottleNeck(20, 10),
            BottleNeck(20, 10),
            BottleNeck(20, 10),
            conv2d_block(in_channels=20, out_channels=40, kernel_size=3, padding=(1, 1), bn=True),
            conv2d_block(in_channels=40, out_channels=80, kernel_size=3, padding=(1, 1), bn=True),
            BottleNeck(80, 20),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten()
        )
        self.out_net = OutLayer(80*14*12)

    def forward(self,x):
        x = self.conv_net(x)
        out = self.out_net(x)
        return out


class ResECGNet(nn.Module):
    def __init__(self,num_bottle_neck):
        super().__init__()

        # create layers for convolutional part
        conv_layers = []
        conv_layers.append(conv2d_block(in_channels=1, out_channels=10, kernel_size=3, padding=(1, 1)))
        conv_layers.append(conv2d_block(in_channels=10, out_channels=64, kernel_size=3, padding=(1, 1), bn=True))
        bottle_necks = [BottleNeck(64, 24) for _ in range(num_bottle_neck)]
        conv_layers = conv_layers+bottle_necks
        # first part of our net is conv net
        self.conv_net = nn.Sequential(*conv_layers)
        # we need to flatten convolutional output
        self.flat = nn.Flatten()
        # output layer
        self.out_layers = OutLayer(64*14*12)

    def forward(self,x):
        x = self.conv_net(x)
        x = self.flat(x)
        out = self.out_layers(x)
        return out