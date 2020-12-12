import torch
import torch.nn.functional as F
import torch.nn as nn


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.fc1 = nn.Linear(in_features=14, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=100)
        self.fc4 = nn.Linear(in_features=100, out_features=100)
        self.fc5 = nn.Linear(in_features=100, out_features=55)
        self.out = nn.Linear(in_features=55, out_features=9)
        self.loss_weights = torch.full((1,9),1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.out(x)
        return x