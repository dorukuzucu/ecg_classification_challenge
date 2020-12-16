"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from src.data.data_loader import ECGParquetDataloader
from src.model.utils import dict_to_torch
"""
import numpy as np
import pandas as pd

def load_penalty_csv():
    w = pd.read_csv("weights.csv")
    np_w = w.to_numpy()
    labels = np_w[:,0]
    weights = np_w[:,1:]
    return torch.from_numpy(weights)#, torch.from_numpy(labels)

class SmoothDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predicted, target):
        batches = target.size(0)
        flat_predicted = predicted.view(batches, -1)
        flat_target = target.view(batches, -1)

        numerator = (flat_predicted * flat_target).sum(1) + self.epsilon
        denominator = flat_predicted.sum(1) + flat_target.sum(1) + self.epsilon

        loss = 1 - 2 * numerator / denominator / batches
        return loss.sum()


class DiceLossWithPenalty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):
        num_classes = target.size(-1)
        loss = 0
        smooth_dice_loss = SmoothDiceLoss()
        for class_no in range(num_classes):
            loss += smooth_dice_loss(predicted[:, class_no], target[:, class_no])
        return loss

class L1LossWithPenalty(nn.L1Loss):
    """
    This method takes 2 parameters and calculates a loss based on penalty weights.
    Penalty weights are decided by physionet.
    We will utilize these weights in our loss function
    """
    def __init__(self):
        self.penalty_weights = load_penalty_csv()

    def forward(self,predicted,labels):
        pass

class MSELossWithPenalty(nn.MSELoss):
    pass

"""
import pandas as pd

w = pd.read_csv("weights.csv")
np_w = w.to_numpy()

print(np_w[:, 1:].shape)
"""
class L1Loss(nn.Module)

b_pred = torch.from_numpy(np.array([[1, 0, 0, 0, 1], [1, 0, 0, 0, 1]])).float()
b_lab = torch.from_numpy(np.array([[1, 0, 0, 0, 1], [0, 0, 0, 1, 0]])).float()
pred = torch.from_numpy(np.array([0, 0, 0, 1, 0])).view(1, -1).float()
lab = torch.from_numpy(np.array([0, 0, 0, 1, 0])).view(1, -1).float()

sdl = SmoothDiceLoss()
dll = DiceLossWithLogits()

print("single loss=", sdl.forward(b_pred, b_lab))
print("multiclass loss=", dll.forward(b_pred, b_lab))

loss = torch.nn.L1Loss()
loss2 = torch.nn.MSELoss()
print("L1_Loss", loss(b_pred, b_lab))
print("MSE", loss(b_pred, b_lab))
"""
