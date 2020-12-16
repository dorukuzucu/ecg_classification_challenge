
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from src.data.data_loader import ECGParquetDataloader
from src.model.utils import dict_to_torch
import numpy as np
import pandas as pd

"""
Method that will be used to load weights as a penalty matrix 
"""
def load_penalty_csv(path):
    w = pd.read_csv(path)
    np_w = w.to_numpy()
    labels = np_w[:,0]
    weights = np_w[:,1:]
    return 1/torch.from_numpy(weights)#, torch.from_numpy(labels)


class SoftDiceLoss(nn.Module):
    """
    We will try to use dice loss, which is mostly used in semantic segmentation
    """
    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predicted, target):
        """
        :param predicted: predictions as NxC tensor.
            N: Number of inputs or batches
            C: Number of classes
        :param target: labels as NxC tensor.
            N: Number of inputs or batches
            C: Number of classes
        :return: Dice Loss:
            numerator is intersection of predicted and target
            denominator is sum of predicted and target
            loss = 1 - 2 * numerator / denominator
        """
        # check shapes
        assert predicted.size() == target.size()

        # calculate numerator and denominator of dice loss
        numerator = (predicted * target).sum(1) + self.epsilon
        denominator = predicted.sum(1) + target.sum(1) + self.epsilon

        # calculate and return loss
        loss = 1 - 2 * numerator / denominator
        return loss.mean()


class L1LossWithPenalty(nn.L1Loss):
    """
    L1LossWithPenalty class takes a matrix path as a loss and calculates a loss based on penalty weights.
    Penalty weights are decided by physionet. We will utilize these weights in our loss function
    Since model can predict more than 1 classes, these cases will be taken into consideration.
    When there is only 1 predicted class and 1 label:
        penalty matrix's corresponding value will be taken into consideration
    Otherwise:
        maximum value of matrix will be used as penalty
    Penalty value will be multiplied by L1 Distance of each input's prediction and labels
    """
    def __init__(self,weight_path):
        super().__init__()
        self.penalty_weights = load_penalty_csv(weight_path)

    def forward(self, predicted, target):
        """
        :param predicted: predictions as NxC tensor.
            N: Number of inputs or batches
            C: Number of classes
        :param target: labels as NxC tensor.
            N: Number of inputs or batches
            C: Number of classes
        :return: L1 loss penalized depending on class predictions
        """
        # check shapes
        assert predicted.size() == target.size()
        # calculate loss
        loss = 0
        for batch in range(predicted.size(0)):
            raw_l1_loss = torch.abs(predicted[batch] - target[batch]).sum()
            if(predicted[batch].sum()==1 and target[batch].sum == 1):
                pred_cls = (predicted[batch] == 1).nonzero().item()
                label_cls = (target[batch] == 1).nonzero().item()
                loss+=raw_l1_loss * self.penalty_weights[pred_cls,label_cls]
            else:
                loss+=raw_l1_loss * self.penalty_weights.max()
        return loss

class MSELossWithPenalty(nn.MSELoss):
    """
        MSELossWithPenalty class takes a matrix path as a loss and calculates a loss based on penalty weights.
        Penalty weights are decided by physionet. We will utilize these weights in our loss function
        Since model can predict more than 1 classes, these cases will be taken into consideration.
        When there is only 1 predicted class and 1 label:
            penalty matrix's corresponding value will be taken into consideration
        Otherwise:
            maximum value of matrix will be used as penalty
        Penalty value will be multiplied by MSE between each input's prediction and labels
        """
    def __init__(self,weight_path):
        super().__init__()
        self.penalty_weights = load_penalty_csv(weight_path)

    def forward(self, predicted, target):
        """
        :param predicted: predictions as NxC tensor.
            N: Number of inputs or batches
            C: Number of classes
        :param target: labels as NxC tensor.
            N: Number of inputs or batches
            C: Number of classes
        :return: MSE Loss penalized depending on class predictions
        """
        # check shapes
        assert predicted.size() == target.size()
        # calculate loss
        loss = 0
        for batch in range(predicted.size(0)):
            raw_mse_loss = torch.square(torch.abs(predicted[batch] - target[batch])).mean()
            if(predicted[batch].sum()==1 and target[batch].sum == 1):
                pred_cls = (predicted[batch] == 1).nonzero().item()
                label_cls = (target[batch] == 1).nonzero().item()
                loss+=raw_mse_loss * self.penalty_weights[pred_cls,label_cls]
            else:
                loss+=raw_mse_loss * self.penalty_weights.max()
        return loss