import pandas as pd
import torch
from torch import nn

"""
Method that will be used to load weights as a penalty matrix 
"""
def load_penalty_csv(path):
    """
    This matrix is normally used to score TP, FP, TN, FN of each classes.
    Since we have this conditions, we will add these as punishment terms to losses
    """
    w = pd.read_csv(path)
    np_w = w.to_numpy()
    labels = np_w[:, 0]
    weights = np_w[:, 1:]
    penalty =  1 / torch.from_numpy(weights)
    return penalty / penalty.max()


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
        if len(predicted.size())>1:
            numerator = torch.add((predicted * target).sum(1),1e-12)
            denominator = torch.add((predicted.sum(1) + target.sum(1)), 1e-12)
        else:
            numerator = torch.add((predicted * target),1e-12)
            denominator = torch.add((predicted + target), 1e-12)
        # calculate and return loss
        loss = 1 - 2 * numerator / denominator
        return loss.mean()


class SoftDiceLossWithPenalty(SoftDiceLoss):
    """
    SoftDiceLossWithPenalty class takes a matrix path as a loss and calculates a loss based on penalty weights.
    Weights are decided by physionet and originally used in scoring.
    We will utilize these weights in our loss function with simple a modification
    penalty matrix's corresponding value will be taken into consideration
    Penalty value will be multiplied by Soft Dice Loss of each input's prediction and labels
    """
    def __init__(self, weight_path):
        super().__init__()
        self.penalty_weights = load_penalty_csv(weight_path)
        self.soft_dice_loss = SoftDiceLoss()

    def forward(self, predicted, target):
        """
        :param predicted: predictions as NxC tensor.
            N: Number of inputs or batches
            C: Number of classes
        :param target: labels as NxC tensor.
            N: Number of inputs or batches
            C: Number of classes
        :return: Dice Loss with a Penalty:
            loss = SoftDiceLoss * penalty_term
        """
        # check shapes
        assert predicted.size() == target.size()
        # calculate loss
        loss = 0
        for batch in range(predicted.size(0)):
            # get raw dice loss
            raw_dice_loss = self.soft_dice_loss.forward(predicted[batch], target[batch])
            # get predicted class and target classes
            pred_cls = torch.argmax(predicted[batch]).item()
            label_cls = torch.argmax(target[batch]).item()
            # add penalty
            loss += raw_dice_loss * self.penalty_weights[pred_cls, label_cls]
        # normalize with batch size
        return loss/predicted.size(0)


class L1LossWithPenalty(nn.Module):
    """
    L1LossWithPenalty class takes a matrix path as a loss and calculates a loss based on penalty weights.
    Weights are decided by physionet and originally used in scoring.
    We will utilize these weights in our loss function with simple a modification
    penalty matrix's corresponding value will be taken into consideration
    Penalty value will be multiplied by L1 distance of each input's prediction and labels
    """
    def __init__(self, weight_path):
        super().__init__()
        self.penalty_weights = load_penalty_csv(weight_path)

    def forward(self, predicted, target):
        """
        :param predicted: predictions as NxC tensor.
            N: Number of inputs or batches
            C: Number of classes
        :param target: labels as N sized tensor.
            N: Number of inputs or batches
        :return: L1 loss penalized depending on class predictions
        """
        # check shapes
        assert predicted.size() == target.size()
        # calculate loss
        loss = 0
        for batch in range(predicted.size(0)):
            # calculate raw loss
            raw_l1_loss = torch.abs(predicted[batch] - target[batch]).sum()
            # get class numbers
            pred_cls = torch.argmax(predicted[batch]).item()
            label_cls = torch.argmax(target[batch]).item()
            # add penalty for specified loss
            loss += raw_l1_loss * self.penalty_weights[pred_cls, label_cls]

        return loss/predicted.size(0)


class MSELossWithPenalty(nn.Module):
    """
     MSELossWithPenalty class takes a matrix path as a loss and calculates a loss based on penalty weights.
     Weights are decided by physionet and originally used in scoring.
     We will utilize these weights in our loss function with simple a modification
     penalty matrix's corresponding value will be taken into consideration
     Penalty value will be multiplied by L1 distance of each input's prediction and labels
     """

    def __init__(self, weight_path):
        super().__init__()
        self.penalty_weights = load_penalty_csv(weight_path)

    def forward(self, predicted, target):
        """
        :param predicted: predictions as NxC tensor.
            N: Number of inputs or batches
            C: Number of classes
        :param target: labels as N tensor.
            N: Number of inputs or batches
        :return: MSE Loss penalized depending on class predictions
        """
        # calculate loss
        loss = 0
        target_one_hot = nn.functional.one_hot(target.to(torch.int64), 27)
        for batch in range(predicted.size(0)):
            # calculate raw loss
            raw_mse_loss = torch.square(torch.abs(predicted[batch] - target_one_hot[batch])).mean()

            # get predicted and target class numbers
            pred_cls = torch.argmax(predicted[batch]).item()
            label_cls = torch.argmax(target_one_hot[batch]).item()

            # add penalty for that loss
            loss += raw_mse_loss * self.penalty_weights[pred_cls, label_cls]
        # normalize loss and return
        return loss/predicted.size(0)