import os
from collections import OrderedDict, namedtuple
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


def create_dict_combination(dict):
    # get combinations
    ordered_dict = OrderedDict(dict)
    combinations = product(*ordered_dict.values())
    # create named tuple to save combinations
    combination_tuple = namedtuple('run', ordered_dict.keys())
    # create array to save namedtupled and iterate over them
    run_configs = []

    # add combinations as configs
    for combination in combinations:
        run_configs.append(combination_tuple(*combination))

    # return value
    return run_configs


def dict_to_torch(dict_inp, feature_count):
    features = [value.tolist() for value in list(dict_inp.values())[1:]]
    labels = [value.tolist() for value in list(dict_inp.values())[0]]
    features = np.array(features).T # to set dimensions from feature_size x batch_size to its transpose
    labels = np.array(labels).T # to set dimensions from feature_size x batch_size to its transpose

    cnn_features = features.reshape((features.shape[0], 1, feature_count, (features.shape[1] // feature_count)))
    return torch.from_numpy(cnn_features).float(), torch.from_numpy(labels).long()

def correct_predictions(predictions, targets):
    """
    :param predictions: input of predicted values. size should be WxC
    :param targets: input of target values. size should be W
    :return: total number pf correct predictions
    """
    # calculate correct predictions over each batch
    correct = 0
    for batch in range(predictions.size(0)):
        if torch.argmax(predictions[batch]).item() == targets[batch].item():
            correct += 1
    return correct


def plot_metrics(result_path_1, result_path_2):
    with open(result_path_1, "r") as file:
        lines = file.readlines()
        train_1 = pd.DataFrame(eval(lines[1]))
        val_1 = pd.DataFrame(eval(lines[2]))
    
    with open(result_path_2, "r") as file:
        lines = file.readlines()
        train_2 = pd.DataFrame(eval(lines[1]))
        val_2 = pd.DataFrame(eval(lines[2]))

    val_losses_1, val_acc_1 = [], []
    val_losses_2, val_acc_2 = [], []
    val_interval = int(len(train_1) / len(val_1))
    for i in range(len(val_1)):
        val_losses_1 += [val_1.loc[i, "loss"]] * int(val_interval)
        val_acc_1 += [val_1.loc[i, "acc"]] * int(val_interval)
        val_losses_2 += [val_2.loc[i, "loss"]] * int(val_interval)
        val_acc_2 += [val_2.loc[i, "acc"]] * int(val_interval)

    # create output folder
    result_1_name = os.path.splitext(result_path_1)[0]
    result_2_name = os.path.splitext(result_path_2)[0]
    Path(result_1_name).mkdir(exist_ok=True)
    # plot train metrics
    # loss
    plt.plot(train_1["loss"], label=result_1_name)
    plt.plot(train_2["loss"], label=result_2_name)
    plt.title('train loss curve')
    plt.savefig(f"{result_1_name}/train_loss.png")
    plt.close()
    # accuracy
    plt.plot(train_1["acc"], label=result_1_name)
    plt.plot(train_2["acc"], label=result_2_name)
    plt.title('train accuracy curve')
    plt.savefig(f"{result_1_name}/train_accuracy.png")
    plt.close()

    # plot validation metrics
    # loss
    plt.plot(val_losses_1, label=result_1_name)
    plt.plot(val_losses_2, label=result_2_name)
    plt.title('validation loss curve')
    plt.savefig(f"{result_1_name}/val_loss.png")
    plt.close()
    # accuracy
    plt.plot(val_acc_1, label=result_1_name)
    plt.plot(val_acc_2, label=result_2_name)
    plt.title('validation accuracy curve')
    plt.savefig(f"{result_1_name}/val_accuracy.png")
    plt.close()


result_path_1 = os.path.join("results", "ecg_net_results", "Ecg_heratbeat_2_cont_0_results.txt")
result_path_2 = os.path.join("results", "ecg_net_results", "ECGNet_mse_0_results.txt")
plot_metrics(result_path_1, result_path_2)
