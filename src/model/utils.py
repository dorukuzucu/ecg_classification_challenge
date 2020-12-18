from itertools import product
from collections import OrderedDict
from collections import namedtuple
import numpy as np
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
    labels_one_hot = nn.functional.one_hot(torch.from_numpy(labels).to(torch.int64),27)
    return torch.from_numpy(cnn_features).float(), labels_one_hot.float()


def correct_predictions(predictions, targets):
    """
    :param predictions: input of predicted values. size should be WxC
    :param targets: input of target values. size should be WxC
    :return: total number pf correct predictions
    """
    # shapes should be the same
    assert predictions.size() == targets.size()
    # calculate correct predictions over each batch
    correct = 0
    for batch in range(predictions.size(0)):
        if torch.equal(torch.argmax(predictions[batch]), torch.argmax(targets[batch])):
            correct += 1
    return correct