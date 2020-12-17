from itertools import product
from collections import OrderedDict
from collections import namedtuple
import numpy as np
import torch


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
    features = [value.tolist() for value in list(dict_inp.values())[:feature_count]]
    labels = [value.tolist() for value in list(dict_inp.values())[feature_count:]]
    return torch.from_numpy(np.array(features).T).float(), torch.from_numpy(np.array(labels).T).float()


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
        if torch.equal(predictions[batch], targets[batch]):
            correct += 1
    return correct
