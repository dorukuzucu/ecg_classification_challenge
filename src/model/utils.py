from itertools import product
from collections import OrderedDict
from collections import namedtuple


def create_dict_combination(dict):
    # get combinations
    ordered_dict = OrderedDict(dict)
    combinations = product(*ordered_dict.values())
    # create named tuple to save combinations
    combination_tuple = namedtuple('run',ordered_dict.keys())
    # create array to save namedtupled and iterate over them
    run_configs = []

    # add combinations as configs
    for combination in combinations:
        run_configs.append(combination_tuple(*combination))

    # return value
    return run_configs