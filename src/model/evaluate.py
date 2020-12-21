import time
from pathlib import Path
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch import nn

from models.ecg_net_models import ECGHeartbeat, ECGNet, ArrhythmiaNet, Model_Ann
from src.model.evaluate_12ECG_score import *
from src.data.data_loader import *
import pandas as pd
import numpy as np

from src.model.utils import dict_to_torch

SAVE_PATH = os.path.join(Path(__file__).parents[2],"results","ecg_net_results")+os.path.sep
WEIGHT_PATH = os.path.join(Path(__file__).parents[2],"data","raw","weights.csv")
TEST_DATASET_PATH = "file:"+os.path.join(Path(__file__).parents[2],"data","processed","test")
TEST_BATCH_SIZE = 150


"""
method to be used to load weights file provided by Physionet
"""
def load_weight_csv(weight_path):
    """
    :param weight_path: weights csv path to be loaded
    :return: weight as numpy array
    """
    weights = pd.read_csv(weight_path)
    return weights.to_numpy()[:, 1:]

def save_confusion_matrix(confusion_matrix, title, file_path):
    """
    :param confusion_matrix:
    :param title: Title of plot
    :param file_path: file path to save image
    :return:
    """

    num_classes = confusion_matrix.shape[0]
    fig,ax = plt.subplots()
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    ax.set_title(title)
    for i in range(num_classes):
        for j in range(num_classes):
            c = confusion_matrix[i][j]
            ax.text(j, i, "{}".format(c), va='center', ha='center')
            time.sleep(0.01)
    save_name = file_path+title+".png"
    fig.set_size_inches(15,15)
    fig.savefig(save_name)


def evaluate_model(model,test_data_path, batch_size):
    """
    :param model: PRETRAINED model to be evaluated
    :param test_data_path: test data set path
    :param weight_path: weights to be loaded to evaluate score
    :param batch_size: batch size for data set
    :return: calculated classes and target classes as numpy arrays
    """
    test_loader = ECGParquetDataloader(test_data_path)
    dataset = test_loader.new_loader(num_epochs=1,batch_size=batch_size)

    all_predictions = torch.zeros(1,27)
    all_labels = torch.zeros(1)

    for batch in dataset:
        # we need to preprocess parquet data it order to feed it to network
        features, labels = dict_to_torch(batch, feature_count=14)
        predictions = model(features)

        all_predictions = torch.cat((all_predictions, predictions))
        all_labels = torch.cat((all_labels, labels))

    return all_predictions[1:].detach().numpy(), all_labels[1:].detach().numpy()

def calculate_confusion_matrix(targets, preds):
    """
    :param targets: target values as a numpy array with size (number_of_recordings, number_of_classes)
    :param preds: predicted values as a numpy array with size (number_of_recordings, number_of_classes)
    :return: confusion matrix
    """
    # get number of recordings and classes
    num_recordings, num_classes = np.shape(targets)
    confusion_matrix = np.zeros((num_classes, num_classes))

    # calculate class indexes
    target = targets.argmax(1)
    pred = preds.argmax(1)
    # calculate confusion matrix
    for record in range(num_recordings):
        normalization = np.sum(targets[record, :] == preds[record, :])
        confusion_matrix[target[record], pred[record]]+= 1
    return confusion_matrix.astype(int)

def calculate_score(target, preds):
    weights = load_weight_csv(WEIGHT_PATH)
    preds_one_hot = nn.functional.one_hot(torch.from_numpy(preds.argmax(1)).to(torch.int64), 27)
    target_one_hot = nn.functional.one_hot(torch.from_numpy(target).to(torch.int64), 27)

    conf_mat = calculate_confusion_matrix(targets=target_one_hot.numpy(), preds=preds_one_hot.numpy())
    score = conf_mat * weights
    return conf_mat,np.sum(score)


def load_model(model_name,state_dict_path):
    state_dict = torch.load(state_dict_path)
    if model_name=="ecg_net":
        model = ECGNet()
    elif model_name == "arrhythmia_net":
        model = ArrhythmiaNet()
    elif model_name == "ecg_heartbeat_net":
        model = ECGHeartbeat()
    elif model_name == "ann_net":
        model = Model_Ann()
    else:
        raise ValueError("Please choose a valid model: ecg_net, arrhythmia_net, ecg_heartbeat_net")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def calculate_results(model_name,model_path,result_name):
    mdl = load_model(model_name, state_dict_path=model_path)
    preds, target = evaluate_model(model=mdl, test_data_path=TEST_DATASET_PATH, batch_size=TEST_BATCH_SIZE)
    confusion_mat, score = calculate_score(target=target, preds=preds)
    save_confusion_matrix(confusion_mat, model_name+"_"+result_name+"_confusion_matrix", SAVE_PATH)
    with open(SAVE_PATH + model_name+"_"+result_name+"_score.txt", "w") as file:
        file.write(str(score) + "\n")
        file.write(str(confusion_mat)+"\n")
        file.close()
    print("Results are save to:{}".format(SAVE_PATH))

"""
ecg_ce_path = os.path.join(Path(__file__).parents[2],"results","ecg_net_results")+os.path.sep+"ECGNet_ce_loss_0_model"
ecg_mse_path = os.path.join(Path(__file__).parents[2],"results","ecg_net_results")+os.path.sep+"ECGNet_mse_0_model"
ecg_heartbeat = os.path.join(Path(__file__).parents[2],"results","ecg_net_results")+os.path.sep+"Ecg_heratbeat_2_cont_0_model"

calculate_results(model_name="ecg_net",model_path=ecg_ce_path,result_name="mse_loss")
calculate_results(model_name="ecg_net",model_path=ecg_mse_path,result_name="ce_loss")
calculate_results(model_name="ecg_heartbeat_net",model_path=ecg_heartbeat,result_name="mse_loss")
"""