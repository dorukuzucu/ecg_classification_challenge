# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Q9HbojKopQ-d7YZ1-bOmkR5CnvcnpJ95
"""

# Commented out IPython magic to ensure Python compatibility.
# change user and pwd with your username and password in github
#!git clone https://user:pwd@github.com/dorukuzucu/deep_learning_interim_project.git

# colab initialize pwd to /content
# we need to change directory to our project to acces the data
# %cd deep_learning_interim_project/

#download data
#! wget https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_CPSC.tar.gz

#!pip install -r requirements.txt
# after venv is done you need to restart the runtime

# Commented out IPython magic to ensure Python compatibility.
# %cd deep_learning_interim_project
import os
from pathlib import Path

from src.model import train

#config = train.DUMMY_PARAMS
config = {
    'learning_rate': [0.05], # a float
    'batch_size': [100], # an integer
    'epochs': [50], # an integer
    'optimizer_type': ["Adam"], # ["Adam", "SGD"]
    'loss_fn': ["ce_loss"], # ["ce_loss", "penalty_l1", "penalty_mse"]
    'epochs_for_val': [5], # an integer
    'weight_decay': [1e-2], # a float
    'momentum': [0], # a float
    'device':["cpu"]
}
model = train.ECGNet() # ArrhythmiaNet, ECGHeartbeat, ECGNet, Model_2, Model_Ann

mngr = train.TrainManager(model=model, processed_data_path=train.DATA_PATH, training_config=config, run_name="ECGNet_ce_loss")
mngr.train()

