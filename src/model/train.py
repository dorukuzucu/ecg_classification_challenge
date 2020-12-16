from collections import OrderedDict
import torch
from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader
from torch.autograd import Variable
from src.model.utils import create_dict_combination, dict_to_torch
from src.data.losses import *
from models.dummy_model import TestNet
from src.data.data_loader import ECGParquetDataloader
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

# TODO set a method for epoch train
# TODO begin_run, begin_epoch methods
# TODO save best model
# TODO calculate&save metrics
class TrainManager:
    def __init__(self, model, processed_data_path, training_config):
        self.model = model
        self.val_loader = ECGParquetDataloader(processed_data_path + "/val")
        self.train_loader = ECGParquetDataloader(processed_data_path + "/train")
        self.runs = create_dict_combination(training_config)
        self.dataset = None
        self.optimizer = None
        self.criterion = None

    def train(self):
        for run in self.runs:
            # set optimizer and loss function for this run
            self.set_optimizer(learning_rate=run.learning_rate, optim_type=run.optimizer, weight_decay=run.weight_decay,momentum=run.momentum)
            self.set_criterion(run.loss_fn)

            # iterate over epochs
            for epoch in range(run.epochs):
                # get dataset for this epoch to be trained on
                self.dataset = self.train_loader.new_loader(num_epochs=1, batch_size=run.batch_size)
                self.set_criterion(run.loss_fn)

                # iterate over batches
                epoch_loss = 0
                for batch_data in self.dataset:
                    # we need to proprocess parquet data it order to feed it to network
                    features, labels = dict_to_torch(batch_data, batch_size=run.batch_size, feature_count=14)
                    predictions = self.model(features.float())

                    # calculate loss and update weights for batch
                    loss = self.criterion(predictions, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss
                print("Epoch:{} Loss:{}".format(epoch, epoch_loss))
                """
                Validation epoch, every x epoch decided by user, we will validate our model
                """
                if epoch % run.epochs_for_val == 0:
                    print("Validating model")
                    self.dataset = self.val_loader.new_loader(num_epochs=1, batch_size=run.batch_size)
                    val_loss = 0
                    for batch_data in self.dataset:
                        features, labels = dict_to_torch(batch_data, batch_size=run.batch_size, feature_count=14)
                        predictions = self.model(features.float())
                        criterion = nn.BCEWithLogitsLoss(pos_weight=self.model.loss_weights)
                        loss = criterion(predictions, labels.float())
                        val_loss += loss
                    print("Validation Epoch:{} Loss:{}".format(epoch, val_loss))

    def set_optimizer(self, optim_type, learning_rate, weight_decay, momentum):
        if optim_type == "Adam":
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        elif optim_type == "SGD":
            self.optimizer = optim.SGD(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            print("Setting for default optimizer")
            self.optimizer = optim.Adam(params=self.model.parameters(),lr=learning_rate)

    def set_criterion(self,loss):
        if loss == "dice":
            pass
        elif loss == "PenaltyDice":
            pass
        elif loss == "PenaltyBCE":
            pass
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.model.loss_weights)



dummy_model = TestNet()
data_path = r'file:C:\Users\ABRA\Desktop\Ders\Yüksek Lisans\BLG561-Deep Learning\deep_learning_interim_project\data\processed'

dummy_params = {
    'learning_rate': [0.01],
    'batch_size': [3],
    'epochs': [100],
    'num_workers': [0],
    'optimizer': ["Adam"],
    'loss_fn': ["Test"],
    'epochs_for_val': [4],
    'weight_decay': [1e-4],
    'momentum': [0]
}

mngr = TrainManager(model=dummy_model, processed_data_path=loader, training_config=dummy_params)
mngr.train()
