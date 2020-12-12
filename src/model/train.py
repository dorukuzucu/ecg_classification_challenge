from collections import OrderedDict

import torch
from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader
from torch.autograd import Variable

from src.model.utils import create_dict_combination, dict_to_torch
from models.dummy_model import TestNet
from src.data.data_loader import ECGParquetDataloader
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


def zero_one_loss(preds,labels):
    pass

class TrainManager:
    def __init__(self, model, data_loader, training_config):
        self.model = model
        self.data_loader = data_loader
        self.is_params_set = False
        self.config = training_config
        self.runs = None
        self.__generate_combination_of_params()

    def train(self):
        for run in self.runs:
            self.set_optimizer(learning_rate=run.learning_rate,optim_type=run.optimizer)
            for epoch in range(run.epochs):
                new_dataset = self.data_loader.new_loader(num_epochs=1, batch_size=run.batch_size)
                epoch_loss = 0
                for batch_data in new_dataset:
                    features, labels = dict_to_torch(batch_data,batch_size=run.batch_size,feature_count=14)
                    predictions = self.model(features.float())

                    criterion = nn.BCEWithLogitsLoss(pos_weight=self.model.loss_weights)
                    loss = criterion(predictions,labels.float())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss+=loss
                print("Epoch:{} Loss:{}".format(epoch,epoch_loss))
            # TODO parse dataset and feed to network
            pass


    def __generate_combination_of_params(self):
        self.runs = create_dict_combination(self.config)


    def set_optimizer(self,optim_type,learning_rate):
        self.optimizer = optim.Adam(params=self.model.parameters(),lr=learning_rate)


dummy_model = TestNet()

data_path = r'file:C:\Users\ABRA\Desktop\Ders\Yüksek Lisans\BLG561-Deep Learning\deep_learning_interim_project\data\processed'

loader = ECGParquetDataloader(data_path)

dummy_params = {
    'learning_rate':[0.01],
    'batch_size':[3],
    'epochs':[10],
    'num_workers':[0],
    'optimizer': "Adam"
}

mngr = TrainManager(model=dummy_model, data_loader=loader, training_config=dummy_params)
mngr.train()
