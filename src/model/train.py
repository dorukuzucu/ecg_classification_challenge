import torch
from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader

from src.model.utils import create_dict_combination
from src.data.data_loader import ECGParquetDataloader
import pandas as pd
import numpy as np

class TrainManager:
    def __init__(self,model,dataloader: ECGParquetDataloader, training_config):
        self.model = model
        self.dataloader = dataloader
        self.is_params_set = False
        self.config = training_config
        self.__generate_combination_of_params()

    def train(self):
        for run in self.runs:
            for epoch in range(run.epochs):
                new_dataset_loader = self.dataloader.new_loader(num_epochs=1, batch_size=run.batch_size)
                for idx, batch_data in enumerate(new_dataset_loader):


                    pass


    def __generate_combination_of_params(self):
        self.runs = create_dict_combination(self.config)





with DataLoader(make_batch_reader(
        r'file:C:\Users\ABRA\Desktop\Ders\Yüksek Lisans\BLG561-Deep Learning\deep_learning_interim_project\data\processed',
        num_epochs=1
        #,transform_spec=trn
        ),
                batch_size=3) as train_loader:


    print()
    count = 0
    for idx,row in enumerate(train_loader):
        print(row)
        count+=1


    print(count)
