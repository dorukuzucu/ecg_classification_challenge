from src.model.utils import create_dict_combination, dict_to_torch, correct_predictions
from models.losses import *
from models.dummy_model import TestNet
from src.data.data_loader import ECGParquetDataloader
import torch.nn as nn
import torch.optim as optim
import os

# TODO set a method for epoch train
# TODO begin_run, begin_epoch methods
# TODO save best model
# TODO calculate&save metrics
class TrainManager:
    def __init__(self, model, processed_data_path, training_config):
        self.model = model
        self.val_loader = ECGParquetDataloader(os.path.join(processed_data_path, "validation"))
        self.train_loader = ECGParquetDataloader(os.path.join(processed_data_path, "train"))
        self.runs = create_dict_combination(training_config)
        self.dataset = None
        self.optimizer = None
        self.criterion = None

    def train(self):
        for run in self.runs:
            # set optimizer and loss function for this run
            self.begin_run(run=run)

            # iterate over epochs
            for epoch in range(run.epochs):
                # get dataset for this epoch to be trained on
                self.dataset = self.train_loader.new_loader(num_epochs=1, batch_size=run.batch_size)

                # to calculate accuracy each epoch
                correct_prediction_count = 0
                total_predictions = 0

                # iterate over batches
                epoch_loss = 0
                for batch_data in self.dataset:
                    # we need to preprocess parquet data it order to feed it to network
                    features, labels = dict_to_torch(batch_data, feature_count=14)
                    predictions = self.model(features)

                    # calculate loss and update weights for batch
                    loss_out = self.criterion(predictions, labels)
                    self.optimizer.zero_grad()
                    loss_out.backward()
                    self.optimizer.step()
                    epoch_loss += loss_out

                    # add correct and total predictions
                    correct_prediction_count += correct_predictions(predictions, labels)
                    total_predictions += run.batch_size
                print("Epoch:{} Loss:{} Accuracy:{}".format(epoch, epoch_loss, correct_prediction_count/total_predictions*100))
                """
                Validation epoch, every x epoch decided by user, we will validate our model
                """
                if epoch % run.epochs_for_val == 0:
                    print("Validating model")
                    self.dataset = self.val_loader.new_loader(num_epochs=1, batch_size=run.batch_size)

                    # to calculate metrics each epoch
                    correct_prediction_count = 0
                    total_predictions = 0
                    val_loss = 0
                    for batch_data in self.dataset:
                        # we need to preprocess parquet data it order to feed it to network
                        features, labels = dict_to_torch(batch_data, batch_size=run.batch_size, feature_count=14)
                        predictions = self.model(features.float())
                        # calculate loss for batch
                        criterion = nn.BCEWithLogitsLoss(pos_weight=self.model.loss_weights)
                        loss_out = criterion(predictions, labels.float())
                        val_loss += loss_out
                        # add correct and total predictions
                        correct_prediction_count += correct_predictions(predictions, labels)
                        total_predictions += run.batch_size
                    print("Validation Epoch:{} Loss:{} Accuracy:{}".format(epoch, epoch_loss, correct_prediction_count/total_predictions*100))

    def set_optimizer(self, run):
        if run.optimizer_type == "Adam":
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=run.learning_rate, weight_decay=run.weight_decay)
        elif run.optimizer_type == "SGD":
            self.optimizer = optim.SGD(params=self.model.parameters(), lr=run.learning_rate, weight_decay=run.weight_decay,
                                       momentum=run.momentum)

    def set_criterion(self, loss_fn="penalty_dice"):
        if loss_fn == "dice":
            self.criterion = SoftDiceLoss()
        elif loss_fn == "penalty_l1":
            self.criterion = L1LossWithPenalty()
        elif loss_fn == "penalty_mse":
            self.criterion = MSELossWithPenalty()
        elif loss_fn == "penalty_dice":
            self.criterion = SoftDiceLossWithPenalty()

    def begin_run(self, run):
        self.set_optimizer(run)
        self.set_criterion(run.loss_fn)


dummy_model = TestNet()
data_path = r'file:C:\Users\ABRA\Desktop\Ders\Yüksek Lisans\BLG561-Deep Learning\deep_learning_interim_project\data\processed'


dummy_params = {
    'learning_rate': [0.01],
    'batch_size': [3],
    'epochs': [100],
    'num_workers': [0],
    'optimizer_type': ["Adam"],
    'loss_fn': ["penalty_l1"],
    'epochs_for_val': [4],
    'weight_decay': [1e-4],
    'momentum': [0]
}

mngr = TrainManager(model=dummy_model, processed_data_path=data_path, training_config=dummy_params)
mngr.train()
