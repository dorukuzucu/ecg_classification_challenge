from src.model.utils import create_dict_combination, dict_to_torch, correct_predictions
from models.losses import *
from models.dummy_model import ECGNet, Model_1, Model_2
from src.data.data_loader import ECGParquetDataloader
import torch.optim as optim
import os
from pathlib import Path

WEIGHT_PATH = os.path.join(Path(__file__).parents[2],"data","raw","weights.csv")
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
        self.results = []
        self.dataset = None
        self.optimizer = None
        self.criterion = None
        self.RESULT_SAVE_PATH = os.path.join(Path(__file__).parents[2],"results","ecg_net_results")

    def train(self):
        for run_number,run in enumerate(self.runs):
            # set optimizer and loss function for this run
            self.begin_run(run=run)
            # convert model to GPU if it is enabled
            if self.__is_gpu_enabled(run):
                self.model.to(run.device)
            # accuracies and losses to be saved
            losses = []
            accuracies = []
            # iterate over epochs
            for epoch in range(run.epochs):
                """
                Load dataset.
                In order to limitations caused by petastorm Dataloader, 
                we load data for 1 batch and iterate over it during epochs
                """
                self.dataset = self.train_loader.new_loader(num_epochs=1, batch_size=run.batch_size)

                # to calculate accuracy each epoch
                correct_prediction_count = 0
                total_predictions = 0

                # iterate over batches
                epoch_loss = 0
                for batch_data in self.dataset:
                    # we need to preprocess parquet data it order to feed it to network
                    features, labels = dict_to_torch(batch_data, feature_count=14)
                    # move data and labels to GPU if enabled
                    if self.__is_gpu_enabled(run):
                        features = features.to(run.device)
                        labels = labels.to(run.device)

                    # get predictions
                    predictions = self.model.forward(features)
                    # calculate loss and update weights for batch
                    loss_out = self.criterion(predictions, labels)
                    self.optimizer.zero_grad()
                    loss_out.backward()
                    self.optimizer.step()
                    epoch_loss += loss_out.item()

                    # add correct and total predictions
                    correct_prediction_count += correct_predictions(predictions, labels)
                    total_predictions += run.batch_size
                print("Run:{} Epoch:{} Loss:{} Accuracy:{}".format([run.loss_fn,run.optimizer_type, run.learning_rate,run.weight_decay],epoch, epoch_loss, correct_prediction_count/total_predictions))
                losses.append(epoch_loss)
                accuracies.append(correct_prediction_count/total_predictions)
                """
                Validation epoch, every x epoch decided by user, we will validate our model
                """
                if epoch % run.epochs_for_val == run.epochs_for_val-1:
                    print("Validating model")
                    self.dataset = self.val_loader.new_loader(num_epochs=1, batch_size=run.batch_size)

                    # to calculate metrics each epoch
                    correct_prediction_count = 0
                    total_predictions = 0
                    val_loss = 0
                    for batch_data in self.dataset:
                        # we need to preprocess parquet data it order to feed it to network
                        features, labels = dict_to_torch(batch_data, feature_count=14)
                        if self.__is_gpu_enabled(run):
                            features = features.to(run.device)
                            labels = labels.to(run.device)
                        predictions = self.model(features)
                        # calculate loss for batch
                        loss_out = self.criterion(predictions, labels)
                        val_loss += loss_out
                        # add correct and total predictions
                        correct_prediction_count += correct_predictions(predictions, labels)
                        total_predictions += run.batch_size
                    print("Validation Epoch:{} Loss:{} Accuracy:{}".format(epoch, epoch_loss, correct_prediction_count/total_predictions*100))
                # save run results for analyzing later
            self.end_run(run_number,run, losses, accuracies)

    def set_optimizer(self, run):
        if run.optimizer_type == "Adam":
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=run.learning_rate, weight_decay=run.weight_decay)
        elif run.optimizer_type == "SGD":
            self.optimizer = optim.SGD(params=self.model.parameters(), lr=run.learning_rate, weight_decay=run.weight_decay,
                                       momentum=run.momentum)

    def set_criterion(self, loss_fn="penalty_dice"):
        if loss_fn == "dice":
            self.criterion = SoftDiceLoss(WEIGHT_PATH)
        elif loss_fn == "penalty_l1":
            self.criterion = L1LossWithPenalty(WEIGHT_PATH)
        elif loss_fn == "penalty_mse":
            self.criterion = MSELossWithPenalty(WEIGHT_PATH)
        elif loss_fn == "penalty_dice":
            self.criterion = SoftDiceLossWithPenalty(WEIGHT_PATH)

    def begin_run(self, run):
        print("Starting new run")
        self.set_optimizer(run)
        self.set_criterion(run.loss_fn)
        self.model.apply(self.__init_weights)

    def end_run(self,idx,run,losses, accuracies):
        save_path = os.path.join(self.RESULT_SAVE_PATH,str(idx))
        torch.save(self.model.state_dict(),save_path+"_model")
        self.__save_results(idx,run,losses, accuracies,save_path)

    def __save_results(self,idx,run,losses, accuracies,path):
        with open(path+"_results.txt", "w") as file:
            file.write(str(run)+"\n")
            file.write(str(losses)+"\n")
            file.write(str(accuracies)+"\n")
            file.close()

    def __is_gpu_enabled(self, run):
        if run.device == 'cuda' and torch.cuda.is_available():
            return True
        else:
            return False

    def __init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


dummy_model = Model_2()
data_path = r'file:C:\Users\ABRA\Desktop\Ders\Yüksek Lisans\BLG561-Deep Learning\deep_learning_interim_project\data\processed'


dummy_params = {
    'learning_rate': [0.01, 0.05],
    'batch_size': [50],
    'epochs': [50],
    'optimizer_type': ["Adam"],
    'loss_fn': ["dice","penalty_ll"],
    'epochs_for_val': [10],
    'weight_decay': [1e-2],
    'momentum': [0],
    'device':["cuda"]
}

mngr = TrainManager(model=dummy_model, processed_data_path=data_path, training_config=dummy_params)
mngr.train()
