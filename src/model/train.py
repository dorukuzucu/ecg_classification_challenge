from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader
from src.model.utils import create_dict_combination
from src.data.data_loader import ECGParquetDataloader


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
            new_dataset = self.dataloader.new_loader(run.epochs, run.batch_size)
            # TODO parse dataset and feed to network
            pass


    def __generate_combination_of_params(self):
        self.runs = create_dict_combination(self.config)




