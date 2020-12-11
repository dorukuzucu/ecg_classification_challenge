from petastorm.pytorch import DataLoader
from petastorm import TransformSpec, make_batch_reader
from itertools import chain
import torch


class ECGParquetDataloader:
    """
       class to read parquet data
        dataset_path: path of parquet files.
            parquet structures should be the same
            all files will be read by reader
        batch_size: given when creating dataloader. this param handles batchsize for training
        num_epochs: given when creating dataloader. this param handles number of epochs for training
       """
    def __init__(self,path):
        self.dataset_path = path

    # transformation to be used for data. Needs to be handled according to data(if possible)
    def _row_to_tensor(parquet_row):
        feature_list = list(chain.from_iterable(parquet_row.values()))
        return torch.stack((feature_list))

    # return transform spec for data
    def _transform_spec_tensor(self):
        return TransformSpec(self._row_to_tensor)

    # get data loader
    def new_loader(self, num_epochs,batch_size):
        with DataLoader(make_batch_reader(self.dataset_path,
                                          num_epochs=num_epochs
                                          #,transform_spec=self._transform_spec_tensor()
                                          ),batch_size=batch_size) as train_loader:
            return train_loader

