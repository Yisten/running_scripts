from typing import Dict, List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data.dataset import get_logname_tokens, get_data_collate, ProcessedDataset

class NuplanDataModule(LightningDataModule):

    def __init__(self,data_root, train_path, val_path,dataclass,
                 train_batch_size,val_batch_size,num_workers):
        super(NuplanDataModule, self).__init__()
        self.data_root = data_root
        self.train_path = train_path
        self.val_path = val_path
        self.dataclass = dataclass
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
    def prepare_data(self) -> None:

        train_data_logname_tokens = get_logname_tokens(self.train_path)
        val_data_logname_tokens = get_logname_tokens(self.val_path)

        self.train_dataset = ProcessedDataset(
            self.data_root, train_data_logname_tokens, self.dataclass
        )
        self.val_dataset = ProcessedDataset(
            self.data_root, val_data_logname_tokens, self.dataclass
        )
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, shuffle=True,
            batch_size=self.train_batch_size, num_workers=self.num_workers,
            collate_fn=get_data_collate(self.dataclass)
            )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, shuffle=True,
            batch_size=self.val_batch_size, num_workers=self.num_workers,
            collate_fn=get_data_collate(self.dataclass)
            )
    def transfer_batch_to_device(self,batch,device):
        for key in batch.keys():
            batch[key] = batch[key].to_device(device)
        return batch