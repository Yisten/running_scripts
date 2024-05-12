import torch
import logging
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
import os
from pathlib import Path
import gzip
import pickle
from torch.utils.data.dataloader import DataLoader

class ProcessedDataset(Dataset):
    def __init__(self, data_root, data_logname_tokens, dataclass):
        super(ProcessedDataset,self).__init__()
        self.data_root = data_root
        self.dataclass = dataclass
        self.dataclass_names = list(dataclass.keys())
        self.data_list = data_logname_tokens

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_file = os.path.join(self.data_root,self.data_list[idx])
        data = {}
        for name in self.dataclass_names:
            data_path = data_file+"_"+name+".gz"
            with gzip.open(data_path,"r") as f:
                data[name] = self.dataclass[name].deserialize(
                    pickle.load(f)
                    ).to_feature_tensor()
        return data
    
def get_logname_tokens(file_path):
    data_list = [] #log_name+token
    with open(file_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break;
            data_list.append(Path(line.strip()))
    return data_list

def get_data_collate(dataclass):
    
    def _fn(batch):
        batched_data = {}
        for name, datatype in dataclass.items():
            datalist= [data[name] for data in batch]
            batched_data[name] = datatype.collate(datalist)
        return batched_data
    
    return _fn

if __name__ == '__main__':
    from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
    from nuplan.planning.training.preprocessing.features.pyg_feature import HiVTModelFeature
    dataclass = {"hivt_pyg": HiVTModelFeature, "trajectory": Trajectory}
    
    data_root = "/data2/nuplan_data/"
    train_path = "data/cache_train.txt"
    val_path = "data/cache_val.txt"
    train_data_logname_tokens = get_logname_tokens(train_path)
    val_data_logname_tokens = get_logname_tokens(val_path)
    train_dataset = ProcessedDataset(
        data_root,train_data_logname_tokens,dataclass
        )
    val_dataset = ProcessedDataset(
        data_root,val_data_logname_tokens,dataclass
        )
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=24, num_workers=int(os.cpu_count()/2),
        collate_fn=get_data_collate(dataclass)
        )
    for batch in train_dataloader:
        batch