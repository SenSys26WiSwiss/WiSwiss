import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

class ReshapedDataset(Dataset):
    def __init__(self, dataset, new_shape):
        self.dataset = dataset
        data = dataset[0][0]
        data_shape = list(data.shape)
        for i in range(min(len(data_shape), len(new_shape))):
            if new_shape[i] > 0:
                data_shape[i] = new_shape[i]
        self.new_shape = data_shape

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        if len(data.shape) == 2:
            data = F.interpolate(data.unsqueeze(0).unsqueeze(0), \
                size=self.new_shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        elif len(data.shape) == 3:
            data = F.interpolate(data.unsqueeze(0).unsqueeze(0), \
                size=self.new_shape, mode='trilinear', align_corners=False).squeeze(0).squeeze(0)

        return data, label


class ReshapedDataset_hupr(Dataset):
    def __init__(self, dataset, new_shape):
        self.dataset = dataset
        data = dataset[0][0]
        data_shape = list(data.shape)[1:]
        for i in range(min(len(data_shape), len(new_shape))):
            if new_shape[i] > 0:
                data_shape[i] = new_shape[i]
        self.new_shape = data_shape

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        data = data[1,:,:,:]
        data = F.interpolate(data.unsqueeze(0).unsqueeze(0),  \
                size=self.new_shape, mode='trilinear', align_corners=False).squeeze(0).squeeze(0)

        return data, label
