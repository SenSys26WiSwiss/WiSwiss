from torch.utils.data import Dataset
import h5py
import torch
import numpy as np
from tqdm import tqdm

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.hdf5_file = h5py.File(hdf5_path, 'r')  # Open HDF5 in read-only mode
        self.num_samples = len(self.hdf5_file['data'])
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = torch.from_numpy(self.hdf5_file['data'][idx]).float()
        label = torch.from_numpy(np.array(self.hdf5_file['label'][idx]))
        if len(label.shape) < 2:
            label = label.long()
        else:
            label = label.float()
        return data, label
    
    def close(self):
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            self.hdf5_file = None
    
    def __del__(self):
        self.close()


class HDF5Dataset_RADDet(Dataset):
    def __init__(self, hdf5_path):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.hdf5_file = h5py.File(hdf5_path, 'r')  # Open HDF5 in read-only mode
        self.num_samples = len(self.hdf5_file['data'])
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = torch.from_numpy(self.hdf5_file['data'][idx]).float()
        label = torch.from_numpy(np.array(self.hdf5_file['label'][idx])).float()
        raw_boxes = torch.from_numpy(np.array(self.hdf5_file['raw'][idx])).float()
        return data, label, raw_boxes
    
    def close(self):
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            self.hdf5_file = None
    
    def __del__(self):
        self.close()


def save2hdf5(dataset, hdf5_save_path, logger=None):
    with h5py.File(hdf5_save_path, 'w') as f:
        num_samples = len(dataset)
        data_shape = dataset[0][0].shape
        label_shape = dataset[0][1].shape
        if logger is not None:
            logger.info(f'Saving {num_samples} samples to {hdf5_save_path}')
            logger.info(f'Data shape: {data_shape}')
            logger.info(f'Label shape: {label_shape}')
        else:
            print(f'Saving {num_samples} samples to {hdf5_save_path}')
            print(f'Data shape: {data_shape}')
            print(f'Label shape: {label_shape}')

        # Pre-allocate HDF5 dataset
        data_dset = f.create_dataset('data', (num_samples, *data_shape), dtype=np.float32)
        label_dset = f.create_dataset('label', (num_samples, *label_shape), dtype=np.float32)

        # Populate datasets
        for idx in tqdm(range(num_samples), desc='Saving to HDF5'):
            data, label = dataset[idx]
            data_dset[idx] = data.numpy()
            label_dset[idx] = label.numpy()


