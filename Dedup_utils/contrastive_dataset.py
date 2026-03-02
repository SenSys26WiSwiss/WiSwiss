import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

class ContrastiveGaussianDataset(Dataset):
    def __init__(self, dataset, gaussian_noise_std=[0.01, 0.05, 0.1]):
        self.dataset = dataset
        self.gaussian_noise_std = gaussian_noise_std
        self.num_augmentations = len(gaussian_noise_std)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]

        # Generate augmentations
        augmented_samples = [data]
        for std in self.gaussian_noise_std:
            noise = torch.randn_like(data) * std
            augmented_samples.append(data + noise)
        
        # Stack all augmentations
        augmented_batch = torch.stack(augmented_samples, dim=0)
        
        return augmented_batch, label


class ContrastiveDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # Create indices
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.indices)
    
    def __iter__(self):
        # Reset indices if shuffling
        if self.shuffle:
            random.shuffle(self.indices)
        
        # Yield batches
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_data = []
            batch_labels = []
            
            for idx in batch_indices:
                data, label = self.dataset[idx]
                batch_data.append(data)
                batch_labels.append(label)
            
            # Stack batch data
            batch_data = torch.cat(batch_data, dim=0)
            batch_labels = torch.stack(batch_labels, dim=0)
            
            yield batch_data, batch_labels
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size



