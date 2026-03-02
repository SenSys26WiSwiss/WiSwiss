import numpy as np
from scipy import signal
import torch
import torch.nn.functional as F


class DiffCFAR_TRA_batch:
    def __init__(self, mask, threshold, temperature):
        self.mask = mask
        self.threshold = threshold
        self.temperature = temperature
    
    def __call__(self, batch_tra):
        B = batch_tra.shape[0]
        batch_tra_reshaped = batch_tra.reshape(B*batch_tra.shape[1], 1, *batch_tra.shape[2:])
        num_valid_cells_in_window = signal.convolve2d(np.ones((batch_tra.shape[2:]), dtype=float), self.mask, mode='same')
        num_valid_cells_in_window = torch.from_numpy(num_valid_cells_in_window).float().to(batch_tra.device)

        # Use PyTorch's convolution for 2D convolution (differentiable)
        # Reshape mask for torch conv2d [out_channels, in_channels, height, width]
        mask_for_conv = torch.from_numpy(self.mask).float().reshape(1, 1, *self.mask.shape).to(batch_tra.device)

        # Perform convolution with padding to maintain size
        # Calculate the padding amounts
        # For 2D padding, (pad_left, pad_right, pad_top, pad_bottom)
        # PyTorch's F.pad expects padding for the last dimensions first.
        # If rd_power_reshaped is (N, C, H, W) and self.mask.shape gives (kernel_H, kernel_W)
        pad_h = self.mask.shape[0] // 2
        pad_w = self.mask.shape[1] // 2

        # Apply replicate padding
        # The padding tuple for F.pad is (padding_left, padding_right, padding_top, padding_bottom)
        # for the last two dimensions.
        batch_tra_padded = F.pad(batch_tra_reshaped, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
        tra_windowed_sum = F.conv2d(batch_tra_padded, mask_for_conv, padding=0)
        tra_windowed_sum = tra_windowed_sum.reshape(*batch_tra.shape)

        # Compute average noise power
        tra_avg_noise_power = tra_windowed_sum / num_valid_cells_in_window
        # replace 0 with 1e-3
        tra_avg_noise_power = torch.where(tra_avg_noise_power == 0, torch.ones_like(tra_avg_noise_power) * 1e-3, tra_avg_noise_power)
        # Compute SNR
        tra_snr = batch_tra / tra_avg_noise_power

        # Apply threshold (differentiable with sigmoid approximation)
        # Use sigmoid with high temperature for soft thresholding
        soft_threshold = torch.sigmoid(self.temperature * (tra_snr - self.threshold))
        
        return soft_threshold




