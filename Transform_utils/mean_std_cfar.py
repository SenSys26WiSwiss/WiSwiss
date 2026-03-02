import torch

class MeanStdCFAR:
    def __init__(self, lambda_std):
        self.lambda_std = lambda_std
    
    def __call__(self, batch_tra):
        B = batch_tra.shape[0]

        # Reshape to [batch_size, -1] to flatten all non-batch dimensions
        flattened_tra = batch_tra.view(B, -1)  # Shape: [B, N], where N = product of other dims

        # Compute per-sample mean and std
        tra_power_mean = flattened_tra.mean(dim=1)
        tra_power_std = flattened_tra.std(dim=1)

        # Reshape back to [B, 1, 1, ...] for broadcasting with original rd_power
        # (Adds singleton dims to match original shape except batch dim)
        for _ in range(batch_tra.dim() - 1):
            tra_power_mean = tra_power_mean.unsqueeze(-1)
            tra_power_std = tra_power_std.unsqueeze(-1)
        
        # Computer mask and apply
        tra_mask = (batch_tra > (tra_power_mean + self.lambda_std * tra_power_std))
        masked_tra_mask = tra_mask * batch_tra

        return masked_tra_mask

    