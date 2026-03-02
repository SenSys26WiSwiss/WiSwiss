import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Custom contrastive loss that:
    1. Treats sample-augmentation pairs as positive
    2. Treats different-label samples as negative
    3. Does NOT constrain same-label samples (allows intra-class variation)
    """
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        """
        Compute InfoNCE loss
        Args:
            embeddings: [kN, D] tensor where N is batch size, D is projection dimension
            labels: [N] tensor of labels
        """
        batch_size = labels.shape[0]
        total_samples = embeddings.shape[0]
        num_augs = total_samples // batch_size
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Expand labels to match the number of augmentations
        labels_expanded = labels.repeat_interleave(num_augs)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive pairs (sample-augmentation pairs)
        positive_pairs = []
        for i in range(batch_size):
            start_idx = i * num_augs
            end_idx = (i + 1) * num_augs
            # Add all pairs within the same sample's augmentations
            for j in range(start_idx, end_idx):
                for k in range(j + 1, end_idx):
                    positive_pairs.append((j, k))
        
        # Create negative pairs (different labels)
        negative_pairs = []
        for i in range(total_samples):
            for j in range(i + 1, total_samples):
                if labels_expanded[i] != labels_expanded[j]:
                    negative_pairs.append((i, j))

        if not positive_pairs or not negative_pairs:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Compute positive similarities
        pos_similarities = torch.stack([
            similarity_matrix[i, j] for i, j in positive_pairs
        ])

        # Compute negative similarities
        neg_similarities = torch.stack([
            similarity_matrix[i, j] for i, j in negative_pairs
        ])

        # Compute loss: maximize positive similarities, minimize negative similarities
        # Using InfoNCE-style loss
        neg_exp = torch.exp(neg_similarities)

        # For each positive pair, compute contrastive loss against all negatives
        total_loss = 0
        for pos_sim in pos_similarities:
            # Compute probability of positive vs all negatives
            numerator = torch.exp(pos_sim)
            denominator = numerator + neg_exp.sum()
            loss = -torch.log(numerator / denominator)
            total_loss += loss
        
        return total_loss / len(positive_pairs)



