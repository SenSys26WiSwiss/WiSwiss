import numpy as np
import torch
from tqdm import tqdm


def compute_distance_augmentation_from_embeddings(embeddings, labels, metric='euclidean'):
    """Compute distance between each original sample and its augmentations only.
    Avoids building the full N×N distance matrix; O(samples * num_augs * dim) instead of O(N² * dim).
    Expects embeddings layout: [orig_0, aug_0_1, ..., aug_0_{K-1}, orig_1, ...] with K = num_augs per sample.
    """
    total_sample_cnt, embed_dim = embeddings.shape
    sample_cnt = labels.shape[0]
    num_augs = total_sample_cnt // sample_cnt
    if total_sample_cnt != sample_cnt * num_augs:
        raise ValueError(
            f"Embeddings count {total_sample_cnt} must equal sample_cnt * num_augs "
            f"({sample_cnt} * {num_augs} = {sample_cnt * num_augs})"
        )
    if num_augs < 2:
        return {0: []}
    n_aug = num_augs - 1
    # originals: indices 0, num_augs, 2*num_augs, ...
    originals = embeddings[np.arange(sample_cnt) * num_augs]  # (sample_cnt, embed_dim)
    # augs: for each sample, rows [i*num_augs+1 .. (i+1)*num_augs]
    aug_indices = np.arange(sample_cnt * num_augs).reshape(sample_cnt, num_augs)[:, 1:]  # (sample_cnt, n_aug)
    augs = embeddings[aug_indices]  # (sample_cnt, n_aug, embed_dim)

    if metric == 'euclidean':
        # Vectorized: (originals - augs)**2, sum over dim, sqrt
        diff = originals[:, None, :] - augs  # (sample_cnt, n_aug, embed_dim)
        distances = np.sqrt((diff ** 2).sum(axis=-1))  # (sample_cnt, n_aug)
    elif metric == 'cosine':
        # Vectorized: 1 - (a·b)/(||a|| ||b||); guard against zero norm to avoid 0/0
        eps = 1e-8
        orig_norms = np.linalg.norm(originals, axis=1, keepdims=True)
        aug_norms = np.linalg.norm(augs, axis=2, keepdims=True)
        originals_norm = originals / (orig_norms + eps)
        augs_norm = augs / (aug_norms + eps)
        cosine_sim = (originals_norm[:, None, :] * augs_norm).sum(axis=-1)  # (sample_cnt, n_aug)
        distances = 1.0 - cosine_sim
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    aug_distance_dict = {j: distances[:, j].tolist() for j in range(n_aug)}
    return aug_distance_dict



def evaluate_dataset_augmentation_chunked(model, data_loader, device, chunk_size, metric='euclidean'):
    model.eval()
    all_aug_distance_dict = {}

    chunk_embeddings = []
    chunk_labels = []
    chunk_cnt = 0
    batch_cnt = 0
    total_batch_cnt = len(data_loader)

    with torch.no_grad():
        for data, label in tqdm(data_loader, desc="Processing chunks"):
            data = data.to(device)
            embeddings = model(data)
            chunk_embeddings.append(embeddings.cpu().numpy())
            chunk_labels.append(label.cpu().numpy())
            batch_cnt += 1

            # Process chunk when it reaches the size limit
            if len(np.concatenate(chunk_embeddings, axis=0)) >= chunk_size or batch_cnt == total_batch_cnt:
                chunk_embeddings_array = np.concatenate(chunk_embeddings, axis=0)
                chunk_labels_array = np.concatenate(chunk_labels, axis=0)

                print(f"Processing chunk {chunk_cnt + 1} with {len(chunk_embeddings_array)} samples...")
                aug_distance_dict = compute_distance_augmentation_from_embeddings(chunk_embeddings_array, chunk_labels_array, metric)

                for k, v in aug_distance_dict.items():
                    if k not in all_aug_distance_dict:
                        all_aug_distance_dict[k] = []
                    all_aug_distance_dict[k].extend(v)

                chunk_cnt += 1
                chunk_embeddings = []
                chunk_labels = []

    return all_aug_distance_dict
    