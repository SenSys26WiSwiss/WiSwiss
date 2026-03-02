import h5py
import sys
import os
# Ensure project root is on path when running from any directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging
import faiss
from collections import defaultdict

from Datasets.hdf5 import HDF5Dataset, save2hdf5
from Datasets.subset_dataset import SubsetDataset
from Datasets.reshape_dataset import ReshapedDataset, ReshapedDataset_hupr

from Dedup_utils.cnn_encoder import EncoderCNN
from Dedup_utils.contrastive_dataset import ContrastiveGaussianDataset, ContrastiveDataLoader
from Dedup_utils.cal_embedding_dist import evaluate_dataset_augmentation_chunked
from utils import get_current_time

CURRENT_TIME: str = get_current_time()

os.environ["OMP_NUM_THREADS"] = "2"         # OpenMP (NumPy, SciPy)
os.environ["MKL_NUM_THREADS"] = "2"         # Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = "2"    # OpenBLAS
torch.set_num_threads(2) 


def get_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for data, label in tqdm(data_loader, desc='Getting embeddings'):
            data = data.to(device)
            if len(data.shape) == 5:
                data = data[:,0,:,:,:]
            embedding = model(data)
            embeddings.append(embedding.cpu().numpy())
            labels.append(label.cpu().numpy())
    return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)


def find_nonduplicate_greedy(embeddings, filter_threshold, logger, k=5, nprobe=10):
    """
    Find the nonduplicate embeddings in the dataset.
    Args:
        embeddings: (N, D)
        filter_threshold: (float)
        k: (int). Number of nearest neighbors to find for each image (k=2 to find the image itself and its closest duplicate)
        nprobe: (int). Set how many clusters to search. Higher is more accurate but slower.
    Returns:
        nonduplicate_indices
    """
    num_embeddings = embeddings.shape[0]
    embedding_dim = embeddings.shape[1]
    # Number of clusters to partition the data into. A good rule of thumb is sqrt(num_embeddings).
    nlist = int(np.sqrt(num_embeddings)) 

    ##########################################################
    # Step 1: Train the FAISS index
    ##########################################################
    # 1. Define the quantizer (the coarse clustering component)
    quantizer = faiss.IndexFlatL2(embedding_dim)
    # 2. Define the main index with euclidean distance
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)
    # 3. Train the index on the embeddings
    # This step clusters the vectors into 'nlist' cells
    logger.info("Training the FAISS index...")
    index.train(embeddings)
    # 4. Add all embeddings to the trained index
    logger.info(f"Adding {num_embeddings} embeddings to the index...")
    index.add(embeddings)

    ##########################################################
    # Step 2: Search for deplicates
    ##########################################################
    index.nprobe = nprobe
    # D stores the distances, I stores the indices (IDs) of the neighbors
    D, I = index.search(embeddings, k)
    pairs = np.argwhere((D > 0) & (D < filter_threshold))
    source_indices = pairs[:, 0]
    target_indices = I[source_indices, pairs[:, 1]]

    logger.info("Building an adjacency list for direct neighbors...")
    # A dictionary where keys are item indices and values are lists of their direct neighbors
    neighbors = defaultdict(list)
    for i in range(len(source_indices)):
        source = source_indices[i]
        target = target_indices[i]
        neighbors[source].append(target)
        neighbors[target].append(source) # Ensure the graph is undirected
    
    ##########################################################
    # Step 3: Greedy searching
    ##########################################################
    logger.info("Running the greedy clustering algorithm...")
    # This array will store the cluster representative for each item.
    # -1 means the item has not been assigned to a cluster yet.
    cluster_assignments = np.full(num_embeddings, -1, dtype=int)
    cluster_id_counter = 0

    indices_to_keep = []
    indices_to_drop = []

    # Iterate through every single item
    for i in range(num_embeddings):
        # If this item has already been claimed by another cluster, skip it.
        if cluster_assignments[i] != -1:
            continue

        # This item is unclaimed, so it becomes a new representative.
        # It represents its own cluster.
        representative_id = i
        indices_to_keep.append(i)
        cluster_assignments[i] = representative_id

        # Now, claim all of its direct, unclaimed neighbors.
        # These neighbors will be dropped.
        for neighbor in neighbors[i]:
            if cluster_assignments[neighbor] == -1:
                cluster_assignments[neighbor] = representative_id # Assign neighbor to this cluster
                indices_to_drop.append(neighbor)
    
    logger.info(f"\n--- De-duplication Results (Greedy Method) ---")
    logger.info(f"Total items: {num_embeddings}")
    logger.info(f"Unique items to keep: {len(indices_to_keep)}")
    logger.info(f"Duplicate items to drop: {len(indices_to_drop)}")
    logger.info(f"Dropped percentage: {len(indices_to_drop) / num_embeddings * 100:.2f}%")

    return sorted(list(indices_to_keep))


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{args.save_dir}/{CURRENT_TIME}.txt", mode="w", encoding="utf-8")
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(args)

    # set up model
    model = EncoderCNN(args.embedding_dim, in_channels=args.input_time, model_name=args.model_name).to(device)
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # set up dataset
    ori_dataset = HDF5Dataset(args.data_path)
    if 'hupr' in args.data_path:
        dataset = ReshapedDataset_hupr(ori_dataset, [args.input_time])
        hupr_ori_dataset = ReshapedDataset_hupr(ori_dataset, [ori_dataset[0][0].shape[1]])
        ori_dataset = hupr_ori_dataset
    else:
        dataset = ReshapedDataset(ori_dataset, [args.input_time])
    data_fname = os.path.basename(args.data_path)
    data_fname_prefix, _ = os.path.splitext(data_fname)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    # get embeddings
    embeddings, labels = get_embeddings(model, loader, device)
    # set up contrastive dataset
    contra_dataset = ContrastiveGaussianDataset(dataset, [args.gaussian_noise_std])
    contra_loader = ContrastiveDataLoader(contra_dataset, batch_size=args.batch_size, shuffle=False)

    # obtain dynamic threshold
    aug_distance_dict = evaluate_dataset_augmentation_chunked(
        model, contra_loader, device, args.chunk_size, metric='euclidean')
    aug_distance_array = aug_distance_dict[0]
    aug_distance_array = np.sort(np.array(aug_distance_array))

    for aug_percentile in args.aug_percentile_list:
        aug_threshold = np.percentile(aug_distance_array, aug_percentile)
        logger.info(f'aug_threshold: {aug_threshold} for percentile {aug_percentile}')
        nonduplicate_indices = find_nonduplicate_greedy(embeddings, aug_threshold, logger)
        nonduplicate_dataset = SubsetDataset(ori_dataset, nonduplicate_indices)
        save2hdf5(nonduplicate_dataset, \
            os.path.join(args.save_dir, f'percentile{aug_percentile}_{data_fname_prefix}_{aug_threshold}.hdf5'), \
            logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--gaussian_noise_std', type=float, default=0.5)
    parser.add_argument('--save_dir', type=str, default='../../dedup_datasets')
    parser.add_argument('--input_time', type=int, default=32)

    # model parameters
    parser.add_argument('--model_name', type=str, default='resnet50', help='Model architecture name')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--chunk_size', type=int, default=10000)

    # dynamic filtering parameters
    parser.add_argument('--aug_percentile_list', type=lambda s: [float(x) for x in s.split(',')], default=[10])

    args = parser.parse_args()
    main(args)

    