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
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import argparse
from transformers import get_cosine_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_

from Datasets.hdf5 import HDF5Dataset
from Datasets.subset_dataset import SubsetDataset
from Dedup_utils.cnn_encoder import EncoderCNN
from Dedup_utils.contrastive_dataset import ContrastiveGaussianDataset, ContrastiveDataLoader
from Dedup_utils.infonce_loss import InfoNCELoss
from utils import get_current_time
import random

CURRENT_TIME: str = get_current_time()

os.environ["OMP_NUM_THREADS"] = "2"         # OpenMP (NumPy, SciPy)
os.environ["MKL_NUM_THREADS"] = "2"         # Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = "2"    # OpenBLAS
torch.set_num_threads(2)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up tensorboard writer
    embedding_name = f'embed{args.embedding_dim}'
    gaussian_name = f'gaussianstd{args.gaussian_noise_std}'
    checkpoint_dir = os.path.join(args.checkpoint_root_dir, \
        f'{args.model_name}_{embedding_name}_{gaussian_name}_{CURRENT_TIME}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=checkpoint_dir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{checkpoint_dir}/{CURRENT_TIME}.txt", mode="w", encoding="utf-8")
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(f"Writing tensorboard logs to {checkpoint_dir}") 

    # set up dataset
    mcd_train_dataset = HDF5Dataset(args.data_path)
    contra_dataset = ContrastiveGaussianDataset(mcd_train_dataset, [args.gaussian_noise_std])
    contra_loader = ContrastiveDataLoader(contra_dataset, batch_size=args.batch_size, shuffle=True)
    logger.info(f"Train dataset size: {len(mcd_train_dataset)}")

    # set up model
    model = EncoderCNN(args.embedding_dim, in_channels=args.input_time, model_name=args.model_name).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = len(contra_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # set up contrastive loss function
    contra_func = InfoNCELoss(args.temperature)

    for epoch in range(args.epochs):
        with tqdm(total=len(contra_loader), desc=f'Train round{epoch}/{args.epochs}', unit='batch') as pbar:
            for (data, label) in contra_loader:
                data, label = data.to(device), label.to(device)
                # print(data.shape) # torch.Size([256, 96, 200])
                embeddings = model(data)
                loss = contra_func(embeddings, label)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                pbar.update(1)
                cur_lr = scheduler.get_last_lr()
                pbar.set_postfix(**{'loss': loss.item(), 'lr': cur_lr})

                if pbar.n % args.log_interval == 0:
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} LR: {}'.format(
                        epoch, pbar.n, len(contra_loader), 100. * pbar.n / len(contra_loader), 
                        loss.item(), ', '.join(['{:.6f}'.format(x) for x in cur_lr])))
                    writer.add_scalar('Train/loss', loss.item(), epoch * len(contra_loader) + pbar.n)
                    for i in range(len(cur_lr)):
                        writer.add_scalar(f'Train/lr_group{i}', cur_lr[i], epoch * len(contra_loader) + pbar.n)
        
        if (epoch+1) % args.save_epochs == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch{epoch+1}.pth'))
            logger.info(f"Saved model to {checkpoint_dir}/model_epoch{epoch+1}.pth")

    writer.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--input_time', type=int, default=32)
    parser.add_argument('--input_range', type=int, default=128)
    parser.add_argument('--input_angle', type=int, default=32)

    # model parameters
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--embedding_dim', type=int, default=128)

    # contrastive parameters
    parser.add_argument('--gaussian_noise_std', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.07)

    # training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for training.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Ratio of warmup steps for learning rate scheduler.')

    # log parameters
    parser.add_argument('--checkpoint_root_dir', type=str, default='embedding_mcd_checkpoints', help='Path to the directory to save checkpoints.')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging training status.')
    parser.add_argument('--save_epochs', type=int, default=1, help='Interval for evaluating the model on train set.')
    

    args = parser.parse_args()
    main(args)



