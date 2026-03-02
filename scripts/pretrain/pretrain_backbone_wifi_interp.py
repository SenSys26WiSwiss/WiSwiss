import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import argparse
import yaml
from transformers import get_cosine_schedule_with_warmup
import math

from Datasets.hdf5 import HDF5Dataset
from Datasets.subset_dataset import SubsetDataset
from Models.interp_vit_model import InterpViTModel
from Transform_utils.CSIAmp2DFS import csi2dfs_flexible_batched
from utils import get_current_time, AverageMeter

CURRENT_TIME: str = get_current_time()

os.environ["OMP_NUM_THREADS"] = "2"         # OpenMP (NumPy, SciPy)
os.environ["MKL_NUM_THREADS"] = "2"         # Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = "2"    # OpenBLAS
torch.set_num_threads(2)


def train(model, device, pretrain_loader_list, optimizer, scheduler, eval_steps, 
          epoch, writer, log_interval, logger, mask_ratio, 
          dfs_weight, dfs_criterion, ori_length_list, freq_list):
    model.train()
    epoch_loss = AverageMeter()
    epoch_recon_loss = AverageMeter()
    epoch_dfs_loss = AverageMeter()

    pretrain_iter_list = []
    for pretrain_loader in pretrain_loader_list:
        pretrain_iter_list.append(iter(pretrain_loader))
    
    with tqdm(total=eval_steps*len(pretrain_loader_list), desc=f'Train round{epoch}', unit='batch') as pbar:
        while pbar.n < eval_steps*len(pretrain_loader_list):
            for i in range(len(pretrain_iter_list)):
                pretrain_iter = pretrain_iter_list[i]
                try:
                    batch_data = next(pretrain_iter)
                except StopIteration:
                    pretrain_iter = iter(pretrain_loader_list[i])
                    batch_data = next(pretrain_iter)
                    pretrain_iter_list[i] = pretrain_iter
                
                data = batch_data[0].to(device)
                data = data.unsqueeze(1)
                cur_bs = data.shape[0]
                model_output = model(data, mask_ratio=mask_ratio)
                mse_loss = model_output[0]
                # dfs loss
                if dfs_weight > 0:
                    recovered_batch_data = model_output[5]
                    ori_dfs = csi2dfs_flexible_batched(batch_data[0].to(device), ori_length_list[i], freq_list[i])
                    recovered_dfs = csi2dfs_flexible_batched(recovered_batch_data.squeeze(1), ori_length_list[i], freq_list[i])
                    dfs_loss = dfs_criterion(recovered_dfs, ori_dfs)
                else:
                    dfs_loss = torch.tensor([0.0], device=device)
                total_loss = mse_loss + dfs_weight * dfs_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss.update(total_loss.item(), cur_bs)
                epoch_recon_loss.update(mse_loss.item(), cur_bs)
                epoch_dfs_loss.update(dfs_loss.item(), cur_bs)
                pbar.update(1)
                cur_lr = scheduler.get_last_lr()
                pbar.set_postfix(**{'mse_loss': mse_loss.item(), 'dfs_loss': dfs_loss.item(), 'lr': cur_lr})
                
                if pbar.n % log_interval == 0:
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] MSE_loss: {:.6f} DFS_loss: {:.6f} LR: {}'.format(
                        epoch, pbar.n, eval_steps*len(pretrain_iter_list), 
                        100. * pbar.n / (eval_steps*len(pretrain_iter_list)), 
                        mse_loss.item(), dfs_loss.item(), ', '.join(['{:.6f}'.format(x) for x in cur_lr])))
                    writer.add_scalar('Train/loss', total_loss.item(), epoch*eval_steps*len(pretrain_iter_list) + pbar.n)
                    writer.add_scalar('Train/recon_loss', mse_loss.item(), epoch*eval_steps*len(pretrain_iter_list) + pbar.n)
                    writer.add_scalar('Train/dfs_loss', dfs_loss.item(), epoch*eval_steps*len(pretrain_iter_list) + pbar.n)
                    for i in range(len(cur_lr)):
                        writer.add_scalar(f'Train/lr_group{i}', cur_lr[i], epoch*eval_steps*len(pretrain_iter_list) + pbar.n)

    logger.info(f"Epoch {epoch} Loss: {epoch_loss.avg:.4f}")
    writer.add_scalar('Train/epoch_loss', epoch_loss.avg, epoch)
    writer.add_scalar('Train/epoch_recon_loss', epoch_recon_loss.avg, epoch)
    writer.add_scalar('Train/epoch_dfs_loss', epoch_dfs_loss.avg, epoch)
    return epoch_loss.avg


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # set up tensorboard writer
    patch_name = f'patch{"x".join([str(x) for x in args.patch_size])}'
    bs_name = '-'.join([str(bs) for bs in args.batch_sizes])
    data_name_list = []
    ori_length_list = []
    freq_list = []
    if 'signfi' in args.pretrain_datasets:
        data_name_list.append(f'SignFi{args.signfi_input_freq}x{args.signfi_input_time}')
        ori_length_list.append(200)
        freq_list.append(200)
    if 'widar' in args.pretrain_datasets:
        data_name_list.append(f'Widar{args.widar_input_freq}x{args.widar_input_time}')
        ori_length_list.append(1000)
        freq_list.append(1000)
    if 'mmfi' in args.pretrain_datasets:
        data_name_list.append(f'MM-Fi{args.mmfi_input_freq}x{args.mmfi_input_time}')
        ori_length_list.append(32)
        freq_list.append(100)
    data_name = '-'.join(data_name_list)
    mask_name = 'pipe' if args.pipe_mask else 'random'
    dfs_name = f'dfs_{args.dfs_weight}' if args.dfs_weight > 0 else 'no_dfs'
    if args.prefix_name:
        dfs_name = args.prefix_name + '_' + dfs_name
    checkpoint_dir = os.path.join(args.checkpoint_root_dir, \
        f'{args.max_shape[0]}x{args.max_shape[1]}_{dfs_name}_{patch_name}_{args.model_size}_mask{args.mask_ratio}{mask_name}_{data_name}_{bs_name}_lr{args.lr}_{CURRENT_TIME}')
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
    
    # set up datasets: we will include multiple datasets during pretraining
    pretrain_loader_list = []
    max_batch_cnt = -1
    for i, dataset_name in enumerate(args.pretrain_datasets):
        if dataset_name == 'signfi':
            train_dataset = HDF5Dataset(args.signfi_data_path)
            cur_train_loader = DataLoader(train_dataset, batch_size=args.batch_sizes[i], shuffle=True, num_workers=2)
            pretrain_loader_list.append(cur_train_loader)
            max_batch_cnt = max(max_batch_cnt, len(cur_train_loader))
            logger.info(f"SignFi: train dataset size {len(train_dataset)}")

        if dataset_name == 'widar':
            ori_dataset = HDF5Dataset(args.widar_data_path)
            if args.widar_subset_indices_path is not None:
                subset_indices = np.load(args.widar_subset_indices_path)
                train_dataset = SubsetDataset(ori_dataset, subset_indices)
            else:
                train_dataset = ori_dataset
            cur_train_loader = DataLoader(train_dataset, batch_size=args.batch_sizes[i], shuffle=True, num_workers=2)
            pretrain_loader_list.append(cur_train_loader)
            max_batch_cnt = max(max_batch_cnt, len(cur_train_loader))
            logger.info(f"Widar: train dataset size {len(train_dataset)}")

        if dataset_name == 'mmfi':
            ori_dataset = HDF5Dataset(args.mmfi_data_path)
            if args.mmfi_subset_indices_path is not None:
                subset_indices = np.load(args.mmfi_subset_indices_path)
                train_dataset = SubsetDataset(ori_dataset, subset_indices)
            else:
                train_dataset = ori_dataset
            cur_train_loader = DataLoader(train_dataset, batch_size=args.batch_sizes[i], shuffle=True, num_workers=2)
            pretrain_loader_list.append(cur_train_loader)
            max_batch_cnt = max(max_batch_cnt, len(cur_train_loader))
            logger.info(f"MM-Fi: train dataset size {len(train_dataset)}")

    # set up DFS instance
    dfs_criterion = nn.L1Loss()

    # set up model
    model = InterpViTModel(patch_size=args.patch_size, patch_stride=args.patch_stride, 
                         max_input_shape=args.max_shape,
                         input_channels=1, model_size=args.model_size, 
                         train_stage=0, pipe_mask=args.pipe_mask).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.eval_steps < 0:
        args.eval_steps = math.ceil(max_batch_cnt / args.log_interval) * args.log_interval
    num_training_steps = args.eval_steps * args.epochs * len(pretrain_loader_list)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    logger.info(f"Model size: {args.model_size}, eval steps: {args.eval_steps}")
    
    
    for epoch in range(args.epochs):
        train_loss = train(model, device, pretrain_loader_list, optimizer, scheduler, args.eval_steps, 
                           epoch, writer, args.log_interval, logger, args.mask_ratio, 
                           args.dfs_weight, dfs_criterion, ori_length_list, freq_list)
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'epoch{epoch}_trainloss{train_loss:.4f}.pth'))
            
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # SignFi parameters
    parser.add_argument('--signfi_data_path', type=str)
    parser.add_argument('--signfi_input_freq', type=int, default=96)
    parser.add_argument('--signfi_input_time', type=int, default=192)
    
    # Widar parameters
    parser.add_argument('--widar_data_path', type=str)
    parser.add_argument('--widar_subset_indices_path', type=str, default=None)
    parser.add_argument('--widar_input_freq', type=int, default=96)
    parser.add_argument('--widar_input_time', type=int, default=512)

    # MM-Fi parameters
    parser.add_argument('--mmfi_data_path', type=str)
    parser.add_argument('--mmfi_subset_indices_path', type=str, default=None)
    parser.add_argument('--mmfi_input_freq', type=int, default=114)
    parser.add_argument('--mmfi_input_time', type=int, default=32)

    # training data parameters
    parser.add_argument('--pretrain_datasets', type=lambda s: s.split(','), default=['signfi', 'widar', 'mmfi'])
    parser.add_argument('--max_shape', type=lambda s: [int(x) for x in s.split(',')], default=[96, 512])

    # model parameters
    parser.add_argument('--patch_size', type=lambda s: [int(x) for x in s.split(',')], default=[16, 16])
    parser.add_argument('--patch_stride', type=lambda s: [int(x) for x in s.split(',')], default=[16, 16])
    parser.add_argument('--model_size', type=str, default='tiny')
    parser.add_argument('--pipe_mask', type=bool, default=False, help='Whether to use pipe mask.')

    # DFS parameters
    parser.add_argument('--dfs_weight', type=float, default=0.1, help='Weight for DFS loss.')

    # training parameters
    parser.add_argument('--batch_sizes', type=list, default=[32, 32, 32])
    parser.add_argument('--eval_steps', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for training.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Ratio of warmup steps for learning rate scheduler.')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='Ratio of the mask size to the CSI sequence length.')
    parser.add_argument('--save_interval', type=int, default=1, help='Interval for saving checkpoints.')
    
    # log parameters
    parser.add_argument('--checkpoint_root_dir', type=str, default='pretrain_interp_checkpoints', help='Path to the directory to save checkpoints.')
    parser.add_argument('--log_interval', type=int, default=100, help='Interval for logging training status.')
    parser.add_argument('--prefix_name', type=str, default=None, help='Prefix name for the checkpoint.')

    args = parser.parse_args()
    main(args)