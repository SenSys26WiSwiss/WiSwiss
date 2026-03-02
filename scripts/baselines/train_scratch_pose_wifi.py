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
from transformers import get_cosine_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_

from Models.rope_vit_model import RopeViTModel
from Datasets.hdf5 import HDF5Dataset
from Datasets.subset_dataset import SubsetDataset
from utils import get_current_time, AverageMeter

CURRENT_TIME: str = get_current_time()

os.environ["OMP_NUM_THREADS"] = "2"         # OpenMP (NumPy, SciPy)
os.environ["MKL_NUM_THREADS"] = "2"         # Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = "2"    # OpenBLAS
torch.set_num_threads(2)


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def train(model, device, train_loader, optimizer, scheduler, epoch, writer, log_interval, logger):
    model.train()
    epoch_loss = AverageMeter()
    
    with tqdm(total=len(train_loader), desc=f'Train round{epoch}/{args.epochs}', unit='batch') as pbar:
        for (data, target) in train_loader:
            data, target = data.to(device), target.to(device)
            data = data.unsqueeze(1)
            output = model(data)
            output = output.view(target.size(0), target.size(1), -1)
            
            optimizer.zero_grad()
            loss = mpjpe(output, target)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            pbar.update(1)
            cur_lr = scheduler.get_last_lr()
            epoch_loss.update(loss.item(), data.size(0))
            pbar.set_postfix(**{'loss': loss.item(), 'lr': cur_lr})
            
            if pbar.n % log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] MPJPE Loss: {:.6f} LR: {}'.format(
                    epoch, pbar.n, len(train_loader), 100. * pbar.n / len(train_loader), 
                    loss.item(), ', '.join(['{:.6f}'.format(x) for x in cur_lr])))
                writer.add_scalar('Train/MPJPE', loss.item(), epoch * len(train_loader) + pbar.n)
                for i in range(len(cur_lr)):
                    writer.add_scalar(f'Train/lr_group{i}', cur_lr[i], epoch * len(train_loader) + pbar.n)

    logger.info(f"Train Epoch {epoch} MPJPE Loss: {epoch_loss.avg:.4f}")
    return epoch_loss.avg


def test(model, device, test_loader, epoch, writer, logger):
    model.eval()
    epoch_loss = AverageMeter()

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            data = data.unsqueeze(1)
            output = model(data)
            output = output.view(target.size(0), target.size(1), -1)
            loss = mpjpe(output, target)
            epoch_loss.update(loss.item(), data.size(0))

    logger.info(f"Test Epoch {epoch} MPJPE Loss: {epoch_loss.avg:.4f}")
    writer.add_scalar('Test/MPJPE', epoch_loss.avg, epoch)
    return epoch_loss.avg



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up tensorboard writer
    patch_name = f'patch{"x".join([str(x) for x in args.patch_size])}'
    shape_name = f'{args.input_freq}x{args.input_time}'
    lr_name = f'lr{args.lr}'
    rope_name = 'concat' if args.rope_use_concat else 'add'
    rope_name = rope_name + ('_learnable' if args.rope_learnable_freq else '_const')
    if args.rope_use_concat:
        rope_name = rope_name + ('_cont' if args.rope_freq_cont else '_sep')
    checkpoint_dir = os.path.join(args.checkpoint_root_dir, \
        f'{args.prefix_name}_{args.model_size}_{rope_name}_{patch_name}_{shape_name}_{lr_name}_{CURRENT_TIME}')
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
    ori_train_dataset = HDF5Dataset(args.train_hdf5_path)
    ori_test_dataset = HDF5Dataset(args.test_hdf5_path)
    if args.subset_indices_path is not None:
        subset_indices = np.load(args.subset_indices_path)
        train_dataset = SubsetDataset(ori_train_dataset, subset_indices)
    else:
        train_dataset = ori_train_dataset
    if args.test_subset_indices_path is not None:
        subset_indices = np.load(args.test_subset_indices_path)
        test_dataset = SubsetDataset(ori_test_dataset, subset_indices)
    else:
        test_dataset = ori_test_dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    logger.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

    # set up model
    model = RopeViTModel(patch_size=args.patch_size, patch_stride=args.patch_stride, 
                         input_channels=1, model_size=args.model_size, 
                         train_stage=2, pretrained_ckpt_path=None, device=device, 
                         label_dim=args.number_joints * args.joint_dim, 
                         rope_use_concat=args.rope_use_concat, rope_use_add=args.rope_use_add, 
                         rope_divide_ratio=args.rope_divide_ratio, rope_learnable_freq=args.rope_learnable_freq, rope_freq_cont=args.rope_freq_cont, 
                         qkv_bias=False, qk_scale=None, proj_drop=0., attn_drop=0., path_drop=0., mlp_drop=0.).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    best_test_loss = 100000
    best_epoch = 0
    best_train_loss = 100000
    best_state_dict = None
    for epoch in range(args.epochs):
        train_loss = train(model, device, train_loader, optimizer, scheduler, epoch, writer, args.log_interval, logger)
        if (epoch+1) % args.eval_epochs == 0 or epoch == args.epochs-1:
            test_loss = test(model, device, test_loader, epoch, writer, logger)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
                best_train_loss = train_loss
                best_state_dict = model.state_dict()
        if train_loss < args.es_train_loss:
            break
    
    torch.save(best_state_dict, os.path.join(checkpoint_dir, 
        f"epoch{best_epoch}_trainMPJPE{best_train_loss:.4f}_testMPJPE{best_test_loss:.4f}.pth"))
    
    logger.info(f"Best Test MPJPE Loss: {best_test_loss:.4f}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--train_hdf5_path', type=str)
    parser.add_argument('--test_hdf5_path', type=str)
    parser.add_argument('--number_joints', type=int, default=18, help='Number of joints to use.')
    parser.add_argument('--joint_dim', type=int, default=2, help='Dimension of each joint.')
    parser.add_argument('--input_freq', type=int, default=96)
    parser.add_argument('--input_time', type=int, default=16)
    parser.add_argument('--subset_indices_path', type=str, default=None)
    parser.add_argument('--test_subset_indices_path', type=str, default=None)
    parser.add_argument('--task_name', type=str, default=None)

    # model parameters
    parser.add_argument('--patch_size', type=lambda s: [int(x) for x in s.split(',')], default=[16, 16])
    parser.add_argument('--patch_stride', type=lambda s: [int(x) for x in s.split(',')], default=[16, 16])
    parser.add_argument('--model_size', type=str, default='tiny')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for training.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Ratio of warmup steps for learning rate scheduler.')
    parser.add_argument('--prefix_name', type=str, default='scratch', help='Prefix name for the checkpoint.')
    parser.add_argument('--rope_theta_base', type=float, default=10000, help='Base theta for Rope.')
    parser.add_argument('--rope_use_concat', type=bool, default=False, help='Whether to use concat method for Rope.')
    parser.add_argument('--rope_use_add', type=bool, default=True, help='Whether to use add method for Rope.')
    parser.add_argument('--rope_divide_ratio', type=tuple, default=(0.5, 0.5), help='Divide ratio for Rope.')
    parser.add_argument('--rope_learnable_freq', type=bool, default=True, help='Whether to use learnable freqs for Rope.')
    parser.add_argument('--rope_freq_cont', type=bool, default=False, help='Whether to use continuous freqs for Rope.')
    parser.add_argument('--es_train_loss', type=float, default=0.015, help='Early stopping training loss.')
    
    # log parameters
    parser.add_argument('--checkpoint_root_dir', type=str, default='scratch_wipose_checkpoints', help='Path to the directory to save checkpoints.')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging training status.')
    parser.add_argument('--eval_epochs', type=int, default=1, help='Interval for evaluating the model on test set.')
    
    args = parser.parse_args()
    
    main(args)
