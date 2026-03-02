import sys
import os
import math
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

from accelerate import Accelerator

from Models.interp_vit_model import InterpViTModel
from Datasets.hdf5 import HDF5Dataset
from Datasets.subset_dataset import SubsetDataset
from utils import get_current_time

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


def train(accelerator, model, train_loader, optimizer, scheduler, epoch, writer, log_interval, logger):
    model.train()
    device = accelerator.device
    running_loss_sum = 0.0
    running_count = 0

    progress_bar = tqdm(
        total=len(train_loader),
        desc=f'Train round{epoch}/{args.epochs}',
        unit='batch',
        disable=not accelerator.is_main_process
    )

    for (data, target) in train_loader:
        data, target = data.to(device), target.to(device)
        if len(data.shape) == 4:
            data = data.unsqueeze(1)
        output = model(data)
        output = output.view(target.size(0), target.size(1), -1)
        loss = mpjpe(output, target)

        with accelerator.accumulate(model):
            accelerator.backward(loss)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running_loss_sum += loss.item() * data.size(0)
        running_count += data.size(0)

        if accelerator.sync_gradients:
            cur_lr = scheduler.get_last_lr()
            progress_bar.set_postfix(**{'loss': loss.item(), 'lr': cur_lr[0]})

            if accelerator.is_main_process and progress_bar.n % log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] MPJPE Loss: {:.6f} LR: {}'.format(
                    epoch, progress_bar.n, len(train_loader), 100. * progress_bar.n / len(train_loader),
                    loss.item(), ', '.join(['{:.6f}'.format(x) for x in cur_lr])))
                writer.add_scalar('Train/MPJPE', loss.item(), epoch * len(train_loader) + progress_bar.n)
                for i in range(len(cur_lr)):
                    writer.add_scalar(f'Train/lr_group{i}', cur_lr[i], epoch * len(train_loader) + progress_bar.n)

        progress_bar.update(1)

    # Gather loss sum and count across processes for epoch average
    stats = torch.tensor([running_loss_sum, running_count], device=device, dtype=torch.float64)
    gathered = accelerator.gather(stats).view(-1, 2)
    total_loss_sum = gathered[:, 0].sum().item()
    total_count = gathered[:, 1].sum().item()
    epoch_loss_avg = total_loss_sum / total_count if total_count > 0 else 0.0

    if accelerator.is_main_process:
        logger.info(f"Train Epoch {epoch} MPJPE Loss: {epoch_loss_avg:.4f}")
        writer.add_scalar('Train/epoch_MPJPE', epoch_loss_avg, epoch)
    return epoch_loss_avg


def test(accelerator, model, test_loader, epoch, writer, logger):
    model.eval()
    device = accelerator.device
    running_loss_sum = 0.0
    running_count = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, disable=not accelerator.is_main_process):
            data, target = data.to(device), target.to(device)
            if len(data.shape) == 4:
                data = data.unsqueeze(1)
            # data = model.ensure_fixed_shape(data)
            output = model(data)
            output = output.view(target.size(0), target.size(1), -1)
            loss = mpjpe(output, target)
            running_loss_sum += loss.item() * data.size(0)
            running_count += data.size(0)

    # Gather loss sum and count across processes
    stats = torch.tensor([running_loss_sum, running_count], device=device, dtype=torch.float64)
    gathered = accelerator.gather(stats).view(-1, 2)
    total_loss_sum = gathered[:, 0].sum().item()
    total_count = gathered[:, 1].sum().item()
    epoch_loss_avg = total_loss_sum / total_count if total_count > 0 else 0.0

    if accelerator.is_main_process:
        logger.info(f"Test Epoch {epoch} MPJPE Loss: {epoch_loss_avg:.4f}")
        writer.add_scalar('Test/MPJPE', epoch_loss_avg, epoch)
    return epoch_loss_avg



def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device

    writer = None
    logger = logging.getLogger(__name__)
    checkpoint_dir = ""

    # set up tensorboard writer and logging (main process only)
    if accelerator.is_main_process:
        patch_name = f'patch{"x".join([str(x) for x in args.patch_size])}'
        shape_name = f'{args.input_time}x{args.input_range}x{args.input_angle}'
        lr_name = f'lr{args.lr}head{args.head_lr}x'
        max_shape_name = f'{args.max_shape[0]}x{args.max_shape[1]}x{args.max_shape[2]}'
        checkpoint_dir = os.path.join(args.checkpoint_root_dir, \
            f'{args.prefix_name}_{args.model_size}_{max_shape_name}_{patch_name}_{shape_name}_{lr_name}_{CURRENT_TIME}')
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
    if accelerator.is_main_process:
        logger.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

    # set up model
    model = InterpViTModel(patch_size=args.patch_size, patch_stride=args.patch_stride,
                         max_input_shape=args.max_shape,
                         input_channels=2, model_size=args.model_size,
                         train_stage=1, pretrained_ckpt_path=args.model_ckpt_path, device=device,
                         label_dim=args.number_joints * args.joint_dim,
                         qkv_bias=False, qk_scale=None, proj_drop=0., attn_drop=0., path_drop=0., mlp_drop=0.).to(device)

    # diff lr optimizer
    mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias']
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, model.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]
    if accelerator.is_main_process:
        logger.info('The mlp header uses {:.1f} x larger lr'.format(args.head_lr))
    optimizer = optim.AdamW([{'params': base_params, 'lr': args.lr}, {'params': mlp_params, 'lr': args.lr * args.head_lr}], weight_decay=args.weight_decay)
    num_training_steps = math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    model, optimizer, scheduler, train_loader, test_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, test_loader)

    if accelerator.is_main_process:
        logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}, Total optimizer steps: {num_training_steps}")

    best_test_loss = 100000.0
    best_epoch = 0
    best_train_loss = 100000.0
    best_state_dict = None
    for epoch in range(args.epochs):
        train_loss = train(accelerator, model, train_loader, optimizer, scheduler, epoch, writer, args.log_interval, logger)
        if (epoch+1) % args.eval_epochs == 0 or epoch == args.epochs-1:
            test_loss = test(accelerator, model, test_loader, epoch, writer, logger)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
                best_train_loss = train_loss
                best_state_dict = accelerator.unwrap_model(model).state_dict()
        if train_loss < args.es_train_loss:
            break

    if accelerator.is_main_process:
        torch.save(best_state_dict, os.path.join(checkpoint_dir,
            f"epoch{best_epoch}_trainMPJPE{best_train_loss:.4f}_testMPJPE{best_test_loss:.4f}.pth"))
        logger.info(f"Best Test MPJPE Loss: {best_test_loss:.4f}")
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--train_hdf5_path', type=str)
    parser.add_argument('--test_hdf5_path', type=str)
    parser.add_argument('--number_joints', type=int, default=14, help='Number of joints to use.')
    parser.add_argument('--joint_dim', type=int, default=2, help='Dimension of each joint.')
    parser.add_argument('--input_time', type=int, default=8)
    parser.add_argument('--input_range', type=int, default=64)
    parser.add_argument('--input_angle', type=int, default=64)
    parser.add_argument('--subset_indices_path', type=str, default=None)
    parser.add_argument('--test_subset_indices_path', type=str, default=None)
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--max_shape', type=lambda s: [int(x) for x in s.split(',')], default=[32,256,64])

    # model parameters
    parser.add_argument('--patch_size', type=lambda s: [int(x) for x in s.split(',')], default=[2, 16, 16])
    parser.add_argument('--patch_stride', type=lambda s: [int(x) for x in s.split(',')], default=[2, 16, 16])
    parser.add_argument('--model_size', type=str, default='tiny')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps for gradient accumulation.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for training.')
    parser.add_argument('--head_lr', type=float, default=2, help='Learning rate for the mlp head.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Ratio of warmup steps for learning rate scheduler.')
    parser.add_argument('--model_ckpt_path', type=str, default=None, help='Path to the checkpoint of the pre-trained model.')
    parser.add_argument('--prefix_name', type=str, default='finetune', help='Prefix name for the checkpoint.')
    parser.add_argument('--rope_theta_base', type=float, default=10000, help='Base theta for Rope.')
    parser.add_argument('--rope_use_concat', type=bool, default=False, help='Whether to use concat method for Rope.')
    parser.add_argument('--rope_use_add', type=bool, default=True, help='Whether to use add method for Rope.')
    parser.add_argument('--rope_divide_ratio', type=tuple, default=(0.5, 0.5), help='Divide ratio for Rope.')
    parser.add_argument('--rope_learnable_freq', type=bool, default=True, help='Whether to use learnable freqs for Rope.')
    parser.add_argument('--rope_freq_cont', type=bool, default=False, help='Whether to use continuous freqs for Rope.')
    parser.add_argument('--es_train_loss', type=float, default=0.015, help='Early stopping training loss.')
    
    # log parameters
    parser.add_argument('--checkpoint_root_dir', type=str, default='finetune_wipose_checkpoints', help='Path to the directory to save checkpoints.')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging training status.')
    parser.add_argument('--eval_epochs', type=int, default=1, help='Interval for evaluating the model on test set.')
    
    args = parser.parse_args()
    
    main(args)
