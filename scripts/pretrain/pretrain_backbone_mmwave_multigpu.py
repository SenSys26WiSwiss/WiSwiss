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
import math
from transformers import get_cosine_schedule_with_warmup

from accelerate import Accelerator

from Datasets.hdf5 import HDF5Dataset
from Datasets.subset_dataset import SubsetDataset
from Models.rope_vit_model import RopeViTModel
from Transform_utils.cfar2d import DiffCFAR_TRA_batch
from Transform_utils.mean_std_cfar import MeanStdCFAR
from utils import get_current_time, AverageMeter

CURRENT_TIME: str = get_current_time()

os.environ["OMP_NUM_THREADS"] = "2"         # OpenMP (NumPy, SciPy)
os.environ["MKL_NUM_THREADS"] = "2"         # Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = "2"    # OpenBLAS
torch.set_num_threads(2)


def train(accelerator, model, pretrain_loader_list, optimizer, scheduler, eval_steps,
          epoch, writer, log_interval, logger, mask_ratio,
          cfar_weight, cfar_instance, cfar_criterion):
    """
    The training loop for one epoch. Uses accelerator for multi-GPU / gradient accumulation.
    """
    model.train()
    epoch_loss = AverageMeter()
    epoch_recon_loss = AverageMeter()
    epoch_cfar_loss = AverageMeter()

    pretrain_iter_list = [iter(loader) for loader in pretrain_loader_list]

    progress_bar = tqdm(
        total=eval_steps * len(pretrain_loader_list),
        desc=f'Train epoch {epoch}',
        unit='batch',
        disable=not accelerator.is_main_process
    )

    device = accelerator.device

    while progress_bar.n < eval_steps * len(pretrain_loader_list):
        for i in range(len(pretrain_iter_list)):
            pretrain_iter = pretrain_iter_list[i]
            try:
                batch_data = next(pretrain_iter)
            except StopIteration:
                pretrain_iter = iter(pretrain_loader_list[i])
                batch_data = next(pretrain_iter)
                pretrain_iter_list[i] = pretrain_iter

            with accelerator.accumulate(model):
                data = batch_data[0].to(device)
                data = data.unsqueeze(1)
                cur_bs = data.shape[0]

                model_output = model(data, mask_ratio=mask_ratio)
                mse_loss = model_output[0]

                if cfar_weight > 0 and cfar_instance is not None and cfar_criterion is not None:
                    recovered_batch_data = model_output[5]
                    ori_cfar_threshold = cfar_instance(batch_data[0].to(device))
                    recovered_cfar_threshold = cfar_instance(recovered_batch_data.squeeze(1))
                    cfar_loss = cfar_criterion(ori_cfar_threshold, recovered_cfar_threshold)
                else:
                    cfar_loss = torch.tensor([0.0], device=device)

                total_loss = mse_loss + cfar_weight * cfar_loss

                accelerator.backward(total_loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                avg_total_loss = accelerator.gather(total_loss).mean()
                avg_mse_loss = accelerator.gather(mse_loss).mean()
                avg_cfar_loss = accelerator.gather(cfar_loss).mean()

                epoch_loss.update(avg_total_loss.item(), cur_bs * accelerator.num_processes * accelerator.gradient_accumulation_steps)
                epoch_recon_loss.update(avg_mse_loss.item(), cur_bs * accelerator.num_processes * accelerator.gradient_accumulation_steps)
                epoch_cfar_loss.update(avg_cfar_loss.item(), cur_bs * accelerator.num_processes * accelerator.gradient_accumulation_steps)

                cur_lr = scheduler.get_last_lr()
                progress_bar.set_postfix(**{'mse_loss': avg_mse_loss.item(), 'cfar_loss': avg_cfar_loss.item(), 'lr': cur_lr[0]})

            progress_bar.update(1)

            if accelerator.is_main_process and progress_bar.n % log_interval == 0:
                if accelerator.sync_gradients:
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] MSE_loss: {:.6f} CFAR_loss: {:.6f} LR: {}'.format(
                        epoch, progress_bar.n, eval_steps * len(pretrain_iter_list),
                        100. * progress_bar.n / (eval_steps * len(pretrain_iter_list)),
                        avg_mse_loss.item(), avg_cfar_loss.item(), ', '.join([f'{x:.6f}' for x in cur_lr])))

                    global_step = epoch * eval_steps * len(pretrain_iter_list) + progress_bar.n
                    writer.add_scalar('Train/loss', avg_total_loss.item(), global_step)
                    writer.add_scalar('Train/recon_loss', avg_mse_loss.item(), global_step)
                    writer.add_scalar('Train/cfar_loss', avg_cfar_loss.item(), global_step)
                    for i_lr, lr_val in enumerate(cur_lr):
                        writer.add_scalar(f'Train/lr_group{i_lr}', lr_val, global_step)

    if accelerator.is_main_process:
        logger.info(f"Epoch {epoch} Loss: {epoch_loss.avg:.4f}")
        writer.add_scalar('Train/epoch_loss', epoch_loss.avg, epoch)
        writer.add_scalar('Train/epoch_recon_loss', epoch_recon_loss.avg, epoch)
        writer.add_scalar('Train/epoch_cfar_loss', epoch_cfar_loss.avg, epoch)

    return epoch_loss.avg


def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device

    writer = None
    logger = logging.getLogger(__name__)
    checkpoint_dir = ""

    if accelerator.is_main_process:
        patch_name = f'patch{"x".join([str(x) for x in args.patch_size])}'
        bs_name = '-'.join([str(bs) for bs in args.batch_sizes])
        data_name_list = []
        if 'dcdr' in args.pretrain_datasets:
            data_name_list.append(f'DCDR{args.dcdr_input_time}x{args.dcdr_input_range}x{args.dcdr_input_angle}')
        if 'mcd' in args.pretrain_datasets:
            data_name_list.append(f'MCD{args.mcd_input_time}x{args.mcd_input_range}x{args.mcd_input_angle}')
        if 'rtpose' in args.pretrain_datasets:
            data_name_list.append(f'RT-Pose{args.rtpose_input_time}x{args.rtpose_input_range}x{args.rtpose_input_angle}')
        if 'xrf55' in args.pretrain_datasets:
            data_name_list.append(f'XRF55{args.xrf55_input_time}x{args.xrf55_input_range}x{args.xrf55_input_angle}')
        if 'hupr' in args.pretrain_datasets:
            data_name_list.append(f'HUPR{args.hupr_input_time}x{args.hupr_input_range}x{args.hupr_input_angle}')
        if 'raddet' in args.pretrain_datasets:
            data_name_list.append(f'RADDet{args.raddet_input_time}x{args.raddet_input_range}x{args.raddet_input_angle}')
        data_name = '-'.join(data_name_list)
        rope_name = 'concat' if args.rope_use_concat else 'add'
        rope_name = rope_name + ('_learnable' if args.rope_learnable_freq else '_const')
        if args.rope_use_concat:
            rope_name = rope_name + ('_cont' if args.rope_freq_cont else '_sep')
        mask_name = 'pipe' if args.pipe_mask else 'random'
        cfar_name = f'cfar_{args.cfar_type}{args.cfar_weight}_threshold{args.cfar_threshold}' if args.cfar_weight > 0 else 'no_cfar'
        if args.prefix_name:
            cfar_name = args.prefix_name + '_' + cfar_name
        checkpoint_dir = os.path.join(args.checkpoint_root_dir,
            f'{cfar_name}_{rope_name}_{patch_name}_{args.model_size}_mask{args.mask_ratio}{mask_name}_{data_name}_{bs_name}_lr{args.lr}_{CURRENT_TIME}')
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
        logger.info(args)
        logger.info(f"Writing tensorboard logs to {checkpoint_dir}")

    # set up datasets: same logic as step3_pretrain_backbone_mmwave.py
    pretrain_loader_list = []
    max_batch_cnt = -1
    for i, dataset_name in enumerate(args.pretrain_datasets):
        if dataset_name == 'dcdr':
            train_dataset = HDF5Dataset(args.dcdr_data_path)
            cur_train_loader = DataLoader(train_dataset, batch_size=args.batch_sizes[i], shuffle=True, num_workers=2, pin_memory=True)
            pretrain_loader_list.append(cur_train_loader)
            max_batch_cnt = max(max_batch_cnt, len(cur_train_loader))
            if accelerator.is_main_process:
                logger.info(f"SignFi: train dataset size {len(train_dataset)}")

        if dataset_name == 'mcd':
            ori_dataset = HDF5Dataset(args.mcd_data_path)
            if args.mcd_subset_indices_path is not None:
                subset_indices = np.load(args.mcd_subset_indices_path)
                train_dataset = SubsetDataset(ori_dataset, subset_indices)
            else:
                train_dataset = ori_dataset
            cur_train_loader = DataLoader(train_dataset, batch_size=args.batch_sizes[i], shuffle=True, num_workers=2, pin_memory=True)
            pretrain_loader_list.append(cur_train_loader)
            max_batch_cnt = max(max_batch_cnt, len(cur_train_loader))
            if accelerator.is_main_process:
                logger.info(f"MCD: train dataset size {len(train_dataset)}")

        if dataset_name == 'rtpose':
            ori_dataset = HDF5Dataset(args.rtpose_data_path)
            if args.rtpose_subset_indices_path is not None:
                subset_indices = np.load(args.rtpose_subset_indices_path)
                train_dataset = SubsetDataset(ori_dataset, subset_indices)
            else:
                train_dataset = ori_dataset
            cur_train_loader = DataLoader(train_dataset, batch_size=args.batch_sizes[i], shuffle=True, num_workers=2, pin_memory=True)
            pretrain_loader_list.append(cur_train_loader)
            max_batch_cnt = max(max_batch_cnt, len(cur_train_loader))
            if accelerator.is_main_process:
                logger.info(f"RT-Pose: train dataset size {len(train_dataset)}")

        if dataset_name == 'xrf55':
            ori_dataset = HDF5Dataset(args.xrf55_data_path)
            if args.xrf55_subset_indices_path is not None:
                subset_indices = np.load(args.xrf55_subset_indices_path)
                train_dataset = SubsetDataset(ori_dataset, subset_indices)
            else:
                train_dataset = ori_dataset
            cur_train_loader = DataLoader(train_dataset, batch_size=args.batch_sizes[i], shuffle=True, num_workers=2, pin_memory=True)
            pretrain_loader_list.append(cur_train_loader)
            max_batch_cnt = max(max_batch_cnt, len(cur_train_loader))
            if accelerator.is_main_process:
                logger.info(f"XRF55: train dataset size {len(train_dataset)}")

        if dataset_name == 'hupr':
            ori_dataset = HDF5Dataset(args.hupr_data_path)
            if args.hupr_subset_indices_path is not None:
                subset_indices = np.load(args.hupr_subset_indices_path)
                train_dataset = SubsetDataset(ori_dataset, subset_indices)
            else:
                train_dataset = ori_dataset
            cur_train_loader = DataLoader(train_dataset, batch_size=args.batch_sizes[i], shuffle=True, num_workers=2, pin_memory=True)
            pretrain_loader_list.append(cur_train_loader)
            max_batch_cnt = max(max_batch_cnt, len(cur_train_loader))
            if accelerator.is_main_process:
                logger.info(f"HuPR: train dataset size {len(train_dataset)}")

        if dataset_name == 'raddet':
            ori_dataset = HDF5Dataset(args.raddet_data_path)
            if args.raddet_subset_indices_path is not None:
                subset_indices = np.load(args.raddet_subset_indices_path)
                train_dataset = SubsetDataset(ori_dataset, subset_indices)
            else:
                train_dataset = ori_dataset
            cur_train_loader = DataLoader(train_dataset, batch_size=args.batch_sizes[i], shuffle=True, num_workers=2, pin_memory=True)
            pretrain_loader_list.append(cur_train_loader)
            max_batch_cnt = max(max_batch_cnt, len(cur_train_loader))
            if accelerator.is_main_process:
                logger.info(f"RADDet: train dataset size {len(train_dataset)}")

    # set up CFAR instance: same logic as step3_pretrain_backbone_mmwave.py
    if args.cfar_weight > 0:
        if args.cfar_type == '2d':
            win_width, win_height, guard_width, guard_height = args.cfar_2d_win_param
            mask = np.ones((2 * win_height + 1, 2 * win_width + 1), dtype=bool)
            mask[win_height - guard_height:win_height + 1 + guard_height, win_width - guard_width:win_width + 1 + guard_width] = 0
            cfar_instance = DiffCFAR_TRA_batch(mask, args.cfar_threshold, args.cfar_temperature)
            cfar_criterion = nn.L1Loss()
        elif args.cfar_type == 'mean_std':
            cfar_instance = MeanStdCFAR(lambda_std=args.cfar_lambda_std)
            cfar_criterion = nn.L1Loss()
        else:
            raise ValueError(f"Invalid CFAR type: {args.cfar_type}")
    else:
        cfar_instance, cfar_criterion = None, None

    # set up model: same constructor as step3_pretrain_backbone_mmwave.py (no use_ape)
    model = RopeViTModel(patch_size=args.patch_size, patch_stride=args.patch_stride,
                         input_channels=1, model_size=args.model_size,
                         train_stage=0, pipe_mask=args.pipe_mask,
                         rope_use_concat=args.rope_use_concat, rope_use_add=args.rope_use_add,
                         rope_divide_ratio=args.rope_divide_ratio, rope_learnable_freq=args.rope_learnable_freq, rope_freq_cont=args.rope_freq_cont,
                         qkv_bias=False, qk_scale=None, proj_drop=0., attn_drop=0., path_drop=0., mlp_drop=0.)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.eval_steps < 0:
        args.eval_steps = math.ceil(max_batch_cnt / args.log_interval) * args.log_interval
    num_training_steps = math.ceil(args.eval_steps * args.epochs * len(pretrain_loader_list) / args.gradient_accumulation_steps)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    if accelerator.is_main_process:
        logger.info(f"Model size: {args.model_size}, eval steps: {args.eval_steps}")

    # prepare for distributed training
    model, optimizer, scheduler, *pretrain_loader_list = accelerator.prepare(model, optimizer, scheduler, *pretrain_loader_list)

    for epoch in range(args.epochs):
        train_loss = train(accelerator, model, pretrain_loader_list, optimizer, scheduler, args.eval_steps,
                           epoch, writer, args.log_interval, logger, args.mask_ratio,
                           args.cfar_weight, cfar_instance, cfar_criterion)
        if accelerator.is_main_process:
            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), os.path.join(checkpoint_dir, f'epoch{epoch}_trainloss{train_loss:.4f}.pth'))

    if accelerator.is_main_process:
        writer.close()
        logger.info("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DCDR parameters
    parser.add_argument('--dcdr_data_path', type=str)
    parser.add_argument('--dcdr_input_time', type=int, default=64)
    parser.add_argument('--dcdr_input_range', type=int, default=128)
    parser.add_argument('--dcdr_input_angle', type=int, default=128)

    # MCD parameters
    parser.add_argument('--mcd_data_path', type=str)
    parser.add_argument('--mcd_input_time', type=int, default=32)
    parser.add_argument('--mcd_input_range', type=int, default=128)
    parser.add_argument('--mcd_input_angle', type=int, default=32)
    parser.add_argument('--mcd_subset_indices_path', type=str, default=None)

    # RT-Pose parameters
    parser.add_argument('--rtpose_data_path', type=str)
    parser.add_argument('--rtpose_input_time', type=int, default=16)
    parser.add_argument('--rtpose_input_range', type=int, default=256)
    parser.add_argument('--rtpose_input_angle', type=int, default=16)
    parser.add_argument('--rtpose_subset_indices_path', type=str, default=None)

    # XRF55 parameters
    parser.add_argument('--xrf55_data_path', type=str)
    parser.add_argument('--xrf55_input_time', type=int, default=16)
    parser.add_argument('--xrf55_input_range', type=int, default=256)
    parser.add_argument('--xrf55_input_angle', type=int, default=64)
    parser.add_argument('--xrf55_subset_indices_path', type=str, default=None)

    # HuPR parameters
    parser.add_argument('--hupr_data_path', type=str)
    parser.add_argument('--hupr_input_time', type=int, default=8)
    parser.add_argument('--hupr_input_range', type=int, default=64)
    parser.add_argument('--hupr_input_angle', type=int, default=64)
    parser.add_argument('--hupr_subset_indices_path', type=str, default=None)

    # RADDet parameters
    parser.add_argument('--raddet_data_path', type=str)
    parser.add_argument('--raddet_input_time', type=int, default=16)
    parser.add_argument('--raddet_input_range', type=int, default=256)
    parser.add_argument('--raddet_input_angle', type=int, default=256)
    parser.add_argument('--raddet_subset_indices_path', type=str, default=None)

    # training data parameters
    parser.add_argument('--pretrain_datasets', type=lambda s: s.split(','), default=['dcdr', 'mcd', 'rtpose'])

    # model parameters
    parser.add_argument('--patch_size', type=lambda s: [int(x) for x in s.split(',')], default=[2, 16, 16])
    parser.add_argument('--patch_stride', type=lambda s: [int(x) for x in s.split(',')], default=[2, 16, 16])
    parser.add_argument('--model_size', type=str, default='tiny')
    parser.add_argument('--pipe_mask', type=bool, default=True, help='Whether to use pipe mask.')

    # CFAR parameters
    parser.add_argument('--cfar_weight', type=float, default=1.0, help='Weight for CFAR loss.')
    parser.add_argument('--cfar_type', type=str, default='2d', help='Type of CFAR loss.')
    parser.add_argument('--cfar_lambda_std', type=float, default=1.0, help='Lambda for CFAR loss.')
    parser.add_argument('--cfar_2d_win_param', type=tuple, default=(5,5,3,3), help='Window param for CFAR loss.')
    parser.add_argument('--cfar_threshold', type=float, default=5.0, help='Threshold for CFAR loss.')
    parser.add_argument('--cfar_temperature', type=float, default=10.0, help='Temperature for CFAR loss.')

    # training parameters
    parser.add_argument('--batch_sizes', type=list, default=[16,16,16])
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Number of steps for gradient accumulation.')
    parser.add_argument('--eval_steps', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for training.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Ratio of warmup steps for learning rate scheduler.')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='Ratio of the mask size to the CSI sequence length.')
    parser.add_argument('--save_interval', type=int, default=1, help='Interval for saving checkpoints.')
    parser.add_argument('--rope_theta_base', type=float, default=10000, help='Base theta for Rope.')
    parser.add_argument('--rope_use_concat', type=bool, default=False, help='Whether to use concat method for Rope.')
    parser.add_argument('--rope_use_add', type=bool, default=True, help='Whether to use add method for Rope.')
    parser.add_argument('--rope_divide_ratio', type=tuple, default=(0.25, 0.375, 0.375), help='Divide ratio for Rope.')
    parser.add_argument('--rope_learnable_freq', type=bool, default=True, help='Whether to use learnable freqs for Rope.')
    parser.add_argument('--rope_freq_cont', type=bool, default=False, help='Whether to use continuous freqs for Rope.')

    # log parameters
    parser.add_argument('--checkpoint_root_dir', type=str, default='pretrain_mask_mmwave_checkpoints', help='Path to the directory to save checkpoints.')
    parser.add_argument('--log_interval', type=int, default=100, help='Interval for logging training status.')
    parser.add_argument('--prefix_name', type=str, default=None, help='Prefix name for the checkpoint.')

    args = parser.parse_args()
    main(args)
