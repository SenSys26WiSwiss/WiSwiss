import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import argparse
from transformers import get_cosine_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_

from accelerate import Accelerator

from Models.rope_vit_model import RopeViTModel
from Models.yolo_head_tra import decode_yolo_2d, yol2predictions_2d, nms_2d
from Models.yolo_loss_tra import RADDetLoss_2d
from Datasets.hdf5 import HDF5Dataset_RADDet
from Datasets.subset_dataset import SubsetDataset
from Metrics.mAP import mAP_2d
from Datasets.raddet_utils import read_anchors
from utils import get_current_time, AverageMeter

CURRENT_TIME: str = get_current_time()

os.environ["OMP_NUM_THREADS"] = "2"         # OpenMP (NumPy, SciPy)
os.environ["MKL_NUM_THREADS"] = "2"         # Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = "2"    # OpenBLAS
torch.set_num_threads(2)


def train(accelerator, model, train_loader, optimizer, scheduler, epoch, writer, log_interval, logger, 
         patch_size, anchors, yolohead_xy_scale, num_classes, criterion):
    model.train()
    epoch_box_loss = AverageMeter()
    epoch_conf_loss = AverageMeter()
    epoch_category_loss = AverageMeter()
    epoch_total_loss = AverageMeter()
    
    progress_bar = tqdm(
        total=len(train_loader), 
        desc=f'Train round{epoch}/{args.epochs}', 
        unit='batch',
        disable=not accelerator.is_main_process
    )
    
    device = accelerator.device
    
    for (data, target, raw_boxes) in train_loader:
        data, target, raw_boxes = data.to(device), target.to(device), raw_boxes.to(device)
        data = data.unsqueeze(1)
        cur_bs = data.shape[0]
        _, yolo_output = model(data)
        pred_raw, pred = decode_yolo_2d(yolo_output, patch_size, anchors, yolohead_xy_scale, num_classes)
        box_loss, conf_loss, category_loss = criterion(pred_raw, pred, target, raw_boxes[..., :4])
        total_loss = box_loss + conf_loss + category_loss
        
        with accelerator.accumulate(model):
            accelerator.backward(total_loss)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if accelerator.sync_gradients:
            avg_total_loss = accelerator.gather(total_loss).mean()
            avg_box_loss = accelerator.gather(box_loss).mean()
            avg_conf_loss = accelerator.gather(conf_loss).mean()
            avg_category_loss = accelerator.gather(category_loss).mean()
            
            epoch_total_loss.update(avg_total_loss.item(), cur_bs * accelerator.num_processes * accelerator.gradient_accumulation_steps)
            epoch_box_loss.update(avg_box_loss.item(), cur_bs * accelerator.num_processes * accelerator.gradient_accumulation_steps)
            epoch_conf_loss.update(avg_conf_loss.item(), cur_bs * accelerator.num_processes * accelerator.gradient_accumulation_steps)
            epoch_category_loss.update(avg_category_loss.item(), cur_bs * accelerator.num_processes * accelerator.gradient_accumulation_steps)
            
            cur_lr = scheduler.get_last_lr()
            progress_bar.set_postfix(**{'total_loss': avg_total_loss.item(), 
                    'box_loss': avg_box_loss.item(), 'conf_loss': avg_conf_loss.item(), 'category_loss': avg_category_loss.item(), 
                    'lr': cur_lr[0]})
        
        progress_bar.update(1)
        
        if accelerator.is_main_process and progress_bar.n % log_interval == 0:
            if accelerator.sync_gradients:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] total_loss: {:.6f} box_loss: {:.6f} conf_loss: {:.6f} category_loss: {:.6f} LR: {}'.format(
                    epoch, progress_bar.n, len(train_loader), 100. * progress_bar.n / len(train_loader), 
                    avg_total_loss.item(), avg_box_loss.item(), avg_conf_loss.item(), avg_category_loss.item(), ', '.join(['{:.6f}'.format(x) for x in cur_lr])))
                writer.add_scalar('train/total_loss', avg_total_loss.item(), epoch*len(train_loader) + progress_bar.n)
                writer.add_scalar('train/box_loss', avg_box_loss.item(), epoch*len(train_loader) + progress_bar.n)
                writer.add_scalar('train/conf_loss', avg_conf_loss.item(), epoch*len(train_loader) + progress_bar.n)
                writer.add_scalar('train/category_loss', avg_category_loss.item(), epoch*len(train_loader) + progress_bar.n)
            
                for i in range(len(cur_lr)):
                    writer.add_scalar(f'Train/lr_group{i}', cur_lr[i], epoch * len(train_loader) + progress_bar.n)
    
    if accelerator.is_main_process:
        logger.info(f"Epoch {epoch} total_loss: {epoch_total_loss.avg:.4f}, box_loss: {epoch_box_loss.avg:.4f}, conf_loss: {epoch_conf_loss.avg:.4f}, category_loss: {epoch_category_loss.avg:.4f}")
        writer.add_scalar('train/epoch_total_loss', epoch_total_loss.avg, epoch)
        writer.add_scalar('train/epoch_box_loss', epoch_box_loss.avg, epoch)
        writer.add_scalar('train/epoch_conf_loss', epoch_conf_loss.avg, epoch)
        writer.add_scalar('train/epoch_category_loss', epoch_category_loss.avg, epoch)

    return epoch_total_loss.avg


def test(accelerator, model, test_loader, epoch, writer, logger, 
         patch_size, anchors, yolohead_xy_scale, num_classes, criterion, input_shape, 
         confidence_threshold, nms_iou3d_threshold, mAP_iou_threshold):
    model.eval()
    test_box_loss = AverageMeter()
    test_conf_loss = AverageMeter()
    test_category_loss = AverageMeter()
    test_total_loss = AverageMeter()
    test_mAP = AverageMeter()
    ap_all_class_test = []
    ap_all_class = []
    for class_id in range(num_classes):
        ap_all_class.append([])

    with torch.no_grad():
        for data, target, raw_boxes in tqdm(test_loader, disable=not accelerator.is_main_process):
            data, target, raw_boxes = data.to(accelerator.device), target.to(accelerator.device), raw_boxes.to(accelerator.device)
            data = data.unsqueeze(1)
            cur_bs = data.shape[0]
            _, yolo_output = model(data)
            pred_raw, pred = decode_yolo_2d(yolo_output, patch_size, anchors, yolohead_xy_scale, num_classes)
            box_loss, conf_loss, category_loss = criterion(pred_raw, pred, target, raw_boxes[..., :4])
            test_box_loss.update(box_loss.item(), cur_bs)
            test_conf_loss.update(conf_loss.item(), cur_bs)
            test_category_loss.update(category_loss.item(), cur_bs)
            test_total_loss.update(box_loss.item() + conf_loss.item() + category_loss.item(), cur_bs)

            for sample_id in range(cur_bs):
                raw_boxes_sample = raw_boxes[sample_id]
                pred_sample = pred[sample_id]
                predictions = yol2predictions_2d(pred_sample, confidence_threshold)
                nms_pred = nms_2d(predictions, nms_iou3d_threshold, sigma=0.3, method="nms")
                mean_ap, ap_all_class = mAP_2d(nms_pred, raw_boxes_sample, 
                                            input_shape, ap_all_class, mAP_iou_threshold)
                test_mAP.update(mean_ap, 1)
    
    # Gather metrics from all processes
    test_total_loss_avg = accelerator.gather(torch.tensor(test_total_loss.avg, device=accelerator.device)).mean()
    test_box_loss_avg = accelerator.gather(torch.tensor(test_box_loss.avg, device=accelerator.device)).mean()
    test_conf_loss_avg = accelerator.gather(torch.tensor(test_conf_loss.avg, device=accelerator.device)).mean()
    test_category_loss_avg = accelerator.gather(torch.tensor(test_category_loss.avg, device=accelerator.device)).mean()
    test_mAP_avg = accelerator.gather(torch.tensor(test_mAP.avg, device=accelerator.device)).mean()
    
    for ap_class_i in ap_all_class:
        if len(ap_class_i) == 0:
            class_ap = 0.
        else:
            class_ap = torch.mean(torch.tensor(ap_class_i)).item()
        ap_all_class_test.append(class_ap)
    
    if accelerator.is_main_process:
        logger.info(f'mAP_iou_threshold: {mAP_iou_threshold}')
        logger.info(f"Epoch {epoch} test_total_loss: {test_total_loss_avg.item():.4f}, test_box_loss: {test_box_loss_avg.item():.4f}, test_conf_loss: {test_conf_loss_avg.item():.4f}, test_category_loss: {test_category_loss_avg.item():.4f}, test_mAP: {test_mAP_avg.item():.4f}")
        logger.info(f"ap_all: {test_mAP_avg.item():.4f}, ap_person: {ap_all_class_test[0]:.4f}, ap_bicycle: {ap_all_class_test[1]:.4f}, ap_car: {ap_all_class_test[2]:.4f}, ap_motorcycle: {ap_all_class_test[3]:.4f}, ap_bus: {ap_all_class_test[4]:.4f}, ap_truck: {ap_all_class_test[5]:.4f}")
        writer.add_scalar('test/total_loss', test_total_loss_avg.item(), epoch)
        writer.add_scalar('test/box_loss', test_box_loss_avg.item(), epoch)
        writer.add_scalar('test/conf_loss', test_conf_loss_avg.item(), epoch)
        writer.add_scalar('test/category_loss', test_category_loss_avg.item(), epoch)
        writer.add_scalar('test/mAP', test_mAP_avg.item(), epoch)
        writer.add_scalar('test/AP_person', ap_all_class_test[0], epoch)
        writer.add_scalar('test/AP_bicycle', ap_all_class_test[1], epoch)
        writer.add_scalar('test/AP_car', ap_all_class_test[2], epoch)
        writer.add_scalar('test/AP_motorcycle', ap_all_class_test[3], epoch)
        writer.add_scalar('test/AP_bus', ap_all_class_test[4], epoch)
        writer.add_scalar('test/AP_truck', ap_all_class_test[5], epoch)

    return test_mAP_avg.item()


def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device

    writer = None
    logger = logging.getLogger(__name__)
    checkpoint_dir = ""

    # set up tensorboard writer
    if accelerator.is_main_process:
        patch_name = f'patch{"x".join([str(x) for x in args.patch_size])}'
        shape_name = f'{args.input_time}x{args.input_range}x{args.input_angle}'
        lr_name = f'lr{args.lr}head{args.head_lr}x'
        rope_name = 'concat' if args.rope_use_concat else 'add'
        rope_name = rope_name + ('_learnable' if args.rope_learnable_freq else '_const')
        if args.rope_use_concat:
            rope_name = rope_name + ('_cont' if args.rope_freq_cont else '_sep')
        iou_name = f'mAPiou{args.mAP_iou3d_threshold}'
        checkpoint_dir = os.path.join(args.checkpoint_root_dir, \
            f'{args.prefix_name}_{args.model_size}_{iou_name}_{rope_name}_{patch_name}_{shape_name}_{lr_name}_{CURRENT_TIME}')
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
    input_shape = [args.input_time, args.input_range, args.input_angle]
    anchors = read_anchors(args.anchors_fname)
    ori_train_dataset = HDF5Dataset_RADDet(args.train_hdf5_path)
    ori_test_dataset = HDF5Dataset_RADDet(args.test_hdf5_path)
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
    model = RopeViTModel(patch_size=args.patch_size, patch_stride=args.patch_stride, 
                         input_channels=1, model_size=args.model_size, 
                         train_stage=1, task_type=1, num_anchors=len(anchors), input_shape=input_shape,
                         pretrained_ckpt_path=args.model_ckpt_path, device=device, label_dim=len(args.all_classes), 
                         rope_use_concat=args.rope_use_concat, rope_use_add=args.rope_use_add, 
                         rope_divide_ratio=args.rope_divide_ratio, rope_learnable_freq=args.rope_learnable_freq, rope_freq_cont=args.rope_freq_cont, 
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

    criterion = RADDetLoss_2d(input_shape, args.focal_loss_iou_threshold)

    model, optimizer, scheduler, train_loader, test_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, test_loader)
    
    if accelerator.is_main_process:
        logger.info(f"Model: {args.model_size}, Total optimizer steps: {num_training_steps}")

    best_test_mAP = 0
    best_epoch = 0
    best_state_dict = None
    for epoch in range(args.epochs):
        train_loss = train(accelerator, model, train_loader, optimizer, scheduler, epoch, writer, args.log_interval, logger, \
            args.patch_size, anchors, args.yolohead_xy_scale, len(args.all_classes), criterion)
        if ((epoch+1) % args.eval_epochs == 0 or epoch == args.epochs-1) and (epoch+1) >= args.eval_start_epoch:
            test_mAP = test(accelerator, model, test_loader, epoch, writer, logger, \
                args.patch_size, anchors, args.yolohead_xy_scale, len(args.all_classes), criterion, input_shape, \
                args.confidence_threshold, args.nms_iou3d_threshold, args.mAP_iou3d_threshold)
            if test_mAP > best_test_mAP:
                best_test_mAP = test_mAP
                best_epoch = epoch
                best_state_dict = accelerator.unwrap_model(model).state_dict()
            
            if train_loss < args.es_train_loss:
                break
    
    if accelerator.is_main_process:
        torch.save(best_state_dict, os.path.join(checkpoint_dir, f"epoch{best_epoch}_testmAP{best_test_mAP:.4f}.pth"))
        logger.info(f"Best Test mAP: {best_test_mAP:.4f}")
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--train_hdf5_path', type=str)
    parser.add_argument('--test_hdf5_path', type=str)
    parser.add_argument('--number_joints', type=int, default=14, help='Number of joints to use.')
    parser.add_argument('--joint_dim', type=int, default=2, help='Dimension of each joint.')
    parser.add_argument('--input_time', type=int, default=16)
    parser.add_argument('--input_range', type=int, default=256)
    parser.add_argument('--input_angle', type=int, default=256)
    parser.add_argument('--subset_indices_path', type=str, default=None)
    parser.add_argument('--test_subset_indices_path', type=str, default=None)
    parser.add_argument('--task_name', type=str, default=None)

    # yolo parameters
    parser.add_argument('--yolohead_xy_scale', type=float, default=1.0)
    parser.add_argument('--focal_loss_iou_threshold', type=float, default=0.3)
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    parser.add_argument('--nms_iou3d_threshold', type=float, default=0.1)
    parser.add_argument('--mAP_iou3d_threshold', type=float, default=0.3)
    parser.add_argument('--max_boxes_per_frame', type=int, default=30)
    parser.add_argument('--anchors_fname', type=str, default='Datasets/anchors_2d.txt')
    parser.add_argument('--all_classes', type=list, default=['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'])

    # model parameters
    parser.add_argument('--patch_size', type=lambda s: [int(x) for x in s.split(',')], default=[2, 16, 16])
    parser.add_argument('--patch_stride', type=lambda s: [int(x) for x in s.split(',')], default=[2, 16, 16])
    parser.add_argument('--model_size', type=str, default='tiny')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Number of steps for gradient accumulation.')
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
    parser.add_argument('--es_train_loss', type=float, default=5.0, help='Early stopping training loss.')
    parser.add_argument('--eval_start_epoch', type=int, default=30, help='Start evaluating the model on test set after this epoch.')

    # log parameters
    parser.add_argument('--checkpoint_root_dir', type=str, default='finetune_raddet_checkpoints', help='Path to the directory to save checkpoints.')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging training status.')
    parser.add_argument('--eval_epochs', type=int, default=1, help='Interval for evaluating the model on test set.')
    
    args = parser.parse_args()
    
    main(args)
