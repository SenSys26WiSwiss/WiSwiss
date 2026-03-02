# WiSwiss

This repository provides the official implementation of **WiSwiss**, a systematic framework for self-supervised multi-source pre-training designed to learn general-purpose backbones for wireless sensing. It features a holistic data-to-model pipeline that addresses task heterogeneity, data redundancy, and structural incompatibility. The codebase supports both WiFi CSI and mmWave radar inputs, multiple backbone variants (e.g., RoPE-ViT, PhyMask, interpolation and fixed-shape handling), and reproducible experiments from data curation through downstream evaluation.

## Overview

The pipeline has four main stages:

1. **Data deduplication**  
   Train a contrastive encoder and run greedy deduplication on raw HDF5 datasets to obtain cleaner pre-training sets.

2. **Pre-training**  
   Pre-train a RoPE-ViT backbone on radar/WiFi data with transformation-invariant reconstruction loss.

3. **Fine-tuning**  
   Fine-tune the backbone on downstream tasks: classification, pose estimation, or object detection.

4. **Baselines**  
   Train from scratch (no pre-training) for comparison.

All scripts are intended to be run from the project root (`WiSwiss/`). Example:

```bash
cd WiSwiss
python scripts/dedup/train_dedup_encoder_mmwave.py --data_path /path/to/train.hdf5 ...
```

If you run from another directory, set the project root in `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/WiSwiss python /path/to/WiSwiss/scripts/dedup/train_dedup_encoder_mmwave.py ...
```



## Environment

- Python 3.8+
- PyTorch (with CUDA if using GPU)
- [Hugging Face Accelerate](https://github.com/huggingface/accelerate) for multi-GPU training (pre-training / fine-tuning)

Create a virtual environment and install dependencies:

```bash
cd WiSwiss
pip install -r requirements.txt
# For multi-GPU: pip install accelerate && accelerate config
```


## Data Preparation

- Input data is assumed to be in HDF5 format with at least:
  - `data`: (frequency, time) for WiFi and (time, range, angle) for mmWave. 
  - `label`: per-sample labels (classification) or task-specific format (pose/detection).

- Deduplication outputs new HDF5 files with the same structure but a subset of samples (duplicates removed).

- Downstream datasets should be converted to this HDF5 layout. 

Due to redistribution restrictions outlined in the original access protocols for certain datasets (e.g., MM-DCDR and MCD-Gesture ), we provide direct download links to the official sources below: 
- For WiFi: [SignFi](https://yongsen.github.io/SignFi/), [Widar](https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset), [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi), [XRF55](https://aiotgroup.github.io/XRF55/), [CSI-Bench](https://ai-iot-sensing.github.io/projects/project.html), [WiPose](https://github.com/NjtechCVLab/Wi-PoseDataset). 
- For mmWave: [MM-DCDR](https://github.com/yating-gao/MM-DCDR), [MCD](https://github.com/leeyadong/cross_domain_gesture_dataset/tree/main), [RT-Pose](https://huggingface.co/datasets/uwipl/RT-Pose), [XRF55](https://aiotgroup.github.io/XRF55/), [RADDet](https://github.com/ZhangAoCanada/RADDet), [HuPR](https://huggingface.co/datasets/nirajpkini/HuPR). 



## Pipeline

### 1. Data deduplication

**Step 1a: Train the deduplication encoder**

Contrastive training with Gaussian augmentation; encoder is used to compute embeddings for dedup.

- WiFi:

  ```bash
  python scripts/dedup/train_dedup_encoder_wifi.py \
    --data_path /path/to/signfi_train.hdf5 \
    --checkpoint_root_dir ./dedup_encoder_checkpoints \
    --embedding_dim 128 --gaussian_noise_std 0.5 \
    --epochs 10
  ```

- mmWave:

  ```bash
  python scripts/dedup/train_dedup_encoder_mmwave.py \
    --data_path /path/to/mcd_train.hdf5 \
    --checkpoint_root_dir ./dedup_encoder_checkpoints \
    --embedding_dim 128 --gaussian_noise_std 0.5 \
    --epochs 10
  ```

**Step 1b: Run deduplication**

Uses the trained encoder + FAISS to find near-duplicates and saves a new HDF5 with unique samples.

- WiFi:

  ```bash
  python scripts/dedup/run_dedup_wifi.py \
    --data_path /path/to/pretrain_data.hdf5 \
    --checkpoint_path ./dedup_encoder_checkpoints/.../model_epoch10.pth \
    --save_dir ./dedup_datasets \
    --aug_percentile_list 10
  ```

- mmWave:

  ```bash
  python scripts/dedup/run_dedup_mmwave.py \
    --data_path /path/to/pretrain_data.hdf5 \
    --checkpoint_path ./dedup_encoder_checkpoints/.../model_epoch10.pth \
    --save_dir ./dedup_datasets \
    --aug_percentile_list 10
  ```

Typical hyperparameters:

- **Gaussian variance** (`--gaussian_noise_std`): default 0.5; sensitivity studies use e.g. 0.01, 1.0, 2.0.
- **Distribution percentile** (`--aug_percentile_list`): default 10; higher keeps more samples (less aggressive dedup).
- **CFAR threshold** (mmWave): used in pre-training; default 5; sensitivity studies use e.g. 2, 10, 20.



### 2. Pre-training

Pre-train the RoPE-ViT backbone on (optionally deduplicated) HDF5 data. Multiple dataset names can be mixed (e.g. `--pretrain_datasets dcdr,mcd`).

- WiFi:

  ```bash
  python scripts/pretrain/pretrain_backbone_wifi.py \
    --pretrain_datasets signfi \
    --signfi_data_path /path/to/signfi_train.hdf5 \
    --checkpoint_root_dir ./pretrain_checkpoints \
    --model_size tiny
  ```

- mmWave:

  ```bash
  python scripts/pretrain/pretrain_backbone_mmwave.py \
    --pretrain_datasets mcd \
    --mcd_data_path /path/to/mcd_dedup_train.hdf5 \
    --checkpoint_root_dir ./pretrain_checkpoints \
    --model_size tiny
  ```

Multi-GPU (recommended for pre-training):

```bash
accelerate launch scripts/pretrain/pretrain_backbone_mmwave.py ...
# or
accelerate launch scripts/pretrain/pretrain_backbone_mmwave_multigpu.py ...
```

**Variants** (see `scripts/pretrain/`):

- `pretrain_backbone_*_phymask.py`: implementation of the PhyMask baseline.
- `pretrain_backbone_*_interp.py`: interpolation-based position embedding.
- `pretrain_backbone_*_fixedshape.py`: fixed-shape handling.

Pre-training outputs a checkpoint (e.g. `*.pth`) to load in fine-tuning via `--model_ckpt_path`.


### 3. Fine-tuning

Load the pre-trained backbone and add a task head (classification MLP, pose head, or detection head). Train on downstream HDF5 datasets.

**Classification (e.g. XRF55, SignFi)**

- WiFi:

  ```bash
  python scripts/finetune/finetune_cls_wifi.py \
    --train_hdf5_path /path/to/train.hdf5 \
    --test_hdf5_path /path/to/test.hdf5 \
    --num_classes <N> \
    --model_ckpt_path ./pretrain_checkpoints/.../xxx.pth \
    --checkpoint_root_dir ./finetune_checkpoints
  ```

- mmWave:

  ```bash
  python scripts/finetune/finetune_cls_mmwave.py \
    --train_hdf5_path /path/to/train.hdf5 \
    --test_hdf5_path /path/to/test.hdf5 \
    --num_classes <N> \
    --model_ckpt_path ./pretrain_checkpoints/.../xxx.pth \
    --checkpoint_root_dir ./finetune_checkpoints
  ```

**Pose estimation**

- `scripts/finetune/finetune_pose_wifi.py`, `finetune_pose_mmwave.py`: same pattern as above with task-specific data and labels.

**Detection (e.g. RADDet)**

- `scripts/finetune/finetune_detection_mmwave.py`: detection head + mAP evaluation; requires detection HDF5 format (see `Datasets/hdf5.py` and `Datasets/raddet_utils.py`).

**Shape variants**

- `*_interp.py`, `*_fixedshape.py`: same fine-tuning with interpolation or fixed-shape backbone/head.

All fine-tuning scripts support `--subset_indices_path` / `--test_subset_indices_path` for using a subset of the dataset (e.g. few-shot).


### 4. Baselines (train from scratch)

No pre-training; same architecture and tasks for comparison.

- Classification: `scripts/baselines/train_scratch_cls_wifi.py`, `train_scratch_cls_mmwave.py`
- Pose: `scripts/baselines/train_scratch_pose_wifi.py`, `train_scratch_pose_mmwave.py`


## Project layout
```
WiSwiss/
├── README.md
├── requirements.txt
├── .gitignore
├── utils.py
├── Datasets/           # HDF5 dataset, subset, reshape, RADDet utils (not lowercase "datasets")
├── Dedup_utils/        # Encoder, contrastive dataset, InfoNCE, embedding distance
├── Models/             # RoPE-ViT, PhyMask, interpolation/fixed-shape variants, YOLO head/loss
├── Transform_utils/    # CFAR, CSI transforms
├── Metrics/            # mAP for detection
├── data_scripts/       # Data preparation and conversion scripts
└── scripts/
    ├── dedup/          # Train dedup encoder (WiFi/mmWave), run deduplication
    ├── pretrain/       # Pre-train backbone (base, phymask, interp, fixedshape, multigpu)
    ├── finetune/       # Fine-tune classification, pose, detection (WiFi/mmWave + variants)
    └── baselines/      # Train from scratch (cls, pose)
```
