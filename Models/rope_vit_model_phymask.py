"""
RopeViT with patch-wise PhyMask for pre-training.

This module provides RopeViTModelPhyMask, a subclass of RopeViTModel that uses
adaptive patch masking (PatchPhyMask) instead of random/clustered/pipe masking
during pre-training. The original rope_vit_model.py is unchanged.
"""

import numpy as np
import torch

from .rope_vit_model import RopeViTModel
from .rope_vit_utils import init_grid_location
from .phymask_patch import PatchPhyMask


class RopeViTModelPhyMask(RopeViTModel):
    """
    RopeViT pre-training with patch-wise PhyMask (adaptive masking).

    Same interface as RopeViTModel; only pre-training (train_stage=0) behavior
    changes: masks are chosen by Energy and Coherence at the patch level instead
    of random or clustered selection.
    """

    def __init__(self, patch_size, patch_stride,
                 input_channels, model_size,
                 train_stage, task_type=0, num_anchors=-1, input_shape=[16, 256, 256],
                 pretrained_ckpt_path=None, device=torch.device('cuda'),
                 pipe_mask=False,
                 phymask_method='weighted', phymask_weights=(0.5, 0.5),
                 label_dim=32, rope_theta_base=100,
                 rope_use_concat=True, rope_use_add=False,
                 rope_divide_ratio=(0.25, 0.375, 0.375), rope_learnable_freq=False, rope_freq_cont=True,
                 qkv_bias=False, qk_scale=None, proj_drop=0., attn_drop=0., path_drop=0., mlp_drop=0.):
        super().__init__(
            patch_size=patch_size,
            patch_stride=patch_stride,
            input_channels=input_channels,
            model_size=model_size,
            train_stage=train_stage,
            task_type=task_type,
            num_anchors=num_anchors,
            input_shape=input_shape,
            pretrained_ckpt_path=pretrained_ckpt_path,
            device=device,
            pipe_mask=pipe_mask,
            label_dim=label_dim,
            rope_theta_base=rope_theta_base,
            rope_use_concat=rope_use_concat,
            rope_use_add=rope_use_add,
            rope_divide_ratio=rope_divide_ratio,
            rope_learnable_freq=rope_learnable_freq,
            rope_freq_cont=rope_freq_cont,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            path_drop=path_drop,
            mlp_drop=mlp_drop,
        )
        self.phymask_method = phymask_method
        self.phymask_weights = phymask_weights

    def pretrain_mask(self, x, cluster, mask_patch_ratio):
        """Pre-training forward with patch-wise PhyMask (adaptive masking)."""
        B = x.shape[0]
        C = x.shape[1]
        grid_shape = self._get_mid_shape(self.patch_embed.proj, x.shape[1:])

        if len(x.shape) == 4:
            unfold_x = self.unfold(x).transpose(1, 2)
        elif len(x.shape) == 5:
            unfold_x = x.view(
                B, C,
                grid_shape[0], self.patch_size[0],
                grid_shape[1], self.patch_size[1],
                grid_shape[2], self.patch_size[2],
            )
            unfold_x = unfold_x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
            unfold_x = unfold_x.view(B, grid_shape[0] * grid_shape[1] * grid_shape[2], -1)

        x_patches = self.patch_embed(x)
        cur_num_patches = x_patches.shape[1]

        # Patch-wise PhyMask: adaptive selection of patches to mask
        phymask = PatchPhyMask(
            patch_size=self.patch_size,
            mask_ratio=mask_patch_ratio,
            method=self.phymask_method,
            weights=self.phymask_weights,
            patch_stride=self.patch_stride,
        ).to(x.device)
        mask_index, mask_dense_1d, _, _ = phymask(x)
        mask_patch_cnt = mask_index.shape[1]
        mask_dense = mask_dense_1d.unsqueeze(2).expand(
            B, x_patches.shape[1], x_patches.shape[2]
        )

        mask_tokens = self.mask_embed.expand(B, x_patches.shape[1], -1)
        x_masked = x_patches * mask_dense + (1 - mask_dense) * mask_tokens

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)

        each_dim_loc = init_grid_location(grid_shape)

        if self.rope_use_concat:
            freqs_cis = self.compute_cis(freqs=self.freqs, each_dim_loc=each_dim_loc).to(x.device)
            for i, blk in enumerate(self.blocks):
                x_masked = blk(x_masked, freqs_cis=freqs_cis[i])
        elif self.rope_use_add:
            freqs_cis = self.compute_cis(freqs=self.freqs, each_dim_loc=each_dim_loc).to(x.device)
            for i, blk in enumerate(self.blocks):
                x_masked = blk(x_masked, freqs_cis=freqs_cis[i])

        x_masked = self.norm(x_masked)

        patch_volume = np.prod(self.patch_size)
        pred = torch.empty((B, mask_patch_cnt, C * patch_volume), device=x.device).float()
        target = torch.empty((B, mask_patch_cnt, C * patch_volume), device=x.device).float()

        recovered_unfold_x = torch.zeros_like(unfold_x)
        for i in range(B):
            pred[i] = self.pred_layer(x_masked[i, mask_index[i] + 1, :])
            target[i] = unfold_x[i, mask_index[i], :]
            recovered_unfold_x[i] = unfold_x[i]
            recovered_unfold_x[i, mask_index[i], :] = pred[i]

        if len(x.shape) == 4:
            recovered_unfold_x_transposed = recovered_unfold_x.transpose(1, 2)
            recovered_unfold_x_reshaped = recovered_unfold_x_transposed.view(
                B, C, self.patch_size[0], self.patch_size[1], grid_shape[0], grid_shape[1]
            )
            recovered_unfold_x_reshaped = recovered_unfold_x_reshaped.permute(0, 1, 4, 2, 5, 3).contiguous()
            recovered_x = recovered_unfold_x_reshaped.view(
                B, C,
                grid_shape[0] * self.patch_size[0],
                grid_shape[1] * self.patch_size[1],
            )
        elif len(x.shape) == 5:
            recovered_unfold_x_reshaped = recovered_unfold_x.view(
                B, grid_shape[0], grid_shape[1], grid_shape[2],
                C, self.patch_size[0], self.patch_size[1], self.patch_size[2],
            )
            recovered_unfold_x_reshaped = recovered_unfold_x_reshaped.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
            recovered_x = recovered_unfold_x_reshaped.view(
                B, C,
                grid_shape[0] * self.patch_size[0],
                grid_shape[1] * self.patch_size[1],
                grid_shape[2] * self.patch_size[2],
            )

        mse = torch.mean((pred - target) ** 2)
        return mse, mask_index, pred, target, unfold_x, recovered_x
