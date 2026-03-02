"""
ViT with dynamic input size and interpolated position embedding (no RoPE).
Same 3-stage flow as RopeViTModel. pos_embed is interpolated when input shape differs from max.
Supports 2D and 3D input.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from random import randrange
import random
from timm.layers.weight_init import trunc_normal_

from .yolo_head_tra import detection2d_head
from .vit_utils import Layer_scale_init_Block, Attention

class PatchEmbedNd(nn.Module):
    """ N-dimensional input to Patch Embedding"""
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.dim = len(patch_size)
        if self.dim == 2:
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif self.dim == 3:
            self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}. Only 2D and 3D are supported.")
    
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class InterpViTModel(nn.Module):
    """
    ViT with learned position embedding and dynamic input size.
    pos_embed is allocated for max_input_shape; for other sizes it is
    interpolated or center-cropped. No RoPE.
    """
    def __init__(self, patch_size, patch_stride, max_input_shape,
                 input_channels, model_size, 
                 train_stage, task_type=0, num_anchors=-1, input_shape=[16,256,256],
                 pretrained_ckpt_path=None, device=torch.device('cuda'), 
                 pipe_mask=False, 
                 label_dim=32, 
                 qkv_bias=False, qk_scale=None, proj_drop=0., attn_drop=0., path_drop=0., mlp_drop=0., 
                 resize_to_fixed=False):
        super().__init__()
        self.train_stage = train_stage
        self.task_type = task_type
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.max_input_shape = max_input_shape
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.pipe_mask = pipe_mask
        self._ndim = len(self.max_input_shape)
        if self._ndim not in (2, 3):
            raise ValueError("max_input_shape must be 2D (H,W) or 3D (D,H,W).")
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.path_drop = path_drop
        self.mlp_drop = mlp_drop

        # set model parameters based on size
        if model_size == 'tiny':
            self.embed_dim, self.depth, self.num_heads = 96, 6, 3
        elif model_size == 'small':
            self.embed_dim, self.depth, self.num_heads = 192, 6, 6
        elif model_size == 'base':
            self.embed_dim, self.depth, self.num_heads = 384, 6, 12
        elif model_size == 'large':
            self.embed_dim, self.depth, self.num_heads = 512, 12, 16
        else:
            raise Exception('Model size must be one of tiny, small, base, large')
        self.mlp_ratio = 4.0

        if train_stage == 0:
            # pre-training
            self._init_components()
            patch_volume = np.prod(patch_size)
            self.pred_layer = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, input_channels * patch_volume),
            )

            # Create appropriate unfold layer based on dimension
            if len(patch_size) == 2:
                self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_stride)
            elif len(patch_size) == 3:
                # For 3D, we need to handle unfolding manually since Unfold is 2D only
                self.unfold = None  # Will handle 3D unfolding in forward pass
            
            # we use learnable mask embedding (follow the BEIT paper)
            self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.embed_dim]))
            self.mask_embed = nn.init.xavier_normal_(self.mask_embed)
        
        elif train_stage == 1:
            # fine-tuning
            if pretrained_ckpt_path is None:
                raise ValueError('Please set pretrained_ckpt_path to load a pretrained models.')
            state_dict = torch.load(pretrained_ckpt_path, map_location=device)

            # Extract patch dimensions from pretrained model
            p_patch_size = list(state_dict['patch_embed.proj.weight'].shape[2:])
            if patch_size != p_patch_size:
                raise ValueError("patch_size should be equal to p_patch_size for fine-tuning")

            pretrained_model = InterpViTModel(patch_size=patch_size, patch_stride=patch_stride, 
                                            max_input_shape=max_input_shape, 
                                            input_channels=1, model_size=model_size, 
                                            train_stage=0)
            pretrained_model.load_state_dict(state_dict)

            # Copy components from pretrained model
            self._copy_from_pretrained(pretrained_model)
            self.max_mid_shape = pretrained_model.max_mid_shape
            self.max_num_patches = pretrained_model.max_num_patches

            # mlp head for fine-tuning
            if self.task_type == 0:
                self.mlp_head = nn.Sequential(
                    nn.LayerNorm(self.embed_dim), 
                    nn.Linear(self.embed_dim, label_dim)
                )
            elif self.task_type == 1:
                # get mid shape
                grid_shape = self._get_mid_shape(self.patch_embed.proj, [self.input_channels] + self.input_shape)
                grid_shape = list(grid_shape)
                grid_shape[0] = self.embed_dim * grid_shape[0]
                self.yolo_head = detection2d_head(num_anchors, label_dim, grid_shape)
            
            # adjust patch_embed.proj if stride is different
            if patch_stride != p_patch_size:
                # initialize a new patch embedding layer with desired new stride.
                if len(patch_size) == 2:
                    new_proj = nn.Conv2d(input_channels, self.embed_dim, 
                        kernel_size=patch_size, stride=patch_stride)
                elif len(patch_size) == 3:
                    new_proj = nn.Conv3d(input_channels, self.embed_dim, 
                        kernel_size=patch_size, stride=patch_stride)
                
                # but the weights of patch embedding layer is still got from the pretrained models
                new_proj.weight = nn.Parameter(pretrained_model.patch_embed.proj.weight)
                if pretrained_model.patch_embed.proj.bias is not None:
                    new_proj.bias = nn.Parameter(pretrained_model.patch_embed.proj.bias)
                else:
                    new_proj.bias = None
                self.patch_embed.proj = new_proj
        
        elif train_stage == 2:
            # train from scratch
            self._init_components()
            if self.task_type == 0:
                # mlp head for classification
                self.mlp_head = nn.Sequential(
                    nn.LayerNorm(self.embed_dim), 
                    nn.Linear(self.embed_dim, label_dim)
                )
            elif self.task_type == 1:
                # get mid shape
                grid_shape = self._get_mid_shape(self.patch_embed.proj, [self.input_channels] + max_input_shape)
                grid_shape = list(grid_shape)
                grid_shape[0] = self.embed_dim * grid_shape[0]
                self.yolo_head = detection2d_head(num_anchors, label_dim, grid_shape)

    def _init_components(self):
        # patch embedding
        self.patch_embed = PatchEmbedNd(self.patch_size, self.input_channels, self.embed_dim)
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        self.max_mid_shape = self._get_mid_shape(self.patch_embed.proj, [self.input_channels] + self.max_input_shape)
        self.max_num_patches = int(np.prod(self.max_mid_shape))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.max_num_patches, self.embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Layer_scale_init_Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, 
                proj_drop=self.proj_drop, attn_drop=self.attn_drop, path_drop=self.path_drop, mlp_drop=self.mlp_drop, 
                act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                Attention_block=Attention
            ) for _ in range(self.depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
    
    def _get_mid_shape(self, patch_proj, input_shape):
        test_input = torch.rand(1, *input_shape).to(patch_proj.weight.device)
        test_output = patch_proj(test_input)
        return test_output.shape[2:]
    
    def _copy_from_pretrained(self, pretrained_model):
        """Copy components from pretrained model"""
        if self.input_channels == 1:
            self.patch_embed = pretrained_model.patch_embed
        else:
            self.patch_embed = PatchEmbedNd(self.patch_size, self.input_channels, self.embed_dim)
        self.cls_token = pretrained_model.cls_token
        self.pos_embed = pretrained_model.pos_embed
        self.blocks = pretrained_model.blocks
        self.norm = pretrained_model.norm
    
    def interpolate_pos_embed(self, ori_shape, new_shape):
        ori_pos_embed = self.pos_embed[:, 1:, :]
        if self._ndim == 2:
            ori_h, ori_w = ori_shape[0], ori_shape[1]
            new_h, new_w = new_shape[0], new_shape[1]
            ori_pos_embed_2d = ori_pos_embed.transpose(1, 2).reshape(
                1, self.embed_dim, ori_h, ori_w
            )
            if new_h <= ori_h and new_w <= ori_w:
                start_h = (ori_h - new_h) // 2
                start_w = (ori_w - new_w) // 2
                new_pos_embed_2d = ori_pos_embed_2d[
                    :, :, start_h : start_h + new_h, start_w : start_w + new_w
                ]
            else:
                new_pos_embed_2d = F.interpolate(
                    ori_pos_embed_2d, size=(new_h, new_w),
                    mode="bilinear", align_corners=False,
                )
            new_pos_embed_flat = new_pos_embed_2d.reshape(1, self.embed_dim, -1).transpose(1, 2)
        else:
            ori_d, ori_h, ori_w = ori_shape[0], ori_shape[1], ori_shape[2]
            new_d, new_h, new_w = new_shape[0], new_shape[1], new_shape[2]
            ori_pos_embed_3d = ori_pos_embed.transpose(1, 2).reshape(
                1, self.embed_dim, ori_d, ori_h, ori_w
            )
            if new_d <= ori_d and new_h <= ori_h and new_w <= ori_w:
                start_d = (ori_d - new_d) // 2
                start_h = (ori_h - new_h) // 2
                start_w = (ori_w - new_w) // 2
                new_pos_embed_3d = ori_pos_embed_3d[
                    :, :,
                    start_d : start_d + new_d,
                    start_h : start_h + new_h,
                    start_w : start_w + new_w,
                ]
            else:
                new_pos_embed_3d = F.interpolate(
                    ori_pos_embed_3d,
                    size=(new_d, new_h, new_w),
                    mode="trilinear",
                    align_corners=False,
                )
            new_pos_embed_flat = new_pos_embed_3d.reshape(1, self.embed_dim, -1).transpose(1, 2)
        cls_token = self.pos_embed[:, :1, :]
        new_pos_embed = torch.cat([cls_token, new_pos_embed_flat], dim=1)
        return new_pos_embed
    
    def _cur_pos_embed(self, cur_mid_shape):
        if (
            len(cur_mid_shape) == len(self.max_mid_shape)
            and all(a == b for a, b in zip(cur_mid_shape, self.max_mid_shape))
        ):
            return self.pos_embed
        return self.interpolate_pos_embed(self.max_mid_shape, cur_mid_shape)
    
    def forward(self, x, cluster=3, mask_ratio=0.5):
        if self.train_stage == 0:
            return self.pretrain_mask(x, cluster, mask_ratio)
        else:
            if self.task_type == 0:
                return self.classify_avgtok(x)
            elif self.task_type == 1:
                return self.yolo_detect(x)
    
    def classify_avgtok(self, x):
        # input x: (bs, input_channels, *input_shape)
        B = x.shape[0]
        cur_mid_shape = self._get_mid_shape(self.patch_embed.proj, x.shape[1:])
        cur_pos_embed = self._cur_pos_embed(cur_mid_shape).to(x.dtype).to(x.device)

        x_patches = self.patch_embed(x)
        # add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x_patches], dim=1)
        x = x + cur_pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # average output of all tokens except cls token
        x = torch.mean(x[:, 1:, :], dim=1)  # Skip cls token
        x = self.mlp_head(x)
        return x
    
    def yolo_detect(self, x):
        # input x: (bs, input_channels, *input_shape)
        B = x.shape[0]
        cur_mid_shape = self._get_mid_shape(self.patch_embed.proj, x.shape[1:])
        cur_pos_embed = self._cur_pos_embed(cur_mid_shape).to(x.dtype).to(x.device)

        x_patches = self.patch_embed(x)
        # add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x_patches], dim=1)
        x = x + cur_pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        backbone_feature = x[:, 1:, :].permute(0, 2, 1)

        backbone_feature = backbone_feature.reshape(B, self.embed_dim, *cur_mid_shape)
        backbone_feature = backbone_feature.transpose(1, 2)
        backbone_feature = backbone_feature.reshape(B, cur_mid_shape[0]*self.embed_dim, *cur_mid_shape[1:])
        
        yolo_output = self.yolo_head(backbone_feature)
        return backbone_feature, yolo_output

    
    def gen_maskid_nonclustered(self, cur_num_patches, mask_patch_cnt):
        mask_id = random.sample(range(0, cur_num_patches), mask_patch_cnt)
        return torch.tensor(mask_id)
    
    def gen_maskid_clustered(self, cur_num_patches, grid_shape, cluster, mask_patch_cnt):
        mask_id = []
        # randomize clustering factor
        cur_clus = randrange(1, cluster+1)

        while len(list(set(mask_id))) <= mask_patch_cnt:
            start_id = randrange(cur_num_patches)
            cur_mask = []

            if len(grid_shape) == 2:
                for i in range(0, cur_clus):
                    for j in range(0, cur_clus):
                        mask_cand = start_id + grid_shape[1] * i + j
                        if mask_cand > 0 and mask_cand < cur_num_patches:
                            cur_mask.append(mask_cand)
            elif len(grid_shape) == 3:
                for i in range(0, cur_clus):
                    for j in range(0, cur_clus):
                        for k in range(0, cur_clus):
                            mask_cand = start_id + grid_shape[1] * grid_shape[2] * i + grid_shape[2] * j + k
                            if mask_cand > 0 and mask_cand < cur_num_patches:
                                cur_mask.append(mask_cand)
            
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_patch_cnt]
        return torch.tensor(mask_id)
    
    def gen_maskid_pipe(self, grid_shape, cluster, mask_patch_ratio):
        mask_pair_2d = []
        mask_patch_cnt_2d = int(grid_shape[1] * grid_shape[2] * mask_patch_ratio)
        patch_cnt_2d = grid_shape[1] * grid_shape[2]

        # randomize clustering factor
        cur_clus = randrange(1, cluster+1)

        while len(list(set(mask_pair_2d))) <= mask_patch_cnt_2d:
            start_i = randrange(grid_shape[1])
            start_j = randrange(grid_shape[2])
            cur_mask = []
            for i in range(0, cur_clus):
                for j in range(0, cur_clus):
                    # Calculate 2D coordinates
                    coord_i = start_i + i
                    coord_j = start_j + j
                    # Check if coordinates are within bounds
                    if coord_i < grid_shape[1] and coord_j < grid_shape[2]:
                        cur_mask.append((coord_i, coord_j))

            mask_pair_2d = mask_pair_2d + cur_mask
        mask_pair_2d = list(set(mask_pair_2d))[:mask_patch_cnt_2d]
        
        mask_id = []
        for i in range(len(mask_pair_2d)):
            for ti in range(grid_shape[0]):
                # Calculate 1D index from 2D coordinates
                cur_id = ti * patch_cnt_2d + \
                    mask_pair_2d[i][0] * grid_shape[2] + mask_pair_2d[i][1]
                mask_id.append(cur_id)
        
        return torch.tensor(mask_id)
    

    def pretrain_mask(self, x, cluster, mask_patch_ratio):
        B, C = x.shape[0], x.shape[1]
        grid_shape = self._get_mid_shape(self.patch_embed.proj, x.shape[1:])
        cur_pos_embed = self._cur_pos_embed(grid_shape).to(x.dtype).to(x.device)

        # Handle unfolding based on dimension
        if len(x.shape) == 4:
            unfold_x = self.unfold(x).transpose(1, 2)
        elif len(x.shape) == 5:
            unfold_x = x.view(B, C, grid_shape[0], self.patch_size[0], 
                            grid_shape[1], self.patch_size[1], grid_shape[2], self.patch_size[2])
            unfold_x = unfold_x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
            unfold_x = unfold_x.view(B, grid_shape[0] * grid_shape[1] * grid_shape[2], -1)
        
        x_patches = self.patch_embed(x)
        cur_num_patches = x_patches.shape[1]

        mask_patch_cnt = int(cur_num_patches * mask_patch_ratio)
        mask_index = torch.empty((B, mask_patch_cnt), device=x.device, requires_grad=False).long()
        # batch_size, sequence_len, hidden_size
        mask_dense = torch.ones([B, x_patches.shape[1], x_patches.shape[2]], device=x.device)
        for i in range(B):
            if not self.pipe_mask:
                if cluster > 1:
                    mask_index[i] = self.gen_maskid_clustered(cur_num_patches, grid_shape, cluster, mask_patch_cnt)
                else:
                    mask_index[i] = self.gen_maskid_nonclustered(cur_num_patches, mask_patch_cnt)
            else:
                mask_index[i] = self.gen_maskid_pipe(grid_shape, cluster, mask_patch_ratio)
            
            mask_dense[i, mask_index[i], :] = 0
        
        mask_tokens = self.mask_embed.expand(B, x_patches.shape[1], -1)
        x_masked = x_patches * mask_dense + (1-mask_dense) * mask_tokens
        
        # add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)
        x_masked = x_masked + cur_pos_embed

        for blk in self.blocks:
            x_masked = blk(x_masked)
        x_masked = self.norm(x_masked)

        patch_volume = np.prod(self.patch_size)
        pred = torch.empty((B, mask_patch_cnt, C*patch_volume), device=x.device).float()
        target = torch.empty((B, mask_patch_cnt, C*patch_volume), device=x.device).float()

        recovered_unfold_x = torch.zeros_like(unfold_x)
        for i in range(B):
            pred[i] = self.pred_layer(x_masked[i, mask_index[i] + 1, :])  # +1 for cls token
            target[i] = unfold_x[i, mask_index[i], :]
            recovered_unfold_x[i] = unfold_x[i]
            recovered_unfold_x[i, mask_index[i], :] = pred[i]
        
        # Inverse unfold operations to recover original x
        if len(x.shape) == 4:
            # For 4D: inverse of self.unfold(x).transpose(1, 2)
            # recovered_unfold_x shape: [B, grid_shape[0] * grid_shape[1], C * patch_size[0] * patch_size[1]]
            # Need to transpose back and fold
            recovered_unfold_x_transposed = recovered_unfold_x.transpose(1, 2)  # [B, C * patch_size[0] * patch_size[1], grid_shape[0] * grid_shape[1]]
            recovered_unfold_x_reshaped = recovered_unfold_x_transposed.view(B, C, self.patch_size[0], self.patch_size[1], grid_shape[0], grid_shape[1])
            recovered_unfold_x_reshaped = recovered_unfold_x_reshaped.permute(0, 1, 4, 2, 5, 3).contiguous()
            recovered_x = recovered_unfold_x_reshaped.view(B, C, grid_shape[0] * self.patch_size[0], grid_shape[1] * self.patch_size[1])
            
        elif len(x.shape) == 5:
            # For 5D: inverse of the manual 3D unfolding
            # recovered_unfold_x shape: [B, grid_shape[0] * grid_shape[1] * grid_shape[2], C * patch_size[0] * patch_size[1] * patch_size[2]]
            # Need to reverse the view and permute operations
            recovered_unfold_x_reshaped = recovered_unfold_x.view(B, grid_shape[0], grid_shape[1], grid_shape[2], C, self.patch_size[0], self.patch_size[1], self.patch_size[2])
            recovered_unfold_x_reshaped = recovered_unfold_x_reshaped.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
            recovered_x = recovered_unfold_x_reshaped.view(B, C, grid_shape[0] * self.patch_size[0], grid_shape[1] * self.patch_size[1], grid_shape[2] * self.patch_size[2])
        
        # calculate the MSE loss
        mse = torch.mean((pred - target) ** 2)
        
        return mse, mask_index, pred, target, unfold_x, recovered_x
