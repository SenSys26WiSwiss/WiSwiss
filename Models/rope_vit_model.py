import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from random import randrange
import random
from timm.layers.weight_init import trunc_normal_

from .rope_vit_utils import (
    RoPE_Layer_scale_init_Block, RoPEAttention, 
    init_grid_location, 
    concat_init_constant_cont_freqs, compute_concat_cis, 
    add_init_constant_freqs, add_init_learnable_freqs, compute_add_cis
)
from .yolo_head_tra import detection2d_head

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


class RopeViTModel(nn.Module):
    def __init__(self, patch_size, patch_stride, 
                 input_channels, model_size, 
                 train_stage, task_type=0, num_anchors=-1, input_shape=[16,256,256], # 0: classification, 1: detection
                 pretrained_ckpt_path=None, device=torch.device('cuda'), 
                 pipe_mask=False, 
                 label_dim=32, rope_theta_base=100, 
                 rope_use_concat=True, rope_use_add=False, 
                 rope_divide_ratio=(0.25, 0.375, 0.375), rope_learnable_freq=False, rope_freq_cont=True, 
                 qkv_bias=False, qk_scale=None, proj_drop=0., attn_drop=0., path_drop=0., mlp_drop=0.):
        super().__init__()
        self.train_stage = train_stage
        self.task_type = task_type
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.input_channels = input_channels
        self.pipe_mask = pipe_mask
        # rope parameters
        self.rope_theta_base = rope_theta_base
        self.rope_use_concat = rope_use_concat
        self.rope_use_add = rope_use_add
        self.rope_divide_ratio = rope_divide_ratio
        self.rope_learnable_freq = rope_learnable_freq
        self.rope_freq_cont = rope_freq_cont
        # attention parameters
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
            # initialize RoPE-ViT components
            self._init_rope_vit_components()

            # map the output of transformer to original patch size
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

            pretrained_model = RopeViTModel(patch_size=patch_size, patch_stride=patch_stride, 
                                            input_channels=1, model_size=model_size, 
                                            train_stage=0, 
                                            rope_theta_base=rope_theta_base, 
                                            rope_use_concat=rope_use_concat, rope_use_add=rope_use_add, 
                                            rope_divide_ratio=rope_divide_ratio, rope_learnable_freq=rope_learnable_freq, rope_freq_cont=rope_freq_cont)
            pretrained_model.load_state_dict(state_dict)

            # Copy components from pretrained model
            self._copy_from_pretrained(pretrained_model)
            if self.rope_use_add and self.rope_learnable_freq:
                self.freqs = nn.Parameter(self.freqs.clone(), requires_grad=True)

            # mlp head for fine-tuning
            if self.task_type == 0:
                self.mlp_head = nn.Sequential(
                    nn.LayerNorm(self.embed_dim), 
                    nn.Linear(self.embed_dim, label_dim)
                )
            elif self.task_type == 1:
                # get mid shape
                grid_shape = self._get_mid_shape(self.patch_embed.proj, [1] + input_shape)
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
            # initialize RoPE-ViT components
            self._init_rope_vit_components()
            if self.task_type == 0:
                # mlp head for classification
                self.mlp_head = nn.Sequential(
                    nn.LayerNorm(self.embed_dim), 
                    nn.Linear(self.embed_dim, label_dim)
                )
            elif self.task_type == 1:
                # get mid shape
                grid_shape = self._get_mid_shape(self.patch_embed.proj, [1] + input_shape)
                grid_shape = list(grid_shape)
                grid_shape[0] = self.embed_dim * grid_shape[0]
                self.yolo_head = detection2d_head(num_anchors, label_dim, grid_shape)


    def _init_rope_vit_components(self):
        # patch embedding
        self.patch_embed = PatchEmbedNd(self.patch_size, self.input_channels, self.embed_dim)
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            RoPE_Layer_scale_init_Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, 
                proj_drop=self.proj_drop, attn_drop=self.attn_drop, path_drop=self.path_drop, mlp_drop=self.mlp_drop, 
                act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                Attention_block=RoPEAttention
            ) for _ in range(self.depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        # Initialize RoPE frequencies
        if self.rope_use_concat:
            self.compute_cis = partial(compute_concat_cis, num_heads=self.num_heads)
            if not self.rope_learnable_freq:
                if self.rope_freq_cont:
                    freqs = []
                    for i in range(self.depth):
                        cur_freqs = concat_init_constant_cont_freqs(self.rope_divide_ratio, self.embed_dim, self.num_heads, self.rope_theta_base)
                        for ni in range(len(cur_freqs)):
                            cur_freqs[ni] = cur_freqs[ni].flatten(0)
                            if i == 0:
                                freqs.append(cur_freqs[ni].unsqueeze(0))
                            else:
                                freqs[ni] = torch.cat([freqs[ni], cur_freqs[ni].unsqueeze(0)], dim=0)
                    self.freqs = freqs
        
        elif self.rope_use_add:
            self.compute_cis = partial(compute_add_cis, num_heads=self.num_heads)
            if self.rope_learnable_freq:
                freqs = []
                for i in range(self.depth):
                    cur_freqs = add_init_learnable_freqs(len(self.rope_divide_ratio), self.embed_dim, self.num_heads, self.rope_theta_base)
                    freqs.append(cur_freqs.flatten(1))
                freqs = torch.stack(freqs, dim=1)
                self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
    

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
        self.blocks = pretrained_model.blocks
        self.norm = pretrained_model.norm
        self.freqs = pretrained_model.freqs
        self.compute_cis = pretrained_model.compute_cis
    

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
        # get mid shape
        grid_shape = self._get_mid_shape(self.patch_embed.proj, x.shape[1:])

        x_patches = self.patch_embed(x)
        # add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x_patches], dim=1)

        # get each_dim_loc
        each_dim_loc = init_grid_location(grid_shape)

        # get freqs
        if self.rope_use_concat:
            freqs_cis = self.compute_cis(freqs=self.freqs, each_dim_loc=each_dim_loc).to(x.device)
            for i, blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis[i])
        elif self.rope_use_add:
            freqs_cis = self.compute_cis(freqs=self.freqs, each_dim_loc=each_dim_loc).to(x.device)
            for i, blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis[i])
        
        x = self.norm(x)

        # average output of all tokens except cls token
        x = torch.mean(x[:, 1:, :], dim=1)  # Skip cls token
        x = self.mlp_head(x)
        return x

    def yolo_detect(self, x):
        # input x: (bs, input_channels, *input_shape)
        B = x.shape[0]
        # get mid shape
        grid_shape = self._get_mid_shape(self.patch_embed.proj, x.shape[1:])

        x_patches = self.patch_embed(x)
        # add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x_patches], dim=1)

        # get each_dim_loc
        each_dim_loc = init_grid_location(grid_shape)

        # get freqs
        if self.rope_use_concat:
            freqs_cis = self.compute_cis(freqs=self.freqs, each_dim_loc=each_dim_loc).to(x.device)
            for i, blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis[i])
        elif self.rope_use_add:
            freqs_cis = self.compute_cis(freqs=self.freqs, each_dim_loc=each_dim_loc).to(x.device)
            for i, blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis[i])
        
        backbone_feature = self.norm(x)[:, 1:, :].permute(0, 2, 1)

        backbone_feature = backbone_feature.reshape(B, self.embed_dim, *grid_shape)
        backbone_feature = backbone_feature.transpose(1, 2)
        backbone_feature = backbone_feature.reshape(B, grid_shape[0]*self.embed_dim, *grid_shape[1:])
        
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
        B = x.shape[0]
        C = x.shape[1]
        grid_shape = self._get_mid_shape(self.patch_embed.proj, x.shape[1:])

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

        # get each_dim_loc
        each_dim_loc = init_grid_location(grid_shape)

        # get freqs
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
            patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
            recovered_unfold_x_reshaped = recovered_unfold_x.view(B, grid_shape[0], grid_shape[1], grid_shape[2], C, self.patch_size[0], self.patch_size[1], self.patch_size[2])
            recovered_unfold_x_reshaped = recovered_unfold_x_reshaped.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
            recovered_x = recovered_unfold_x_reshaped.view(B, C, grid_shape[0] * self.patch_size[0], grid_shape[1] * self.patch_size[1], grid_shape[2] * self.patch_size[2])
        
        # calculate the MSE loss
        mse = torch.mean((pred - target) ** 2)
        
        return mse, mask_index, pred, target, unfold_x, recovered_x

