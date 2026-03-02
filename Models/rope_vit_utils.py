import torch
import torch.nn as nn
from typing import Tuple

from .vit_utils import Layer_scale_init_Block, Attention

##### Add the per-dimension
def add_init_constant_freqs(ndim, embed_dim, num_heads, theta_base=10000):
    """ per-layer freqs shape: (ndim, num_heads, head_dim//2) """
    head_dim = embed_dim // num_heads
    freqs = torch.zeros(ndim, num_heads, head_dim//2)
    constant_freqs = 1 / (theta_base ** (torch.arange(0,head_dim,2)[:(head_dim//2)].float() / head_dim))
    freqs[..., :] = constant_freqs.unsqueeze(0).unsqueeze(0)
    return freqs

def add_init_learnable_freqs(ndim, embed_dim, num_heads, theta_base=10000):
    """ per-layer freqs shape: (ndim, num_heads, head_dim//2) """
    head_dim = embed_dim // num_heads
    freqs = torch.zeros(ndim, num_heads, head_dim//2)

    constant_freqs = 1 / (theta_base ** (torch.arange(0,head_dim,2)[:(head_dim//2)].float() / head_dim))
    for i in range(ndim):
        for j in range(num_heads):
            angles = torch.rand(1) * 2 * torch.pi
            freqs[i, j, :] = constant_freqs * torch.cos(angles)
    return freqs


##### Concatenate the per-dimension
def concat_init_constant_sep_freqs(divide_ratio, embed_dim, num_heads, theta_base=10000):
    """ per-layer freqs shape: [(num_heads, cur_head_dim//2), (num_heads, cur_head_dim//2), ...] """
    ndim = len(divide_ratio)
    head_dim = embed_dim // num_heads
    freqs = []
    for i in range(ndim):
        cur_dim = round(head_dim * divide_ratio[i])
        constant_freqs = 1 / (theta_base ** (torch.arange(0,cur_dim,2)[:(cur_dim//2)].float() / head_dim))
        cur_freqs = torch.zeros(num_heads, cur_dim//2)
        cur_freqs[..., :] = constant_freqs.unsqueeze(0)
        freqs.append(cur_freqs)
    return freqs

def concat_init_constant_cont_freqs(divide_ratio, embed_dim, num_heads, theta_base=10000):
    """ per-layer freqs shape: [(num_heads, cur_head_dim//2), (num_heads, cur_head_dim//2), ...] """
    ndim = len(divide_ratio)
    head_dim = embed_dim // num_heads
    freqs = []
    existing_dim = 0
    for i in range(ndim):
        cur_dim = round(head_dim * divide_ratio[i])
        constant_freqs = 1 / (theta_base ** (torch.arange(
            existing_dim, existing_dim+cur_dim, 2)[:(cur_dim//2)].float() / head_dim))
        cur_freqs = torch.zeros(num_heads, cur_dim//2)
        cur_freqs[..., :] = constant_freqs.unsqueeze(0)
        freqs.append(cur_freqs)
        existing_dim += cur_dim
    return freqs


def concat_init_learnable_freqs(divide_ratio, embed_dim, num_heads, theta_base=10000):
    ndim = len(divide_ratio)
    head_dim = embed_dim // num_heads
    freqs = []
    for i in range(ndim):
        cur_dim = round(head_dim * divide_ratio[i])
        constant_freqs = 1 / (theta_base ** (torch.arange(0,cur_dim,2)[:(cur_dim//2)].float() / cur_dim))
        cur_freqs = torch.zeros(num_heads, cur_dim//2)
        for j in range(num_heads):
            random_number = torch.rand(1)
            angles = random_number * 2 * torch.pi
            cur_freqs[j, :] = constant_freqs * torch.cos(angles)
        freqs.append(cur_freqs)
    return freqs


def init_grid_location(grid_shape):
    total_loc_cnt = 1
    for i in range(len(grid_shape)):
        total_loc_cnt *= grid_shape[i]
    
    locs = torch.arange(total_loc_cnt, dtype=torch.float32)
    each_dim_loc = []
    for i in range(len(grid_shape)):
        each_dim_loc.append(locs % grid_shape[-i-1])
        locs = locs // grid_shape[-i-1]
    each_dim_loc.reverse()
    return each_dim_loc


def compute_add_cis(freqs, each_dim_loc, num_heads):
    """
    now the freqs have all layers, shape (ndim, layer_cnt, num_heads*head_dim//2)
    each_dim_loc is the location of each token in each dimension, list of ndim tensors, each tensor has token_cnt elements
    """
    token_cnt = len(each_dim_loc[0])
    ndim = freqs.shape[0]
    layer_cnt = freqs.shape[1]
    # No float 16
    with torch.amp.autocast('cuda', enabled=False):
        for i in range(ndim):
            cur_angle = each_dim_loc[i].unsqueeze(-1).to(freqs[i].device) @ freqs[i].unsqueeze(-2) # (layer_cnt, token_cnt, num_heads*head_dim//2)
            # print(i, "cur_angle.shape", cur_angle.shape)
            cur_angle = cur_angle.view(layer_cnt, token_cnt, num_heads, -1).permute(0, 2, 1, 3) # (layer_cnt, num_heads, token_cnt, head_dim//2)
            # print(i, "cur_angle", cur_angle[0,0,17,:])
            if i == 0:
                angle_sum = cur_angle
            else:
                angle_sum += cur_angle
        freqs_cis = torch.polar(torch.ones_like(angle_sum), angle_sum)
        return freqs_cis

    
def compute_concat_cis(freqs, each_dim_loc, num_heads):
    """
    now the freqs have all layers, a list of ndim tensors, each tensor shape (layer_cnt, num_heads*cur_head_dim//2)
    each_dim_loc is the location of each token in each dimension, list of ndim tensors, each tensor has token_cnt elements
    """
    token_cnt = len(each_dim_loc[0])
    ndim = len(freqs)
    layer_cnt = freqs[0].shape[0]
    # No float 16
    with torch.amp.autocast('cuda', enabled=False):
        for i in range(ndim):
            cur_angle = each_dim_loc[i].unsqueeze(-1) @ freqs[i].unsqueeze(-2) # (layer_cnt, token_cnt, num_heads*cur_head_dim//2)
            # print(i, "cur_angle.shape", cur_angle.shape)
            cur_angle = cur_angle.view(layer_cnt, token_cnt, num_heads, -1).permute(0, 2, 1, 3) # (layer_cnt, num_heads, token_cnt, cur_head_dim//2)
            # print(i, "cur_angle", cur_angle[0,0,8,:])
            if i == 0:
                angle_concat = cur_angle
            else:
                angle_concat = torch.cat([angle_concat, cur_angle], dim=-1)
        freqs_cis = torch.polar(torch.ones_like(angle_concat), angle_concat)    # (layer_cnt, num_heads, token_cnt, head_dim//2)
        return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    # print(freqs_cis.shape, x.shape)
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
    return freqs_cis.view(*shape)


# xq, xk: (bs, num_heads, token_cnt, head_dim)
# freqs: (num_heads, token_cnt, head_dim//2)
def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class RoPEAttention(Attention):
    """Multi-head Attention block with relative position embeddings."""
    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q_rotated, k_rotated = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        q = torch.cat([q[:, :, :1], q_rotated], dim=2)
        k = torch.cat([k[:, :, :1], k_rotated], dim=2)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class RoPE_Layer_scale_init_Block(Layer_scale_init_Block):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = RoPEAttention
        super().__init__(*args, **kwargs)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        
        return x



