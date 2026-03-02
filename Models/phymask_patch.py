"""
Patch-wise PhyMask: adaptive masking for ViT-style patch tokens.

Unlike the per-element PhyMask (test_phymask.py), this module computes Energy and
Coherence at the patch level and outputs which *patches* to mask, compatible with
RopeViT and other patch-based vision transformers.

Input: raw signal (B, C, *spatial_dims) e.g. (B, 1, H, W) or (B, 1, T, H, W).
Output: mask_index (B, mask_patch_cnt), mask_dense (B, num_patches), and score map P.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _unfold_2d(x, patch_size, stride):
    """Unfold 4D (B, C, H, W) into (B, num_patches, C*patch_h*patch_w)."""
    B, C, H, W = x.shape
    ph, pw = patch_size
    unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
    # (B, C*ph*pw, num_patches)
    out = unfold(x)
    return out.transpose(1, 2)  # (B, num_patches, C*ph*pw)


def _unfold_3d(x, patch_size, stride, grid_shape):
    """Unfold 5D (B, C, T, H, W) into (B, num_patches, C*pt*ph*pw)."""
    B, C, T, H, W = x.shape
    pt, ph, pw = patch_size
    # Reshape so each patch is one block
    x = x.view(
        B, C,
        grid_shape[0], pt,
        grid_shape[1], ph,
        grid_shape[2], pw,
    )
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()
    x = x.view(B, grid_shape[0] * grid_shape[1] * grid_shape[2], C * pt * ph * pw)
    return x


def _grid_shape_2d(x_shape, patch_size, stride):
    """(B, C, H, W) -> (grid_h, grid_w)."""
    _, _, H, W = x_shape
    ph, pw = patch_size
    sh, sw = stride if isinstance(stride, (list, tuple)) else (stride, stride)
    return (H // sh, W // sw)


def _grid_shape_3d(x_shape, patch_size, stride):
    """(B, C, T, H, W) -> (grid_t, grid_h, grid_w)."""
    _, _, T, H, W = x_shape
    pt, ph, pw = patch_size
    if isinstance(stride, (list, tuple)):
        st, sh, sw = stride[0], stride[1], stride[2]
    else:
        st, sh, sw = stride, stride, stride
    return (T // st, H // sh, W // sw)


class PatchPhyMask(nn.Module):
    """
    PhyMask applied at the patch level for patch-based ViTs (e.g. RopeViT).

    - Energy: per-patch signal power (sum of squares over patch elements), normalized.
    - Coherence: per-patch temporal/batch consistency (cosine similarity of patch
      vectors across the batch), then combined with energy to form a single
      importance score per patch. Patches with *lowest* scores are masked.
    """

    def __init__(self, patch_size, mask_ratio=0.75, method='bayesian', weights=(0.5, 0.5), patch_stride=None):
        """
        Args:
            patch_size: (ph, pw) for 2D or (pt, ph, pw) for 3D.
            mask_ratio: Fraction of patches to mask (e.g. 0.75 -> mask 75% of patches).
            method: 'bayesian' (P = E*C) or 'weighted' (P = w_c*C + w_e*E).
            weights: (w_c, w_e) for method='weighted'.
            patch_stride: Stride for patching. If None, uses patch_size.
        """
        super().__init__()
        self.patch_size = patch_size if isinstance(patch_size, (list, tuple)) else [patch_size] * 2
        self.stride = patch_stride if patch_stride is not None else self.patch_size
        if isinstance(self.stride, int):
            self.stride = [self.stride] * len(self.patch_size)
        self.mask_ratio = mask_ratio
        self.method = method
        self.weights = weights
        self.ndim = len(self.patch_size)

    def _get_grid_shape(self, x):
        if self.ndim == 2:
            return _grid_shape_2d(x.shape, self.patch_size, self.stride)
        return _grid_shape_3d(x.shape, self.patch_size, self.stride)

    def _unfold(self, x, grid_shape):
        if self.ndim == 2:
            return _unfold_2d(x, self.patch_size, self.stride)
        return _unfold_3d(x, self.patch_size, self.stride, grid_shape)

    def _compute_patch_energy(self, x_patches):
        """
        x_patches: (B, N, D). Energy per patch = sum of squares over D, then min-max norm per sample.
        Returns: (B, N).
        """
        power = (x_patches ** 2).sum(dim=2)  # (B, N)
        B, N = power.shape
        p_flat = power.view(B, -1)
        vmin = p_flat.min(dim=1, keepdim=True)[0]
        vmax = p_flat.max(dim=1, keepdim=True)[0]
        denom = vmax - vmin
        denom[denom == 0] = 1.0
        E = (p_flat - vmin) / denom
        return E.view(B, N)

    def _compute_patch_coherence(self, x_patches):
        """
        x_patches: (B, N, D). For each patch index n, vectors (B, D); L2 normalize,
        then (B,B) similarity, then mean over batch -> (B,) per n. Result (B, N).
        """
        B, N, D = x_patches.shape
        # (B, N, D) -> (N, B, D)
        x_n = x_patches.permute(1, 0, 2)  # (N, B, D)
        x_norm = F.normalize(x_n, p=2, dim=2)  # (N, B, D)
        # (N, B, D) @ (N, D, B) -> (N, B, B)
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))
        # Average over "other" batch dimension (dim=2), get (N, B) -> (B, N)
        coh = sim.mean(dim=2).t()  # (B, N)
        return coh

    def forward(self, x):
        """
        x: (B, C, *spatial) e.g. (B, 1, H, W) or (B, 1, T, H, W).

        Returns:
            mask_index: (B, mask_patch_cnt) long tensor of patch indices to MASK.
            mask_dense: (B, num_patches) float; 1 = keep, 0 = mask (for direct use with patch tokens).
            P: (B, num_patches) importance score map (higher = keep).
            grid_shape: tuple of grid dimensions (for caller).
        """
        grid_shape = self._get_grid_shape(x)
        x_patches = self._unfold(x, grid_shape)  # (B, N, D)
        B, num_patches, _ = x_patches.shape

        E = self._compute_patch_energy(x_patches)   # (B, N)
        C = self._compute_patch_coherence(x_patches)  # (B, N)

        if self.method == 'weighted':
            wc, we = self.weights
            P = (wc * C + we * E) / (wc + we)
        else:
            P = E * C

        num_keep = int(num_patches * (1 - self.mask_ratio))
        num_keep = max(1, min(num_keep, num_patches - 1))
        mask_patch_cnt = num_patches - num_keep

        # Patches with *lowest* P are masked; we want indices of those.
        _, order_asc = torch.sort(P, dim=1, descending=False)  # (B, N) indices from smallest to largest
        mask_index = order_asc[:, :mask_patch_cnt]  # (B, mask_patch_cnt) indices to mask

        mask_dense = torch.ones(B, num_patches, device=x.device, dtype=x.dtype)
        mask_dense.scatter_(1, mask_index, 0.0)  # 0 where masked, 1 where kept

        return mask_index, mask_dense, P, grid_shape


if __name__ == "__main__":
    # Quick test: 2D and 3D inputs
    B, C, H, W = 4, 1, 32, 64
    x2d = torch.rand(B, C, H, W)
    phymask_2d = PatchPhyMask(patch_size=(4, 8), mask_ratio=0.5, patch_stride=(4, 8))
    mask_index, mask_dense, P, grid_shape = phymask_2d(x2d)
    assert mask_index.shape[0] == B and mask_dense.shape == (B, grid_shape[0] * grid_shape[1])
    print("2D test OK: grid_shape =", grid_shape, "mask_index.shape =", mask_index.shape)

    B, C, T, H, W = 2, 1, 8, 16, 16
    x3d = torch.rand(B, C, T, H, W)
    phymask_3d = PatchPhyMask(patch_size=(2, 4, 4), mask_ratio=0.5, patch_stride=(2, 4, 4))
    mask_index3, mask_dense3, P3, grid_shape3 = phymask_3d(x3d)
    assert mask_index3.shape[0] == B
    print("3D test OK: grid_shape =", grid_shape3, "mask_index.shape =", mask_index3.shape)
