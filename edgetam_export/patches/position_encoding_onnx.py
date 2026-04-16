# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional, Tuple

import numpy as np

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}

    def _encode_xy(self, x, y):
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


# Rotary Positional Encoding, adapted from:
# 1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
# 2. https://github.com/naver-ai/rope-vit
# 3. https://github.com/lucidrains/rotary-embedding-torch


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    # Store as real-valued [cos, sin] stacked along last dim for ONNX compatibility
    # freqs_x/y: [N, dim/4], cos/sin: [N, dim/4] each
    cos_x, sin_x = torch.cos(freqs_x), torch.sin(freqs_x)
    cos_y, sin_y = torch.cos(freqs_y), torch.sin(freqs_y)
    # Interleave x and y to match original complex cat layout: [N, dim/2] per component
    cos_all = torch.cat([cos_x, cos_y], dim=-1)  # [N, dim/2]
    sin_all = torch.cat([sin_x, sin_y], dim=-1)  # [N, dim/2]
    return torch.stack([cos_all, sin_all], dim=-1)  # [N, dim/2, 2]


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # freqs_cis: [N, D, 2], x: [..., N, D, 2]
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape[-3:-1] == x.shape[-3:-1] or freqs_cis.shape[:-1] == x.shape[-2:]
    if freqs_cis.ndim == 3 and x.ndim > 3:
        # [N, D, 2] -> [1, ..., 1, N, D, 2]
        shape = [1] * (ndim - 3) + list(freqs_cis.shape)
        return freqs_cis.view(*shape)
    return freqs_cis


def _apply_rotary_real(x_pairs: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary encoding using real-valued arithmetic (ONNX compatible).
    x_pairs: [..., N, D/2, 2] where [0]=real, [1]=imag
    freqs_cis: [..., N, D/2, 2] where [0]=cos, [1]=sin (broadcast-ready)
    Returns: [..., N, D/2, 2]
    """
    x_re = x_pairs[..., 0]
    x_im = x_pairs[..., 1]
    cos = freqs_cis[..., 0]
    sin = freqs_cis[..., 1]
    out_re = x_re * cos - x_im * sin
    out_im = x_re * sin + x_im * cos
    return torch.stack([out_re, out_im], dim=-1)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    # xq: [B, heads, N, C], reshape to [B, heads, N, C/2, 2]
    xq_pairs = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_pairs = (
        xk.float().reshape(*xk.shape[:-1], -1, 2)
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_pairs)
    xq_out = _apply_rotary_real(xq_pairs, freqs_cis).flatten(-2)
    if xk_pairs is None:
        return xq_out.type_as(xq).to(xq.device), xk
    if repeat_freqs_k:
        r = xk_pairs.shape[-3] // xq_pairs.shape[-3]
        freqs_cis = freqs_cis.unsqueeze(-4).expand(
            *([1] * (freqs_cis.ndim - 3)), r, -1, -1, -1
        ).flatten(-4, -3)
    xk_out = _apply_rotary_real(xk_pairs, freqs_cis).flatten(-2)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def apply_rotary_enc_v2(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs: int,
):
    if repeat_freqs == 0:
        assert x.shape[-2] == 0
    if x.shape[-2] == 0:
        return x

    B, N_heads, N_tokens, C_per_head = x.shape
    # freqs_cis: [rope_tokens, D/2, 2]
    rope_tokens = freqs_cis.shape[0]

    if N_tokens == rope_tokens * repeat_freqs:
        x_rope = x
        x_no_rope = None
    else:
        no_rope_tokens = N_tokens // repeat_freqs - rope_tokens
        x = x.view(B, N_heads, repeat_freqs, N_tokens // repeat_freqs, C_per_head)
        x_rope = x[..., no_rope_tokens:, :].reshape(B, N_heads, -1, C_per_head)
        x_no_rope = x[..., :no_rope_tokens, :].reshape(B, N_heads, -1, C_per_head)

    # Reshape to pairs: [B, heads, N, C/2, 2]
    x_pairs = x_rope.float().reshape(*x_rope.shape[:-1], -1, 2)

    if repeat_freqs > 1:
        one_frame_tokens = x_pairs.shape[-3] // repeat_freqs
        x_one_frame = x_pairs[..., :one_frame_tokens, :, :]
        freqs_cis = reshape_for_broadcast(freqs_cis, x_one_frame)
    else:
        freqs_cis = reshape_for_broadcast(freqs_cis, x_pairs)

    if repeat_freqs > 1:
        freqs_cis = freqs_cis.unsqueeze(-4).expand(
            *([1] * (freqs_cis.ndim - 3)), repeat_freqs, -1, -1, -1
        ).flatten(-4, -3)

    x_out = _apply_rotary_real(x_pairs, freqs_cis).flatten(-2)
    x_out = x_out.type_as(x).to(x.device)

    if x_no_rope is not None:
        x_out = x_out.view(B, N_heads, repeat_freqs, -1, C_per_head)
        x_no_rope = x_no_rope.view(B, N_heads, repeat_freqs, -1, C_per_head)
        x_out = torch.cat((x_no_rope, x_out), dim=3).view(
            B, N_heads, N_tokens, C_per_head
        )
    return x_out
