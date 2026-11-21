"""Neural texture: latent grids + a small MLP decoder.

Each grid is a learned [h, w, C] feature table (channels last, so blocks of the
quantizer run along a texel's channels). A query at uv samples every grid with
wrap-around bilinear filtering, concatenates the features with a positional
encoding of the position inside the finest grid cell, and decodes all material
channels with one MLP call.

Storage cost = latent values * bits/value (format) + MLP weights at fp16.
"""

import math

import torch
import torch.nn as nn

from .quant import BITS_PER_VALUE, fake_quant, quantize


def bilinear_wrap(grid, uv):
    """Sample grid [h, w, C] at uv [N, 2] in [0, 1), tiling. Returns [N, C]."""
    h, w, c = grid.shape
    x = uv[:, 0] * w - 0.5
    y = uv[:, 1] * h - 0.5
    x0, y0 = torch.floor(x), torch.floor(y)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0 = x0.long() % w
    y0 = y0.long() % h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    flat = grid.reshape(-1, c)
    g = lambda yy, xx: flat[yy * w + xx]
    top = g(y0, x0) * (1 - fx) + g(y0, x1) * fx
    bot = g(y1, x0) * (1 - fx) + g(y1, x1) * fx
    return top * (1 - fy) + bot * fy


class NeuralTexture(nn.Module):
    def __init__(self, res, out_channels, grids=((4, 16), (8, 16)), hidden=64, layers=2, pe_freqs=2):
        """`grids` lists (downsample factor, channels) per latent grid; `res` is (H, W)."""
        super().__init__()
        H, W = res
        self.grid_shapes = [(H // s, W // s, c) for s, c in grids]
        self.latents = nn.ParameterList(
            [nn.Parameter(torch.empty(h, w, c).uniform_(-0.5, 0.5)) for h, w, c in self.grid_shapes]
        )
        self.fine_res = torch.tensor([self.grid_shapes[0][1], self.grid_shapes[0][0]], dtype=torch.float32)
        self.pe_freqs = pe_freqs
        dims = [sum(c for *_, c in self.grid_shapes) + 4 * pe_freqs] + [hidden] * layers + [out_channels]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.ReLU())
        self.mlp = nn.Sequential(*mods)
        self.fmt = None  # latent format used in forward(); None = full precision

    def encode_pos(self, uv):
        p = uv * self.fine_res.to(uv.device) * 2 * math.pi
        feats = []
        for k in range(self.pe_freqs):
            feats += [torch.sin(p * 2**k), torch.cos(p * 2**k)]
        return torch.cat(feats, dim=-1)

    def quantized_latents(self, fmt=None, hard=False):
        fmt = self.fmt if fmt is None else fmt
        q = quantize if hard else fake_quant
        return [q(g, fmt) for g in self.latents]

    def forward(self, uv, latents=None):
        latents = self.quantized_latents() if latents is None else latents
        feats = [bilinear_wrap(g, uv) for g in latents] + [self.encode_pos(uv)]
        return self.mlp(torch.cat(feats, dim=-1))

    @torch.no_grad()
    def decode_image(self, H, W, fmt=None, chunk=1 << 20):
        lat = self.quantized_latents(fmt, hard=True)
        dev = lat[0].device
        ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
        uv = torch.stack([(xs + 0.5) / W, (ys + 0.5) / H], dim=-1).reshape(-1, 2)
        out = torch.cat([self(uv[i : i + chunk], lat) for i in range(0, uv.shape[0], chunk)])
        return out.view(H, W, -1)

    def storage_bits(self, fmt=None):
        fmt = self.fmt if fmt is None else fmt
        n_lat = sum(g.numel() for g in self.latents)
        n_w = sum(p.numel() for p in self.mlp.parameters())
        return {"latent_bits": n_lat * BITS_PER_VALUE[fmt or "fp32"], "mlp_bits": n_w * 16.0}
