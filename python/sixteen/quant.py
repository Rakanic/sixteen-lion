"""Fake quantization of latent tensors in block-scaled formats.

Every quantizer maps a float tensor to the float values that the format can
represent (simulated quantization). `fake_quant` wraps it in a straight-through
estimator: the forward pass returns the quantized values, the backward pass
returns the gradient unchanged.

Blocks are formed from the last dimension of the tensor after flattening, so a
latent grid stored as [H, W, C] with C = 16 puts one texel's channels in one
NVFP4 block (16 values) and two neighbouring texels in one MX block (32 values).

Formats (OCP Microscaling spec v1.0; NVIDIA NVFP4):

    name     element        block  scale                  bits/value
    fp16     -              -      -                      16
    int8     int8 per-chan  -      fp32 per channel       8
    int4     int4 per-chan  -      fp32 per channel       4
    mxint8   int8 (2^-6)    32     E8M0                   8.25
    mxfp8    E4M3           32     E8M0                   8.25
    mxfp6    E2M3           32     E8M0                   6.25
    mxfp4    E2M1           32     E8M0                   4.25
    nvfp4    E2M1           16     E4M3 (+ fp32/tensor)   4.5
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MiniFloat:
    """A sign-magnitude float with `ebits` exponent and `mbits` mantissa bits."""

    ebits: int
    mbits: int
    max: float  # largest finite magnitude (the formats here saturate)

    @property
    def bias(self):
        return 2 ** (self.ebits - 1) - 1

    @property
    def emin(self):  # exponent of the smallest normal number
        return 1 - self.bias

    @property
    def emax(self):  # exponent of the largest normal number
        return int(torch.tensor(self.max).log2().floor())


E2M1 = MiniFloat(2, 1, 6.0)
E2M3 = MiniFloat(2, 3, 7.5)
E4M3 = MiniFloat(4, 3, 448.0)


def round_minifloat(x, fmt: MiniFloat):
    """Round to nearest (ties to even) onto the grid of `fmt`, saturating."""
    a = x.abs().clamp(max=fmt.max)
    e = torch.floor(torch.log2(a.clamp(min=2.0**fmt.emin))).clamp(min=fmt.emin)
    step = torch.exp2(e - fmt.mbits)
    return torch.sign(x) * (torch.round(a / step) * step).clamp(max=fmt.max)


def _blocks(x, size):
    flat = x.reshape(-1)
    pad = (-flat.numel()) % size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat.view(-1, size), pad


def _unblocks(b, pad, shape):
    flat = b.reshape(-1)
    if pad:
        flat = flat[:-pad]
    return flat.view(shape)


def mx_quant(x, elem, block=32, scale_offset=None):
    """OCP MX: shared power-of-two scale per block, X = 2^(floor(log2 amax) - emax_elem).

    `elem` is a MiniFloat or "int8" (MXINT8: int8 with an implicit 2^-6, so
    emax_elem = 0). `scale_offset` is an optional per-block integer tensor
    added to the shared exponent (hook for scale choices other than max-abs).
    """
    b, pad = _blocks(x, block)
    amax = b.abs().amax(dim=1, keepdim=True).clamp(min=2.0**-127)
    emax_elem = 0 if elem == "int8" else elem.emax
    shared = (torch.floor(torch.log2(amax)) - emax_elem).clamp(-127, 127)
    if scale_offset is not None:
        shared = (shared + scale_offset.view(-1, 1)).clamp(-127, 127)
    scale = torch.exp2(shared)
    v = b / scale
    if elem == "int8":
        q = torch.round(v * 64).clamp(-127, 127) / 64
    else:
        q = round_minifloat(v, elem)
    return _unblocks(q * scale, pad, x.shape)


def nvfp4_quant(x, block=16):
    """NVFP4: E2M1 elements, E4M3 scale per 16 values, fp32 scale per tensor."""
    b, pad = _blocks(x, block)
    amax_t = b.abs().amax().clamp(min=1e-12)
    gscale = amax_t / (E2M1.max * E4M3.max)
    bscale = b.abs().amax(dim=1, keepdim=True) / E2M1.max
    bscale_q = round_minifloat(bscale / gscale, E4M3) * gscale
    bscale_q = torch.where(bscale_q > 0, bscale_q, torch.ones_like(bscale_q))
    q = round_minifloat(b / bscale_q, E2M1) * bscale_q
    return _unblocks(q, pad, x.shape)


def int_per_channel(x, bits):
    """Symmetric uniform quantization with one fp32 scale per channel (last dim)."""
    qmax = 2 ** (bits - 1) - 1
    dims = tuple(range(x.dim() - 1))
    scale = x.abs().amax(dim=dims, keepdim=True).clamp(min=1e-12) / qmax
    return torch.round(x / scale).clamp(-qmax, qmax) * scale


QUANTIZERS = {
    "fp16": lambda x: x.half().float(),
    "int8": lambda x: int_per_channel(x, 8),
    "int4": lambda x: int_per_channel(x, 4),
    "mxint8": lambda x: mx_quant(x, "int8"),
    "mxfp8": lambda x: mx_quant(x, E4M3),
    "mxfp6": lambda x: mx_quant(x, E2M3),
    "mxfp4": lambda x: mx_quant(x, E2M1),
    "nvfp4": nvfp4_quant,
}

BITS_PER_VALUE = {
    "fp32": 32.0,
    "fp16": 16.0,
    "int8": 8.0,
    "int4": 4.0,
    "mxint8": 8 + 8 / 32,
    "mxfp8": 8 + 8 / 32,
    "mxfp6": 6 + 8 / 32,
    "mxfp4": 4 + 8 / 32,
    "nvfp4": 4 + 8 / 16,
}


def quantize(x, fmt):
    if fmt in (None, "fp32"):
        return x
    return QUANTIZERS[fmt](x)


def fake_quant(x, fmt):
    """Quantize in the forward pass, identity gradient in the backward pass."""
    if fmt in (None, "fp32"):
        return x
    return x + (quantize(x.detach(), fmt) - x).detach()
