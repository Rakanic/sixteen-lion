"""CUDA decoder for one neural texture: packing, PyTorch reference, and binding.

    from sixteen.decoder import CudaDecoder
    dec = CudaDecoder.from_checkpoint("out/.../model.pt")   # NVFP4, default model shape
    rgb9 = dec(uv, frame=0)                                   # [N, 9], one STF tap per grid

The kernel (cuda/ntc_decode.cu) reads NVFP4 latents as stored: E2M1 codes,
one E4M3 scale per texel (16 channels = one block) and one fp32 scale per
grid. `reference` decodes the same taps in PyTorch for testing.
"""

import copy
import functools
import pathlib

import torch

from .model import NeuralTexture
from .quant import E2M1, E4M3, round_minifloat

ROOT = pathlib.Path(__file__).resolve().parents[2]
E2M1_MAG = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6.0])
K_PAD = (48, 64, 64)
N_PAD = (64, 64, 16)


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    return load(
        name="ntc_decode",
        sources=[str(ROOT / "cuda" / "ntc_decode.cu")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_120a,code=sm_120a", "--use_fast_math"],
        build_directory=str(_build_dir()),
        verbose=False,
    )


def _build_dir():
    d = ROOT / "out" / "build" / "ntc_decode"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- packing


def pack_nvfp4(grid):
    """[h, w, 16] fp32 latents -> (codes uint8 [h*w*8], scales uint8 [h*w], gscale, dequantized [h, w, 16]).

    Same arithmetic as sixteen.quant.nvfp4_quant with 16-value blocks = one texel.
    """
    h, w, c = grid.shape
    assert c == 16, "one NVFP4 block per texel needs 16 channels"
    b = grid.reshape(-1, 16).float()
    # Mirror nvfp4_quant op for op (tensor scales, same order): after QAT about a
    # third of the latents sit exactly on a rounding boundary, so a 1-ulp
    # difference in a scale flips their rounding.
    gscale_t = b.abs().amax().clamp(min=1e-12) / (E2M1.max * E4M3.max)
    s = round_minifloat(b.abs().amax(1, keepdim=True) / E2M1.max / gscale_t, E4M3)  # E4M3 value
    bscale_q = s * gscale_t
    keep = bscale_q > 0
    bscale_q = torch.where(keep, bscale_q, torch.ones_like(bscale_q))
    s = torch.where(keep, s, torch.ones_like(s) / gscale_t)
    v = round_minifloat(b / bscale_q, E2M1)
    gscale = gscale_t.item()
    mag = (v.abs()[..., None] == E2M1_MAG.to(v.device)).float().argmax(-1)
    codes = (mag | ((v < 0).long() << 3)).to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous().view(-1)
    scales = s.view(-1).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    deq = (v * bscale_q).view(h, w, c)
    return packed, scales, gscale, deq


def pack_weights(mlp):
    """nn.Sequential of 3 Linear layers -> (fragment-ordered fp16 weights as int32, bias fp32 [144])."""
    linears = [m for m in mlp if isinstance(m, torch.nn.Linear)]
    assert [l.in_features for l in linears] == [40, 64, 64] and [l.out_features for l in linears] == [64, 64, 9], \
        "kernel is built for the default model (grids (4,16),(8,16), hidden 64 x 2, 9 outputs)"
    lane = torch.arange(32)
    gq, tq = lane // 4, lane % 4
    frags, biases = [], []
    for lin, kp, np_ in zip(linears, K_PAD, N_PAD):
        W = torch.zeros(kp, np_)
        W[: lin.in_features, : lin.out_features] = lin.weight.detach().float().cpu().T
        Wh = W.half()
        for k in range(kp // 16):
            for n in range(np_ // 8):
                col = n * 8 + gq
                rows = [k * 16 + tq * 2 + o for o in (0, 1, 8, 9)]
                vals = torch.stack([Wh[r, col] for r in rows], dim=1)  # [32 lanes, 4 halves]
                frags.append(vals)
        b = torch.zeros(np_)
        b[: lin.out_features] = lin.bias.detach().float().cpu()
        biases.append(b)
    wfrag = torch.stack(frags).contiguous().view(torch.int32).view(-1)  # per frag: 32 lanes x uint2
    return wfrag, torch.cat(biases)


# ---------------------------------------------------------------- STF taps (bit-exact with the kernel)

M32 = 0xFFFFFFFF


def _pcg(v):
    s = (v * 747796405 + 2891336453) & M32
    w = (((s >> ((s >> 28) + 4)) ^ s) * 277803737) & M32
    return ((w >> 22) ^ w) & M32


def _uniform(seed):
    return (_pcg(seed) >> 8).float() * (1.0 / 16777216.0)


def stf_texels(uv, w, h, seed):
    x = uv[:, 0] * w - 0.5
    y = uv[:, 1] * h - 0.5
    x0, y0 = torch.floor(x), torch.floor(y)
    xi = x0.long() + (_uniform(seed) < (x - x0)).long()
    yi = y0.long() + (_uniform(seed ^ 0x9E3779B9) < (y - y0)).long()
    return (yi % h) * w + (xi % w)


def pixel_seeds(n, frame, device):
    p = torch.arange(n, device=device, dtype=torch.int64)
    s0 = _pcg(((p * 2654435761) & M32) ^ (frame & M32))
    return s0, _pcg(s0)


# ---------------------------------------------------------------- decoder


class CudaDecoder:
    def __init__(self, model: NeuralTexture, device="cuda"):
        self.model = model.to(device).eval()
        self.grids = []
        for g in model.latents:
            codes, scales, gscale, deq = pack_nvfp4(g.detach().to(device))
            h, w, _ = g.shape
            self.grids.append(dict(codes=codes, scales=scales, gscale=gscale, w=w, h=h, deq=deq))
        wfrag, bias = pack_weights(model.mlp)
        self.wfrag, self.bias = wfrag.to(device), bias.to(device)
        sm = torch.cuda.get_device_properties(device).multi_processor_count
        self.blocks = sm * 4

    @classmethod
    def from_checkpoint(cls, path, device="cuda"):
        ck = torch.load(path, weights_only=False, map_location="cpu")
        cfg = ck["cfg"]
        H = W = {"1k": 1024, "2k": 2048, "4k": 4096}[cfg["res"]]
        model = NeuralTexture((H, W), 9, grids=[tuple(g) for g in ck["grids"]], hidden=cfg["hidden"], layers=cfg["layers"],
                             filter=cfg.get("filter", "bilinear"))
        model.load_state_dict(ck["state"])
        return cls(model, device)

    def bytes(self):
        lat = sum(g["codes"].numel() + g["scales"].numel() + 4 for g in self.grids)
        return {"latents": lat, "weights_fp16": self.wfrag.numel() * 4 + self.bias.numel() * 4}

    def __call__(self, uv, frame=0):
        g0, g1 = self.grids
        return _ext().decode(g0["codes"], g0["scales"], g0["gscale"], g0["w"], g0["h"],
                             g1["codes"], g1["scales"], g1["gscale"], g1["w"], g1["h"],
                             uv.contiguous(), int(frame), self.wfrag, self.bias, self.blocks)

    @torch.no_grad()
    def reference(self, uv, frame=0, half=False):
        """PyTorch decode of the same STF taps (fp32, or fp16 weights/activations if half=True)."""
        s0, s1 = pixel_seeds(uv.shape[0], frame, uv.device)
        feats = []
        for g, seed in zip(self.grids, (s0, s1)):
            t = stf_texels(uv, g["w"], g["h"], seed)
            feats.append(g["deq"].reshape(-1, 16)[t])
        x = torch.cat(feats + [self.model.encode_pos(uv)], dim=-1)
        if half:
            return copy.deepcopy(self.model.mlp).half()(x.half()).float()
        return self.model.mlp(x)
