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


# ================================================================ v2: concat4 model, FP4-direct layer 1


@functools.lru_cache(maxsize=None)
def _ext_v2(min_blocks=1, prefetch=0):
    """min_blocks: blocks of 256 threads per SM the compiler must fit (caps registers per thread)."""
    from torch.utils.cpp_extension import load

    name = f"ntc_decode_v2_mb{min_blocks}_pf{prefetch}"
    d = ROOT / "out" / "build" / name
    d.mkdir(parents=True, exist_ok=True)
    return load(
        name=name,
        sources=[str(ROOT / "cuda" / "ntc_decode_v2.cu")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_120a,code=sm_120a", "--use_fast_math",
                           f"-DMIN_BLOCKS={min_blocks}", f"-DPREFETCH={prefetch}"],
        build_directory=str(d),
        verbose=False,
    )


def nvfp4_rows(x):
    """NVFP4 of a 2-D tensor, blocks of 16 along each row, mirroring nvfp4_quant.

    Returns (codes [R, C] int64 E2M1 codes, scales [R, C/16] E4M3 values,
    gscale (0-dim tensor), dequantized [R, C]).
    """
    R, C = x.shape
    b = x.reshape(-1, 16).float()
    gscale_t = b.abs().amax().clamp(min=1e-12) / (E2M1.max * E4M3.max)
    s = round_minifloat(b.abs().amax(1, keepdim=True) / E2M1.max / gscale_t, E4M3)
    bscale_q = s * gscale_t
    keep = bscale_q > 0
    bscale_q = torch.where(keep, bscale_q, torch.ones_like(bscale_q))
    s = torch.where(keep, s, torch.ones_like(s) / gscale_t)
    v = round_minifloat(b / bscale_q, E2M1)
    mag = (v.abs()[..., None] == E2M1_MAG.to(v.device)).float().argmax(-1)
    codes = mag | ((v < 0).long() << 3)
    return codes.view(R, C), s.view(R, C // 16), gscale_t, (v * bscale_q).view(R, C)


def frag_fp16(Wkn):
    """[K, N] weights (K, N multiples of 16, 8) -> FP16 B fragments as int32, order (k-tile, n-tile, lane, 2 words)."""
    K, N = Wkn.shape
    lane = torch.arange(32)
    gq, tq = lane // 4, lane % 4
    Wh = Wkn.half().cpu()
    frags = []
    for k in range(K // 16):
        for n in range(N // 8):
            rows = [k * 16 + tq * 2 + o for o in (0, 1, 8, 9)]
            frags.append(torch.stack([Wh[r, n * 8 + gq] for r in rows], dim=1))
    return torch.stack(frags).contiguous().view(torch.int32).view(-1)


class CudaDecoderV2:
    """Decoder for concat4 models with NVFP4 layer-1 weights (train with filter="concat4", w1_fmt="nvfp4"
    or "nvfp4x2", two NVFP4 terms W ~ q(W) + q(W - q(W)), one FP4 MMA each)."""

    MODES = {"fp16": 0, "fp4": 1, "fp4x2": 2}

    def __init__(self, model: NeuralTexture, res, device="cuda"):
        assert model.filter == "concat4" and model.w1_fmt in ("nvfp4", "nvfp4x2"), "needs concat4 + w1_fmt nvfp4[x2]"
        self.terms = 2 if model.w1_fmt == "nvfp4x2" else 1
        lin = [m for m in model.mlp if isinstance(m, torch.nn.Linear)]
        assert [l.in_features for l in lin] == [140, 64, 64] and [l.out_features for l in lin] == [64, 64, 9]
        self.model = model.to(device).eval()
        self.H, self.W = res
        self.grids = []
        for g in model.latents:
            h, w, c = g.shape
            codes, s, gs, deq = nvfp4_rows(g.detach().to(device).reshape(-1, 16))
            packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).contiguous().view(-1)
            scales = s.view(-1).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
            self.grids.append(dict(codes=packed, scales=scales, gscale=gs.item(), w=w, h=h, deq=deq.view(h, w, c)))
        # layer 1, latent part: [64 out, 128 in] NVFP4 along the input
        W1 = lin[0].weight.detach().float().to(device)
        terms, rest = [], W1[:, :128].contiguous()
        for _ in range(self.terms):
            terms.append(nvfp4_rows(rest))
            rest = rest - terms[-1][3]
        wdeq = sum(t[3] for t in terms)
        lane = torch.arange(32, device=device)
        gq, tq = lane // 4, lane % 4
        fp4, sf, self.k = [], [], []
        for k, (t, (wc, ws, wgs, _)) in ((k, tt) for k in range(2) for tt in enumerate(terms)):
            self.k.append(self.grids[k]["gscale"] * wgs.item())
            for n in range(8):
                row = n * 8 + gq  # output neuron = MMA column
                regs = []
                for j in range(2):
                    cols = k * 64 + 32 * j + tq * 8  # first K index of this register
                    word = torch.zeros(32, dtype=torch.int64, device=device)
                    for i in range(8):
                        word |= wc[row, cols + i] << (4 * i)
                    regs.append(word)
                fp4.append(torch.stack(regs, dim=1))
                sbytes = ws[row][:, k * 4 : k * 4 + 4].to(torch.float8_e4m3fn).view(torch.uint8).long()
                sf.append(sbytes[:, 0] | (sbytes[:, 1] << 8) | (sbytes[:, 2] << 16) | (sbytes[:, 3] << 24))
        to_i32 = lambda t: (t & 0xFFFFFFFF).to(torch.int64).where(t < 2**31, t - 2**32).to(torch.int32)
        self.w1_fp4 = to_i32(torch.stack(fp4)).contiguous().view(-1)
        self.w1_sf = to_i32(torch.stack(sf)).contiguous().view(-1)
        # FP16-unpack reads the same FP4 registers as the FP4 path; its K-16 tile (grid k, j) holds,
        # at fragment row r, latent input k*64 + texel*16 + ch (see compute<MODE_F16> in the kernel)
        perm = []
        for k in range(2):
            for j in range(4):
                for r in range(16):
                    t, o, hi = (r % 8) // 2, r % 2, r // 8
                    texel = (t >> 1) + (2 if j >= 2 else 0)
                    perm.append(k * 64 + texel * 16 + (t & 1) * 8 + (j & 1) * 4 + hi * 2 + o)
        self.w1_fp16 = frag_fp16(wdeq.T[perm].contiguous()).to(device)  # dequantized, gscale folded in
        pe = torch.zeros(16, 64)
        pe[:12] = W1[:, 128:140].T.cpu()
        self.w1_pe = frag_fp16(pe).to(device)
        W2, W3 = lin[1].weight.detach().float().T.cpu(), torch.zeros(64, 16)
        W3[:, :9] = lin[2].weight.detach().float().T.cpu()
        self.w23 = torch.cat([frag_fp16(W2), frag_fp16(W3)]).to(device)
        b3 = torch.zeros(16)
        b3[:9] = lin[2].bias.detach().float().cpu()
        self.bias = torch.cat([lin[0].bias.detach().float().cpu(), lin[1].bias.detach().float().cpu(), b3]).to(device)
        self.k_t = torch.tensor(self.k, dtype=torch.float32, device=device)
        self.blocks = torch.cuda.get_device_properties(device).multi_processor_count * 4

    @classmethod
    def from_checkpoint(cls, path, device="cuda"):
        ck = torch.load(path, weights_only=False, map_location="cpu")
        cfg = ck["cfg"]
        H = W = {"1k": 1024, "2k": 2048, "4k": 4096}[cfg["res"]]
        model = NeuralTexture((H, W), 9, grids=[tuple(g) for g in ck["grids"]], hidden=cfg["hidden"],
                              layers=cfg["layers"], filter=cfg.get("filter", "bilinear"), w1_fmt=cfg.get("w1_fmt"))
        model.load_state_dict(ck["state"])
        return cls(model, (H, W), device)

    def __call__(self, uv, frame=0, fp4=True, min_blocks=1, blocks=None, mode=None, prefetch=0):
        """mode: "fp16", "fp4" (needs a 1-term model), "fp4x2" (2-term);
        default: the FP4-direct mode matching the model if fp4 else "fp16"."""
        if mode is None:
            mode = ("fp4x2" if self.terms == 2 else "fp4") if fp4 else "fp16"
        assert mode not in ("fp4", "fp4x2") or {"fp4": 1, "fp4x2": 2}[mode] == self.terms, "mode does not match the model"
        g0, g1 = self.grids
        return _ext_v2(min_blocks, prefetch).decode(g0["codes"], g0["scales"], g0["gscale"], g0["w"], g0["h"],
                                g1["codes"], g1["scales"], g1["gscale"], g1["w"], g1["h"],
                                self.W, self.H, uv.contiguous(), int(frame), self.w1_fp4, self.w1_sf, self.w1_fp16,
                                self.w1_pe, self.w23, self.bias, self.k_t, self.MODES[mode],
                                blocks or self.blocks)

    def stf_centers(self, uv, frame=0):
        """Texel centers picked by the kernel's output-space STF (bit-exact)."""
        s0, _ = pixel_seeds(uv.shape[0], frame, uv.device)
        t = stf_texels(uv, self.W, self.H, s0)
        return torch.stack([(t % self.W + 0.5) / self.W, (t // self.W + 0.5) / self.H], dim=-1)

    @torch.no_grad()
    def reference(self, uv, frame=0):
        """The PyTorch model (quantized latents and weights, fp32 math) at the same texel centers."""
        return self.model(self.stf_centers(uv, frame), [g["deq"] for g in self.grids])
