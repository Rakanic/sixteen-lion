"""PyTorch baselines for decoder v2, same work as the CUDA kernel.

Each baseline does what the kernel does for one 1080p frame: output-space STF
texel choice, the 4 neighbouring latent texels of both grids, positional
encoding + fractions, and the 3-layer MLP. Latents and layer-1 weights are
quantized once up front (pre-quantized), so only decoding is timed.

    pixi run python tools/bench_baselines.py <concat4 + nvfp4-weights model.pt> [--x2 <nvfp4x2-weights model.pt>]
"""

import argparse
import json
import statistics

import torch
import torch.nn.functional as F

from sixteen.decoder import M32, CudaDecoderV2, _pcg, stf_texels
from sixteen.model import neighbours4_wrap

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from profile_decoder import frame_uv, timed  # noqa: E402


def make_forward(dec, dtype):
    lin = [m for m in dec.model.mlp if isinstance(m, torch.nn.Linear)]
    w1 = lin[0].weight.detach().float().clone()
    from sixteen.quant import fake_quant
    w1[:, :128] = fake_quant(w1[:, :128], "nvfp4")  # pre-quantized once
    ws = [w1.to(dtype), lin[1].weight.detach().to(dtype), lin[2].weight.detach().to(dtype)]
    bs = [l.bias.detach().to(dtype) for l in lin]
    grids = [g["deq"].to(dtype) for g in dec.grids]
    model = dec.model

    def fwd(uv, pix):
        # same STF choice as the kernel; the pixel index is an input (the kernel
        # gets it from the thread id), which also avoids a torch.compile int32 bug
        s0 = _pcg(((pix * 2654435761) & M32) ^ 0)
        t = stf_texels(uv, dec.W, dec.H, s0)
        uvc = torch.stack([(t % dec.W + 0.5) / dec.W, (t // dec.W + 0.5) / dec.H], dim=-1)
        taps, fracs = zip(*[neighbours4_wrap(g, uvc) for g in grids])
        x = torch.cat(list(taps) + [model.encode_pos(uvc).to(dtype)] + [f.to(dtype) for f in fracs], dim=-1)
        h = F.relu(F.linear(x, ws[0], bs[0]))
        h = F.relu(F.linear(h, ws[1], bs[1]))
        return F.linear(h, ws[2], bs[2])

    return fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--x2", help="concat4 model with two-term NVFP4 layer-1 weights: also time FP4-direct two-part")
    ap.add_argument("--out", default="results/decoder/v2_baselines.json")
    args = ap.parse_args()
    dec = CudaDecoderV2.from_checkpoint(args.ckpt)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    uv = frame_uv(4.0)
    ref = dec.reference(uv, 0)
    pix = torch.arange(uv.shape[0], device="cuda", dtype=torch.int64)
    variants = {
        "pytorch eager fp32 (CUDA cores)": (torch.float32, False, False),
        "pytorch eager fp32, TF32 Tensor Cores": (torch.float32, True, False),
        "pytorch eager fp16 (Tensor Cores)": (torch.float16, False, False),
        "pytorch compiled fp32": (torch.float32, False, True),
        "pytorch compiled fp16 (Tensor Cores)": (torch.float16, False, True),
    }
    r = {"frame": "1920x1080, one lookup per pixel, texture tiled 4x", "gpu": torch.cuda.get_device_name(), "ms": {}, "max_abs_err_vs_fp32_ref": {}}
    for name, (dtype, tf32, comp) in variants.items():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        f = make_forward(dec, dtype)
        if comp:
            f = torch.compile(f, mode="max-autotune-no-cudagraphs")
        out = f(uv, pix).float()
        r["max_abs_err_vs_fp32_ref"][name] = (out - ref).abs().max().item()
        r["ms"][name] = statistics.median(timed(lambda: f(uv, pix), iters=20, warmup=3) for _ in range(3))
        print(f"{name:42s} {r['ms'][name]:8.3f} ms   max err {r['max_abs_err_vs_fp32_ref'][name]:.4f}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    for name, kw in {"CUDA FP16-unpack (ours)": dict(fp4=False, min_blocks=1), "CUDA FP4-direct (ours)": dict(fp4=True, min_blocks=4)}.items():
        out = dec(uv, 0, blocks=sm * 4, **kw)
        r["max_abs_err_vs_fp32_ref"][name] = (out - ref).abs().max().item()
        r["ms"][name] = statistics.median(timed(lambda: dec(uv, 0, blocks=sm * 4, **kw)) for _ in range(5))
        print(f"{name:42s} {r['ms'][name]:8.3f} ms   max err {r['max_abs_err_vs_fp32_ref'][name]:.4f}", flush=True)
    if args.x2:
        d2 = CudaDecoderV2.from_checkpoint(args.x2)
        name, ref2 = "CUDA FP4-direct two-part weights (ours)", d2.reference(uv, 0)
        r["max_abs_err_vs_fp32_ref"][name] = (d2(uv, 0, blocks=sm * 4, min_blocks=4) - ref2).abs().max().item()
        r["ms"][name] = statistics.median(timed(lambda: d2(uv, 0, blocks=sm * 4, min_blocks=4)) for _ in range(5))
        print(f"{name:42s} {r['ms'][name]:8.3f} ms   max err {r['max_abs_err_vs_fp32_ref'][name]:.4f}", flush=True)
    fp4 = r["ms"]["CUDA FP4-direct (ours)"]
    r["speedup_of_fp4_direct_over"] = {k: v / fp4 for k, v in r["ms"].items()}
    json.dump(r, open(args.out, "w"), indent=1)
    print(json.dumps({k: round(v, 2) for k, v in r["speedup_of_fp4_direct_over"].items()}, indent=1))


if __name__ == "__main__":
    main()
