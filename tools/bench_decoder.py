"""Accuracy and speed of the CUDA decoder on one material.

    pixi run python tools/bench_decoder.py <model.pt>

Accuracy: kernel vs PyTorch on the same STF taps; and the STF image averaged
over frames vs the bilinear PyTorch decode the model was trained with.
Speed: one 1920x1080 frame of decodes (one texture lookup per pixel).
"""

import argparse
import json
import math

import torch

from sixteen.decoder import CudaDecoder


def timed(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--frames", type=int, default=64, help="STF frames to average for the accuracy check")
    args = ap.parse_args()
    dec = CudaDecoder.from_checkpoint(args.ckpt)
    r = {"ckpt": args.ckpt, "bytes": dec.bytes()}

    # 1. kernel vs reference, same taps
    torch.manual_seed(0)
    uv = torch.rand(1 << 20, 2, device="cuda")
    out = dec(uv, 3)
    r["max_abs_err_vs_fp16_ref"] = (out - dec.reference(uv, 3, half=True)).abs().max().item()
    r["mean_abs_err_vs_fp32_ref"] = (out - dec.reference(uv, 3)).abs().mean().item()

    # 2. STF average over frames vs bilinear decode (what training optimized), on a 512x512 texel crop
    H = W = 2048
    ys, xs = torch.meshgrid(torch.arange(512, device="cuda"), torch.arange(512, device="cuda"), indexing="ij")
    uvc = torch.stack([(xs + 0.5) / W, (ys + 0.5) / H], -1).view(-1, 2)
    with torch.no_grad():
        lat = [g["deq"] for g in dec.grids]
        bil = dec.model(uvc, lat)
    acc = torch.zeros_like(bil)
    psnr_by_frames = {}
    for f in range(args.frames):
        acc += dec(uvc, f)
        if (f + 1) in (1, 4, 16, 64):
            mse = ((acc / (f + 1)).clamp(0, 1) - bil.clamp(0, 1)).pow(2).mean().item()
            psnr_by_frames[f + 1] = round(10 * math.log10(1 / max(mse, 1e-12)), 2)
    r["stf_vs_bilinear_psnr_db_by_frames"] = psnr_by_frames

    # 3. speed: one 1080p frame, texture tiled 4x across the screen
    n = 1920 * 1080
    ys, xs = torch.meshgrid(torch.arange(1080, device="cuda"), torch.arange(1920, device="cuda"), indexing="ij")
    uvs = torch.stack([((xs + 0.5) / 1920 * 4) % 1, ((ys + 0.5) / 1080 * 4) % 1], -1).view(-1, 2).contiguous()
    ms = timed(lambda: dec(uvs, 0))
    ms_ref = timed(lambda: dec.reference(uvs, 0, half=True), iters=10, warmup=2)
    r["cuda_ms_1080p"] = ms
    r["cuda_us_per_mpix"] = ms * 1000 / (n / 1e6)
    r["pytorch_fp16_ms_1080p"] = ms_ref
    flops = n * 2 * (48 * 64 + 64 * 64 + 64 * 16)  # padded MMA work
    r["tensor_tflops_effective"] = flops / (ms * 1e-3) / 1e12
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
