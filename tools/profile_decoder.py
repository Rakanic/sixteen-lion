"""Profile decoder v2 (FP4-direct vs FP16-unpack) on one material.

Run on an idle GPU. Two modes:

    pixi run python tools/profile_decoder.py <model.pt> --time      # event timing, both paths
    pixi run ncu --kernel-name regex:decode_v2 --launch-skip 4 --launch-count 2 --set full \
        -o out/profile_decoder/v2 python tools/profile_decoder.py <model.pt>   # Nsight Compute

The workload is one 1920x1080 frame, one lookup per pixel, the 2k texture
tiled 4x across the screen (neighbouring pixels share latent texels, as in a
rendered frame).
"""

import argparse
import json

import torch

from sixteen.decoder import CudaDecoderV2


def frame_uv(tiles=4.0):
    ys, xs = torch.meshgrid(torch.arange(1080, device="cuda"), torch.arange(1920, device="cuda"), indexing="ij")
    return torch.stack([((xs + 0.5) / 1920 * tiles) % 1, ((ys + 0.5) / 1080 * tiles) % 1], -1).view(-1, 2).contiguous()


def timed(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--time", action="store_true", help="event timing instead of single launches for ncu")
    ap.add_argument("--tiles", type=float, default=4.0, help="texture repeats across the screen (texel reuse)")
    args = ap.parse_args()
    dec = CudaDecoderV2.from_checkpoint(args.ckpt)
    uv = frame_uv(args.tiles)
    if not args.time:
        for fp4 in (False, True, False, True):  # ncu: skip the first pair (warmup), profile the second
            dec(uv, 0, fp4=fp4)
            dec(uv, 1, fp4=fp4)
        torch.cuda.synchronize()
        return
    r = {"ckpt": args.ckpt, "tiles": args.tiles, "gpu": torch.cuda.get_device_name()}
    for rep in range(3):
        r[f"fp16_unpack_ms_{rep}"] = timed(lambda: dec(uv, 0, fp4=False))
        r[f"fp4_direct_ms_{rep}"] = timed(lambda: dec(uv, 0, fp4=True))
    f16 = min(r[f"fp16_unpack_ms_{i}"] for i in range(3))
    f4 = min(r[f"fp4_direct_ms_{i}"] for i in range(3))
    r.update(fp16_unpack_ms=f16, fp4_direct_ms=f4, speedup=f16 / f4, fp4_us_per_mpix=f4 * 1e3 / (1920 * 1080 / 1e6))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
