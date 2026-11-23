"""Time decoder v2 layer-1 modes on one 1080p frame (interleaved, best launch setting per mode).

    pixi run python tools/bench_w1_modes.py [--out results/decoder/w1_modes.json]

Modes: fp16 (FP16-unpack), fp4 (FP4-direct, NVFP4 weights), fp4x2 (FP4-direct,
two-term NVFP4 weights, 2 MMAs per grid); each with and without prefetching the
next tile's latents. Timing does not depend on the weight values, so random
models are used. (An FP4 latents x FP8 weights mode, m16n8k32 kind::mxf8f6f4,
was timed in commit c7a2461 at 1.18x over FP16-unpack and then removed.)
"""

import argparse
import json
import pathlib
import statistics
import sys

import torch

from sixteen.decoder import CudaDecoderV2
from sixteen.model import NeuralTexture

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from profile_decoder import frame_uv, timed  # noqa: E402


def model(w1):
    torch.manual_seed(0)
    m = NeuralTexture((2048, 2048), 9, filter="concat4", w1_fmt=w1)
    with torch.no_grad():
        for g in m.latents:
            g.normal_(0, 0.7)
    return CudaDecoderV2(m, (2048, 2048))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/decoder/w1_modes.json")
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()
    d1, d2 = model("nvfp4"), model("nvfp4x2")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    uv = frame_uv(4.0)
    modes = ("fp16", "fp4", "fp4x2")
    runs = {(m, mb, pf): (d2 if m == "fp4x2" else d1) for m in modes for mb in (1, 2, 4) for pf in (0, 1)}
    t = {k: [] for k in runs}
    for _ in range(args.reps):  # interleave so load from other users hits all modes alike
        for (m, mb, pf), dec in runs.items():
            t[(m, mb, pf)].append(timed(lambda: dec(uv, 0, mode=m, min_blocks=mb, blocks=sm * 4, prefetch=pf)))
    med = {f"{m}_mb{mb}_pf{pf}": statistics.median(v) for (m, mb, pf), v in t.items()}
    best = {f"{m}_pf{pf}": min(v for k, v in med.items() if k.startswith(m + "_") and k.endswith(f"pf{pf}"))
            for m in modes for pf in (0, 1)}
    r = {"gpu": torch.cuda.get_device_name(), "frame": "1920x1080, tiles 4", "median_ms": med, "best_ms": best,
         "speedup_over_fp16_same_prefetch": {m: best["fp16_" + m.split("_")[-1]] / v for m, v in best.items()}}
    pathlib.Path(args.out).write_text(json.dumps(r, indent=1))
    print(json.dumps({"best_ms": {k: round(v, 4) for k, v in best.items()},
                      "speedup": {k: round(v, 3) for k, v in r["speedup_over_fp16_same_prefetch"].items()}}, indent=1))


if __name__ == "__main__":
    main()
