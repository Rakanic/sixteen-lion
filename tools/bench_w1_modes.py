"""Time decoder v2 layer-1 modes on one 1080p frame (interleaved, best launch setting per mode).

    pixi run python tools/bench_w1_modes.py [--out results/decoder/w1_modes.json]

Modes: fp16 (FP16-unpack), fp4 (FP4-direct, NVFP4 weights), fp4x2 (FP4-direct,
two-term NVFP4 weights, 2 MMAs per grid), fp4xfp8 (FP4 latents x FP8 weights,
m16n8k32 kind::mxf8f6f4; timing only, values not exact). Timing does not depend
on the weight values, so random models are used.
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
    runs = {(mode, mb): (d2 if mode == "fp4x2" else d1) for mode in ("fp16", "fp4", "fp4x2", "fp4xfp8") for mb in (1, 4)}
    t = {k: [] for k in runs}
    for _ in range(args.reps):  # interleave so load from other users hits all modes alike
        for (mode, mb), dec in runs.items():
            t[(mode, mb)].append(timed(lambda: dec(uv, 0, mode=mode, min_blocks=mb, blocks=sm * 4)))
    med = {f"{m}_mb{mb}": statistics.median(v) for (m, mb), v in t.items()}
    best = {m: min(med[f"{m}_mb1"], med[f"{m}_mb4"]) for m in ("fp16", "fp4", "fp4x2", "fp4xfp8")}
    r = {"gpu": torch.cuda.get_device_name(), "frame": "1920x1080, tiles 4", "median_ms": med, "best_ms": best,
         "speedup_over_fp16": {m: best["fp16"] / v for m, v in best.items()}}
    pathlib.Path(args.out).write_text(json.dumps(r, indent=1))
    print(json.dumps({"best_ms": {k: round(v, 4) for k, v in best.items()},
                      "speedup_over_fp16": {k: round(v, 3) for k, v in r["speedup_over_fp16"].items()}}, indent=1))


if __name__ == "__main__":
    main()
