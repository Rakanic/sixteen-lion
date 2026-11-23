"""Decoder v2 time per material and per viewing distance, on one 1920x1080 frame (one lookup per pixel).

    pixi run python tools/bench_decode_distance.py --fit out/bistro-v2 [--out results/decoder/decode_time_distance.json]

"Distance" is the texel-to-pixel ratio m: how many texels one pixel step crosses. m = 1 is one texel per pixel, m = 8
is a surface 8x farther away (or a texture 8x denser), m = 0.25 is a close-up. The kernel only reads texture level 0,
so far-away surfaces scatter the latent fetches; this table shows what that costs. Pixel (x, y) reads
uv = ((x + 0.5) m / 2048 + 0.31, (y + 0.5) m / 2048 + 0.17) mod 1, a flat surface facing the camera (no grazing angles).
The "typical m" column is the median (p90) texels-per-pixel of that material in the BistroInterior/entrance view, from the G-buffer.

Timing: CUDA events over `--iters` launches, median over `--reps` repetitions, all cases interleaved so load from other
jobs on the GPU hits every case alike. Run on an idle GPU for paper numbers; the JSON records the GPU utilisation seen.
"""

import argparse
import glob
import json
import pathlib
import statistics
import subprocess

import torch

from sixteen.decoder import CudaDecoderV2

M_VALUES = [0.25, 0.5, 1, 2, 4, 8, 16]


def grid_uv(m, tex=2048):
    ys, xs = torch.meshgrid(torch.arange(1080, device="cuda"), torch.arange(1920, device="cuda"), indexing="ij")
    return torch.stack([((xs + 0.5) * m / tex + 0.31) % 1, ((ys + 0.5) * m / tex + 0.17) % 1], -1).view(-1, 2).contiguous()


def timed(fn, iters, warmup=5):
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


def smi():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"], text=True).strip().splitlines()
    except Exception:
        return []


def typical_m(stems):
    """Median and p90 texels per pixel of each material in the BistroInterior entrance view."""
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from render_bistro import CAMERAS
    from sixteen.render.gbuffer import rasterize_gbuffer
    from sixteen.render.neural_ref import stem_of
    from sixteen.render.renderer import Renderer
    from sixteen.render.scene import load_scene

    sc = load_scene("BistroInterior")
    r = Renderer(sc)
    gb = rasterize_gbuffer(r.ctx, r.sg, CAMERAS["BistroInterior"]["entrance"], 1920, 1080)
    d = gb["duv"].reshape(-1, 4)  # du/dx, du/dy, dv/dx, dv/dy
    rho = torch.maximum(torch.hypot(d[:, 0], d[:, 2]), torch.hypot(d[:, 1], d[:, 3])) * 2048
    mat, valid = gb["mat"].reshape(-1), gb["valid"].reshape(-1)
    out = {}
    for i, m in enumerate(sc.materials):
        if m.base_color and stem_of(m) in stems:
            sel = valid & (mat == i)
            if sel.any():
                v = rho[sel]
                out[stem_of(m)] = (float(v.median()), float(v.quantile(0.9)), float(sel.float().mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="out/bistro-v2")
    ap.add_argument("--out", default="results/decoder/decode_time_distance.json")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()
    cks = sorted(glob.glob(f"{a.fit}/bistro/*/m/shade4/nvfp4/model.pt"))
    decs = {pathlib.Path(c).parts[-5]: CudaDecoderV2.from_checkpoint(c) for c in cks}
    stems = list(decs)
    terms = {d.terms for d in decs.values()}
    assert len(terms) == 1, "mixed one- and two-term fits"
    fp4mode = "fp4x2" if terms == {2} else "fp4"
    modes = ["fp16", fp4mode]
    uvs = {m: grid_uv(m) for m in M_VALUES}
    cases = [(s, m, mode) for s in stems for m in M_VALUES for mode in modes]
    util0 = smi()
    t = {c: [] for c in cases}
    for rep in range(a.reps):
        for c in cases:
            s, m, mode = c
            t[c].append(timed(lambda: decs[s](uvs[m], 0, mode=mode), a.iters))
    util1 = smi()
    med = {c: statistics.median(v) for c, v in t.items()}
    tm = typical_m(set(stems))
    res = dict(fit=a.fit, gpu=torch.cuda.get_device_name(), iters=a.iters, reps=a.reps, m_values=M_VALUES, modes=modes,
               gpu_util_before=util0, gpu_util_after=util1,
               ms={f"{s}|{m}|{mode}": med[(s, m, mode)] for (s, m, mode) in cases},
               ms_spread={f"{s}|{m}|{mode}": (min(t[(s, m, mode)]), max(t[(s, m, mode)])) for (s, m, mode) in cases},
               typical_m_entrance={s: dict(median=v[0], p90=v[1], pixel_share=v[2]) for s, v in tm.items()})
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    short = lambda n: n.replace("MASTER_Interior_01_", "").replace("MASTER_", "").replace("_BLENDSHADER", "")
    for mode in modes:
        print(f"\ndecode ms per 1080p frame, mode {mode}  (columns: texels per pixel m)")
        print("| material | typical m (p90) | " + " | ".join(f"{m:g}" for m in M_VALUES) + " |")
        print("|---|---|" + "---|" * len(M_VALUES))
        for s in stems:
            tmv = tm.get(s)
            print(f"| {short(s)} | " + (f"{tmv[0]:.1f} ({tmv[1]:.1f})" if tmv else "-") + " | " + " | ".join(f"{med[(s, m, mode)]:.3f}" for m in M_VALUES) + " |")
        print("| **median over materials** | | " + " | ".join(f"**{statistics.median(med[(s, m, mode)] for s in stems):.3f}**" for m in M_VALUES) + " |")
    print("\nGPU util before/after:", util0, util1)


if __name__ == "__main__":
    main()
