"""Stage-1 renderer measurements on a real scene: stage timing and convergence of the stochastic path.

    LD_LIBRARY_PATH=$CONDA_PREFIX/lib pixi run python tools/renderer_check.py --scene BistroInterior --camera entrance

Writes results/renderer/stage1_<scene>_<camera>.json. Timing is wall-clock with a device sync per stage
and is only meaningful on an idle GPU: the GPU utilization seen at start is recorded in the file.
"""

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

import flip_evaluator as flip
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from render_bistro import CAMERAS  # noqa: E402
from sixteen.render.gbuffer import Camera
from sixteen.render.renderer import TAA, Renderer, jitter_at
from sixteen.render.scene import load_scene
from sixteen.render.shade import to_display
from sixteen.render.sources import AtlasSource
from sixteen.render.textures import TextureAtlas

ROOT = pathlib.Path(__file__).resolve().parent.parent


def gpu_util():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [l.strip() for l in out.strip().splitlines()]


def psnr(a, b):
    return float(10 * torch.log10(1.0 / (a - b).pow(2).mean().clamp(min=1e-12)))


def score(test, ref):
    a, b = to_display(test).clamp(0, 1), to_display(ref).clamp(0, 1)
    _, mean_flip, _ = flip.evaluate(b.cpu().numpy(), a.cpu().numpy(), "LDR")
    return {"flip": float(mean_flip), "psnr": psnr(a, b)}


def lerp_cam(a, b, t):
    mix = lambda p, q: tuple(x * (1 - t) + y * t for x, y in zip(p, q))
    return Camera(eye=mix(a.eye, b.eye), target=mix(a.target, b.target), fovy=a.fovy, near=a.near, far=a.far)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="BistroInterior")
    ap.add_argument("--camera", default="entrance")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--path-frames", type=int, default=100)
    ap.add_argument("--ref-frames", type=int, default=256)
    args = ap.parse_args()
    W, H = map(int, args.size.split("x"))
    util = gpu_util()

    scene = load_scene(args.scene)
    cams = CAMERAS[args.scene]
    cam = cams[args.camera]
    r = Renderer(scene)
    atlas = TextureAtlas(r.store)
    tri, stf = AtlasSource(atlas, "trilinear"), AtlasSource(atlas, "stf")
    res = {"scene": args.scene, "camera": args.camera, "size": [W, H], "triangles": len(scene.tri), "materials": len(scene.materials),
           "atlas_mib": atlas.nbytes() / 2**20, "gpu_util_at_start": util, "torch": torch.__version__}

    # --- timing along a short camera path (A -> B), per stage, after warm-up
    names = list(cams)
    cam_b = cams[names[1]] if len(names) > 1 else Camera(eye=tuple(np.add(cam.eye, (1, 0, 0))), target=cam.target, fovy=cam.fovy, far=cam.far)
    for label, src in (("original", tri), ("original-stf", stf)):
        for f in range(5):
            r.render(src, cam, W, H, frame=f, jitter=jitter_at(f))
        per = {"raster": [], "texture": [], "shade": [], "total": []}
        for f in range(args.path_frames):
            c = lerp_cam(cam, cam_b, f / max(args.path_frames - 1, 1))
            ms = r.render(src, c, W, H, frame=f, jitter=jitter_at(f))["ms"]
            for k in per:
                per[k].append(ms[k])
        res[f"ms_{label}"] = {k: {"median": statistics.median(v), "p95": float(np.percentile(v, 95))} for k, v in per.items()}

    # --- convergence of the stochastic path on a static camera
    # The reference uses frames 0..ref_frames-1; every tested sequence uses frames from TEST (independent
    # jitter and STF seeds), so the error is not understated by shared samples.
    TEST = 100_000
    ref = r.accumulate(stf, cam, W, H, args.ref_frames)  # converged STF render
    res["ref_frames"] = args.ref_frames
    res["stf_vs_converged"] = {}
    for n in (1, 4, 16, 64):
        res["stf_vs_converged"][str(n)] = score(r.accumulate(stf, cam, W, H, n, start=TEST), ref)
    res["trilinear_1frame_vs_converged_stf"] = score(r.render(tri, cam, W, H)["rgb"], ref)
    res["trilinear_64_vs_converged_stf"] = score(r.accumulate(tri, cam, W, H, 64, start=TEST), ref)
    taa = TAA(alpha=0.1)
    out = None
    for f in range(TEST, TEST + 64):
        o = r.render(stf, cam, W, H, frame=f, jitter=jitter_at(f))
        out = taa.resolve(o["rgb"], o["gb"], cam, W, H)
    res["taa_64_frames_vs_converged_stf"] = score(out, ref)

    path = ROOT / "results" / "renderer" / f"stage1_{args.scene}_{args.camera}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
