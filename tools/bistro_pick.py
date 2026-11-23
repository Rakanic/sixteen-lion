"""Pick Bistro materials to fit: the eligible ones that cover the most pixels in a view.

Eligible: all three maps present, 1024+ detail in albedo or normal, no real alpha, no emissive map
(the 9-channel target cannot hold opacity or emissive). Writes a sweep config.

    pixi run python tools/bistro_pick.py --scene BistroInterior --camera entrance -n 9 --out configs/bistro-pilot.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bistro_inventory import dds_info  # noqa: E402
from render_bistro import CAMERAS  # noqa: E402
from sixteen.render.gbuffer import SceneGPU, rasterize_gbuffer  # noqa: E402
from sixteen.render.renderer import Renderer  # noqa: E402
from sixteen.render.scene import load_scene  # noqa: E402


def stem_of(m):
    return m.base_color[: -len("_BaseColor.dds")]


def eligible(scene, m):
    td = scene.texture_dir
    if not (m.base_color and m.normal and m.specular) or m.emissive:
        return False
    b, n = dds_info(td / m.base_color), dds_info(td / m.normal)
    if max(b["w"], n["w"]) < 1024:
        return False
    if b["fmt"] == "DXT5" and np.asarray(Image.open(td / m.base_color))[..., 3].min() < 255:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="BistroInterior")
    ap.add_argument("--camera", default="entrance")
    ap.add_argument("-n", type=int, default=9)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", default="configs/bistro-pilot.yaml")
    a = ap.parse_args()
    scene = load_scene(a.scene)
    r = Renderer(scene)
    gb = rasterize_gbuffer(r.ctx, r.sg, CAMERAS[a.scene][a.camera], 1920, 1080)
    mat = gb["mat"][gb["valid"]]
    counts = torch.bincount(mat, minlength=len(scene.materials)).cpu().numpy()
    total = int(gb["valid"].sum())
    order = np.argsort(-counts)
    chosen, cov = [], 0
    for i in order:
        m = scene.materials[i]
        if counts[i] and eligible(scene, m):
            chosen.append((stem_of(m), int(counts[i])))
            cov += counts[i]
            if len(chosen) == a.n:
                break
    print(f"{len(chosen)} materials cover {100 * cov / total:.1f}% of covered pixels ({100 * total / (1920 * 1080):.0f}% of the frame is geometry)")
    for s, c in chosen:
        print(f"  {s:50s} {100 * c / total:5.1f}%")
    cfg = {
        "name": "bistro-pilot",
        "train": {"steps": a.steps, "qat_start": 0.7},
        "grids": {"m": [[4, 16], [8, 16]]},
        "losses": {"shade4": {"w_texel": 1.0, "w_shading": 4.0}},
        "sweep": {"material": [f"bistro/{s}" for s, _ in chosen], "grids": ["m"], "loss": ["shade4"], "fmt": ["nvfp4"]},
    }
    with open(a.out, "w") as f:
        f.write(f"# Fit the {len(chosen)} most visible Bistro materials of {a.scene}/{a.camera} (tools/bistro_pick.py).\n")
        f.write("# NVFP4, mid grid, texel + 4x shading loss: the project's best-known 4-bit configuration.\n")
        yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == "__main__":
    main()
