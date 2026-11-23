"""Teaser frame for the paper: Bistro at 1920x1080, original on the left of a divider, ours (CUDA decoder v2) on the right.

    pixi run python tools/bistro_teaser.py --camera entrance --fit out/bistro-v2x2 --out out/teaser

Each side is the mean of --frames jittered frames (same camera, jitter and tap seeds for both). Only the fitted materials are neural;
every other pixel uses the original textures on both sides. Writes <out>/teaser_<camera>.jpg (split, small labels) and
<out>/teaser_<camera>_{original,neural}.jpg (full frames), and prints the neural-pixel share.
"""

import argparse
import os
import pathlib
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bistro_demo import font, path_cam  # noqa: E402
from render_bistro import CAMERAS  # noqa: E402
from sixteen.decoder import CudaDecoderV2  # noqa: E402
from sixteen.render.neural_ref import MixedSourceV2, OverrideStore, TensorSet, load_all_fits  # noqa: E402
from sixteen.render.renderer import Renderer  # noqa: E402
from sixteen.render.scene import load_scene  # noqa: E402
from sixteen.render.shade import to_display  # noqa: E402
from sixteen.render.sources import AtlasSource  # noqa: E402
from sixteen.render.textures import TextureAtlas  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="entrance", help="a preset in render_bistro.CAMERAS, or path:<t> for the fly-through path at t in [0,1]")
    ap.add_argument("--fit", default="out/bistro-v2x2")
    ap.add_argument("--out", default="out/teaser")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--ambient", type=float, default=0.3)
    ap.add_argument("--split", type=float, default=0.5, help="divider position as a fraction of the width")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev, W, H = "cuda", 1920, 1080
    scene = load_scene("BistroInterior")
    r = Renderer(scene)
    cam = path_cam(float(a.camera.split(":")[1])) if a.camera.startswith("path:") else CAMERAS["BistroInterior"][a.camera]
    fits = load_all_fits(scene, a.fit, dev)
    atlas = TextureAtlas(OverrideStore(r.store, {f["index"]: TensorSet(f["target"], dev) for f in fits}))
    ref = AtlasSource(atlas, "stf")
    neu = MixedSourceV2(AtlasSource(atlas, "stf"), {f["index"]: CudaDecoderV2.from_checkpoint(f["ckpt"]) for f in fits})
    img = {k: to_display(r.accumulate(s, cam, W, H, a.frames, ambient=a.ambient)).flip(0).clamp(0, 1).mul(255).round().byte().cpu().numpy() for k, s in (("original", ref), ("neural", neu))}
    gb = r.render(ref, cam, W, H)["gb"]
    fitted = torch.zeros(len(scene.materials), dtype=torch.bool, device=dev)
    fitted[[f["index"] for f in fits]] = True
    share = float((gb["valid"] & fitted[gb["mat"].clamp(min=0)]).sum()) / float(gb["valid"].sum())
    x = int(W * a.split)
    split = np.concatenate([img["original"][:, :x], img["neural"][:, x:]], 1).copy()
    split[:, x - 1 : x + 1] = 255
    im = Image.fromarray(split).convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = font(26)
    for text, xy, anchor in (("Original", (28, 26), "la"), ("Neural", (W - 28, 26), "ra")):
        box = d.textbbox(xy, text, font=f, anchor=anchor)
        d.rounded_rectangle((box[0] - 12, box[1] - 7, box[2] + 12, box[3] + 7), radius=11, fill=(0, 0, 0, 110))
        d.text(xy, text, font=f, fill=(255, 255, 255, 235), anchor=anchor)
    tag = a.camera.replace(":", "")
    Image.alpha_composite(im, layer).convert("RGB").save(out / f"teaser_{tag}.jpg", quality=93)
    for k, v in img.items():
        Image.fromarray(v).save(out / f"teaser_{tag}_{k}.jpg", quality=93)
    print(f"{tag}: neural share of the view {share * 100:.1f} %  ({len(fits)} materials)")


if __name__ == "__main__":
    main()
