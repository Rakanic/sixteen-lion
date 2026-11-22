"""Fly-through comparison videos: original textures vs decoder v2 (FP4-direct) vs decoder v1.

    pixi run python tools/bistro_videos.py --fit out/bistro-v2 --fit-v1 out/bistro-pilot --out results/renderer/neural-compare-v2

Writes (all rendered with the same camera path, jitter and TAA, 30 fps):
  wipe_original_vs_v2.mp4   a divider sweeps across one view: left original (decoded BC data), right our NVFP4
                            textures through CUDA decoder v2 (concat4 fits, FP4 block-scaled MMA, output-space STF)
  v1_vs_v2.mp4              left CUDA decoder v1 (bilinear fits, latent-space tap, FP16 MMA), right decoder v2
Materials without a fit use their original textures everywhere. v1 and v2 use different fits of the same
materials (bilinear vs concat4 model), each trained with the same loss and budget.
"""

import argparse
import math
import os
import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_bistro import CAMERAS  # noqa: E402
from sixteen.decoder import CudaDecoder, CudaDecoderV2  # noqa: E402
from sixteen.render.gbuffer import Camera  # noqa: E402
from sixteen.render.neural_ref import MixedSource, MixedSourceV2, OverrideStore, TensorSet, load_all_fits  # noqa: E402
from sixteen.render.renderer import TAA, Renderer, jitter_at  # noqa: E402
from sixteen.render.scene import load_scene  # noqa: E402
from sixteen.render.shade import to_display  # noqa: E402
from sixteen.render.sources import AtlasSource  # noqa: E402
from sixteen.render.textures import TextureAtlas  # noqa: E402

# keyframes (eye, target) inside the interior; smoothstep between them
KEYS = [((6.5, 1.7, 4.5), (4.5, 1.6, -8.0)), ((5.0, 1.6, -1.0), (9.0, 1.5, -9.0)), ((1.0, 1.7, -2.0), (9.0, 1.5, -9.0))]


def path_cam(t):
    seg = min(int(t * (len(KEYS) - 1)), len(KEYS) - 2)
    u = t * (len(KEYS) - 1) - seg
    u = u * u * (3 - 2 * u)
    (e0, g0), (e1, g1) = KEYS[seg], KEYS[seg + 1]
    mix = lambda a, b: tuple(x * (1 - u) + y * u for x, y in zip(a, b))
    return Camera(eye=mix(e0, e1), target=mix(g0, g1), fovy=70, far=100)


def label(img, text, xy, size=26):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = ImageFont.load_default(size=size)
    x, y = xy
    box = d.textbbox((x, y), text, font=f)
    d.rectangle((box[0] - 8, box[1] - 5, box[2] + 8, box[3] + 5), fill=(0, 0, 0))
    d.text((x, y), text, fill=(255, 255, 255), font=f)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="out/bistro-v2", help="concat4 + NVFP4 weights fits (decoder v2)")
    ap.add_argument("--fit-v1", default="out/bistro-pilot", help="bilinear fits (decoder v1)")
    ap.add_argument("--out", default="results/renderer/neural-compare-v2")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--ambient", type=float, default=0.3)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev, W, H = "cuda", 1280, 720

    scene = load_scene("BistroInterior")
    r = Renderer(scene)
    fits, fits1 = load_all_fits(scene, a.fit, dev), load_all_fits(scene, a.fit_v1, dev)
    ref_over = {f["index"]: TensorSet(f["target"], dev) for f in fits}
    atlas_ref = TextureAtlas(OverrideStore(r.store, ref_over))
    s_ref = AtlasSource(atlas_ref, "stf")
    s_v2 = MixedSourceV2(AtlasSource(atlas_ref, "stf"), {f["index"]: CudaDecoderV2.from_checkpoint(f["ckpt"]) for f in fits})
    s_v1 = MixedSource(AtlasSource(atlas_ref, "stf"), {f["index"]: CudaDecoder.from_checkpoint(f["ckpt"]) for f in fits1})
    print(f"{len(fits)} v2 fits, {len(fits1)} v1 fits", flush=True)

    sources = {"ref": s_ref, "v2": s_v2, "v1": s_v1}
    taa = {k: TAA(alpha=0.1) for k in sources}
    frames = {"wipe": [], "v1": []}
    for f in range(a.frames):
        cam = path_cam(f / (a.frames - 1))
        img = {}
        for k, src in sources.items():
            o = r.render(src, cam, W, H, frame=f, jitter=jitter_at(f), ambient=a.ambient)
            img[k] = (to_display(taa[k].resolve(o["rgb"], o["gb"], cam, W, H)).flip(0).clamp(0, 1) * 255).round().byte().cpu().numpy()
        # wipe: divider sweeps once across the frame and back
        x = int(W * (0.5 + 0.42 * math.sin(2 * math.pi * f / a.frames)))
        w = np.concatenate([img["ref"][:, :x], img["v2"][:, x:]], 1).copy()
        w[:, max(x - 1, 0) : x + 1] = 255
        w = label(w, "Original (decoded BC textures)", (16, 14))
        w = label(w, "Ours: NVFP4, decoder v2", (W - 440, 14))
        frames["wipe"].append(w)
        half = lambda im: np.asarray(Image.fromarray(im).resize((960, 540), Image.LANCZOS))
        v = np.concatenate([half(img["v1"]), half(img["v2"])], 1)
        v = label(v, "Decoder v1 (FP16 MMA, latent-space tap)", (12, 10), 22)
        v = label(v, "Decoder v2 (FP4-direct, output-space STF)", (972, 10), 22)
        frames["v1"].append(v)
        if f % 20 == 0:
            print(f"frame {f}/{a.frames}", flush=True)

    for name, fn in (("wipe", "wipe_original_vs_v2.mp4"), ("v1", "v1_vs_v2.mp4")):
        w = imageio.get_writer(out / fn, fps=30, codec="libx264", pixelformat="yuv420p", macro_block_size=1, ffmpeg_params=["-crf", "27", "-preset", "slow"])
        for fr in frames[name]:
            w.append_data(fr)
        w.close()
        print(fn, round((out / fn).stat().st_size / 1e6, 2), "MB")
    # a still from the middle of each, for the page poster and for checking
    Image.fromarray(frames["wipe"][a.frames // 4]).save(out / "wipe_poster.jpg", quality=88)
    Image.fromarray(frames["v1"][a.frames // 2]).save(out / "v1_poster.jpg", quality=88)


if __name__ == "__main__":
    main()
