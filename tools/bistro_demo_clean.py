"""Clean demo videos: the image is the point, text is small and translucent.

    # three vertical strips of one continuous frame: original | one-term | two-term (decoder v2, CUDA)
    pixi run python tools/bistro_demo_clean.py strips --fit-one out/bistro-v2 --fit-two out/bistro-v2x2 --out results/renderer/demo-clean
    # original on the left of a sweeping divider, ours (two-term) on the right
    pixi run python tools/bistro_demo_clean.py wipe --fit-two out/bistro-v2x2 --out results/renderer/demo-clean

1920x1080, 30 fps, 1350 frames along the BistroInterior path of tools/bistro_demo.py. Same camera, jitter, tap seeds and TAA for every
render. Overlays: short labels in small translucent pills, and one faint line at the bottom with the memory of the ten neural materials and
FLIP (mean over the video so far, over neural pixels; evaluated every `--flip-every` frames). Everything else (speedups, per-material
numbers) lives in the paper and the artifact, not on the video.
"""

import argparse
import json
import math
import os
import pathlib
import sys

import flip_evaluator as flip
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bistro_demo import font, path_cam  # noqa: E402
from sixteen.decoder import CudaDecoderV2  # noqa: E402
from sixteen.render.neural_ref import MixedSourceV2, OverrideStore, TensorSet, load_all_fits  # noqa: E402
from sixteen.render.renderer import TAA, Renderer, jitter_at  # noqa: E402
from sixteen.render.scene import load_scene  # noqa: E402
from sixteen.render.shade import to_display  # noqa: E402
from sixteen.render.sources import AtlasSource  # noqa: E402
from sixteen.render.textures import TextureAtlas  # noqa: E402


def pill(layer, text, xy, size=18, alpha_bg=95, alpha_fg=225, anchor="la"):
    d = ImageDraw.Draw(layer)
    f = font(size)
    box = d.textbbox(xy, text, font=f, anchor=anchor)
    d.rounded_rectangle((box[0] - 9, box[1] - 5, box[2] + 9, box[3] + 5), radius=9, fill=(0, 0, 0, alpha_bg))
    d.text(xy, text, font=f, fill=(255, 255, 255, alpha_fg), anchor=anchor)


def faint(layer, text, xy, size=15, alpha=150, anchor="ls"):
    d = ImageDraw.Draw(layer)
    f = font(size)
    d.text((xy[0] + 1, xy[1] + 1), text, font=f, fill=(0, 0, 0, alpha // 2), anchor=anchor)  # soft shadow for legibility
    d.text(xy, text, font=f, fill=(255, 255, 255, alpha), anchor=anchor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["strips", "wipe"])
    ap.add_argument("--fit-one", default="out/bistro-v2")
    ap.add_argument("--fit-two", default="out/bistro-v2x2")
    ap.add_argument("--out", default="results/renderer/demo-clean")
    ap.add_argument("--frames", type=int, default=1350)
    ap.add_argument("--ambient", type=float, default=0.3)
    ap.add_argument("--flip-every", type=int, default=10)
    ap.add_argument("--crf", type=int, default=29)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev, W, H = "cuda", 1920, 1080

    scene = load_scene("BistroInterior")
    r = Renderer(scene)
    fits2 = load_all_fits(scene, a.fit_two, dev)
    fits1 = load_all_fits(scene, a.fit_one, dev) if a.mode == "strips" else []
    stems = {f["stem"] for f in fits2} & ({f["stem"] for f in fits1} if fits1 else {f["stem"] for f in fits2})
    fits2 = [f for f in fits2 if f["stem"] in stems]
    fits1 = [f for f in fits1 if f["stem"] in stems]
    atlas = TextureAtlas(OverrideStore(r.store, {f["index"]: TensorSet(f["target"], dev) for f in fits2}))
    mk = lambda fits: MixedSourceV2(AtlasSource(atlas, "stf"), {f["index"]: CudaDecoderV2.from_checkpoint(f["ckpt"]) for f in fits})
    src = dict(ref=AtlasSource(atlas, "stf"), two=mk(fits2))
    if a.mode == "strips":
        src["one"] = mk(fits1)
    fitted = torch.zeros(len(scene.materials), dtype=torch.bool, device=dev)
    fitted[[f["index"] for f in fits2]] = True
    cmp_ = json.load(open("results/renderer/neural-compare-v2/neural_compare.json"))
    bc_mib, our_mib = cmp_["bc_bytes_top_level_total"] / 2**20, cmp_["neural_bytes_total"] / 2**20
    print(f"{a.mode}: {len(stems)} neural materials", flush=True)

    taa = {k: TAA(alpha=0.1) for k in src}
    to8 = lambda t: (t.flip(0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    name = "demo_clean_strips.mp4" if a.mode == "strips" else "demo_clean_wipe.mp4"
    wr = imageio.get_writer(out / name, fps=30, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                            ffmpeg_params=["-crf", str(a.crf), "-preset", "medium"])
    keys = [k for k in ("one", "two") if k in src]
    ssum, sn, log = {k: 0.0 for k in keys}, 0, []
    for f in range(a.frames):
        cam = path_cam(f / (a.frames - 1))
        d = {}
        for k, s in src.items():
            o = r.render(s, cam, W, H, frame=f, jitter=jitter_at(f), ambient=a.ambient)
            d[k] = to_display(taa[k].resolve(o["rgb"], o["gb"], cam, W, H)).clamp(0, 1)
            if k == "ref":
                gb = o["gb"]
        if f % a.flip_every == 0 and f >= 8:
            mask = gb["valid"] & fitted[gb["mat"].clamp(min=0)]
            if mask.any():
                sn += 1
                for k in keys:
                    e, _, _ = flip.evaluate(d["ref"].cpu().numpy(), d[k].cpu().numpy(), "LDR", applyMagma=False)
                    ssum[k] += float(torch.from_numpy(np.asarray(e)[..., 0]).to(dev)[mask].mean())
        run = {k: (ssum[k] / sn if sn else None) for k in keys}
        log.append(dict(frame=f, **{f"run_{k}": run[k] for k in keys}))
        i = {k: to8(v) for k, v in d.items()}
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if a.mode == "strips":
            img = np.concatenate([i["ref"][:, :640], i["one"][:, 640:1280], i["two"][:, 1280:]], 1).copy()
            for x in (640, 1280):
                img[:, x - 1 : x + 1] = 255
            pill(layer, "Original", (24, 22))
            pill(layer, "One-term weights", (664, 22))
            pill(layer, "Two-term weights", (1304, 22))
            faint(layer, f"{bc_mib:.0f} MiB BC", (24, H - 22))
            faint(layer, f"{our_mib:.0f} MiB" + (f"   FLIP {run['one']:.3f}" if sn else ""), (664, H - 22))
            faint(layer, f"{our_mib:.0f} MiB" + (f"   FLIP {run['two']:.3f}" if sn else ""), (1304, H - 22))
        else:
            x = int(W * (0.5 + 0.42 * math.sin(2 * math.pi * f / a.frames * 3)))
            img = np.concatenate([i["ref"][:, :x], i["two"][:, x:]], 1).copy()
            img[:, max(x - 1, 0) : x + 1] = 255
            pill(layer, "Original", (24, 22))
            pill(layer, "Neural", (W - 24, 22), anchor="ra")
            faint(layer, f"{our_mib:.0f} MiB neural textures vs {bc_mib:.0f} MiB BC" + (f"   FLIP {run['two']:.3f}" if sn else ""), (24, H - 22))
        frame = Image.alpha_composite(Image.fromarray(img).convert("RGBA"), layer).convert("RGB")
        wr.append_data(np.asarray(frame))
        if f % 150 == 0:
            print(f"frame {f}/{a.frames}", flush=True)
    wr.close()
    (out / (name[:-4] + "_metrics.json")).write_text(json.dumps(dict(mode=a.mode, frames=a.frames, flip_every=a.flip_every,
        final_running_mean={k: (ssum[k] / sn if sn else None) for k in keys}, bc_mib=bc_mib, ours_mib=our_mib)))
    print(name, round((out / name).stat().st_size / 1e6, 1), "MB", {k: (round(ssum[k] / sn, 4) if sn else None) for k in keys})


if __name__ == "__main__":
    main()
