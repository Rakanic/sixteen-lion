"""Longer demo videos of Bistro with the original textures against ours (decoder v2), with live metrics.

    pixi run python tools/bistro_demo.py --fit out/bistro-v2 --out results/renderer/demo --frames 1350

Both sides use the same camera path, jitter, tap seeds and TAA, so differences are compression and decoding only.
Writes (30 fps):
  demo_wipe.mp4         1280x720. A divider sweeps across the view: left the original (decoded BC textures), right ours
                        (NVFP4 latents through CUDA decoder v2). A strip shows the memory of the ten neural materials
                        (BC vs ours), the share of pixels that are neural, and FLIP of the current frame.
  demo_error_view.mp4   1920x1080. Original | ours | FLIP map (colorbar; gray = not neural) | live numbers
                        (current and running-mean FLIP over neural pixels, memory, neural-pixel share).
FLIP is evaluated every `--flip-every` frames (flip_evaluator, "LDR", default viewing conditions) and held in between.
FLIP numbers here are over neural pixels only; pixels of materials we did not fit are identical by construction.
"""

import argparse
import json
import math
import os
import pathlib
import sys

import flip_evaluator as flip
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sixteen.decoder import CudaDecoderV2  # noqa: E402
from sixteen.render.gbuffer import Camera  # noqa: E402
from sixteen.render.neural_ref import MixedSourceV2, OverrideStore, TensorSet, load_all_fits  # noqa: E402
from sixteen.render.renderer import TAA, Renderer, jitter_at  # noqa: E402
from sixteen.render.scene import load_scene  # noqa: E402
from sixteen.render.shade import to_display  # noqa: E402
from sixteen.render.sources import AtlasSource  # noqa: E402
from sixteen.render.textures import TextureAtlas  # noqa: E402

# keyframes (eye, target), all inside the interior (checked on a contact sheet); smoothstep between them
KEYS = [
    ((6.5, 1.7, 4.5), (4.5, 1.6, -8.0)),   # entrance, bar in the distance
    ((4.0, 1.7, 1.0), (8.5, 1.4, -3.0)),   # tables and the right wall
    ((1.0, 1.7, -2.0), (9.0, 1.5, -9.0)),  # bar
    ((6.0, 1.6, -7.0), (-2.0, 1.8, -2.0)),  # side wall, curtains, door
    ((6.5, 1.7, 4.5), (4.5, 1.6, -8.0)),   # back to the start (loops)
]


def path_cam(t):
    n = len(KEYS) - 1
    seg = min(int(t * n), n - 1)
    u = t * n - seg
    u = u * u * (3 - 2 * u)
    (e0, g0), (e1, g1) = KEYS[seg], KEYS[seg + 1]
    mix = lambda a, b: tuple(x * (1 - u) + y * u for x, y in zip(a, b))
    return Camera(eye=mix(e0, e1), target=mix(g0, g1), fovy=70, far=100)


def font(size):
    return ImageFont.load_default(size=size)


def tag(im, text, xy, size=24, bg=(0, 0, 0)):
    d = ImageDraw.Draw(im)
    f = font(size)
    box = d.textbbox(xy, text, font=f)
    d.rectangle((box[0] - 8, box[1] - 5, box[2] + 8, box[3] + 5), fill=bg)
    d.text(xy, text, fill=(255, 255, 255), font=f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="out/bistro-v2")
    ap.add_argument("--out", default="results/renderer/demo")
    ap.add_argument("--frames", type=int, default=1350)
    ap.add_argument("--ambient", type=float, default=0.3)
    ap.add_argument("--flip-every", type=int, default=5)
    ap.add_argument("--crf", type=int, default=28)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev, W, H = "cuda", 1280, 720

    scene = load_scene("BistroInterior")
    r = Renderer(scene)
    fits = load_all_fits(scene, a.fit, dev)
    ref_over = {f["index"]: TensorSet(f["target"], dev) for f in fits}
    atlas = TextureAtlas(OverrideStore(r.store, ref_over))
    s_ref = AtlasSource(atlas, "stf")
    s_neu = MixedSourceV2(AtlasSource(atlas, "stf"), {f["index"]: CudaDecoderV2.from_checkpoint(f["ckpt"]) for f in fits})
    fitted = torch.zeros(len(scene.materials), dtype=torch.bool, device=dev)
    fitted[[f["index"] for f in fits]] = True
    cmp_ = json.load(open("results/renderer/neural-compare-v2/neural_compare.json"))
    mib = 2**20
    bc_mib, our_mib = cmp_["bc_bytes_top_level_total"] / mib, cmp_["neural_bytes_total"] / mib
    print(f"{len(fits)} neural materials; BC {bc_mib:.1f} MiB, ours {our_mib:.1f} MiB", flush=True)

    cm = matplotlib.colormaps["magma"]
    taa = {"ref": TAA(alpha=0.1), "neu": TAA(alpha=0.1)}
    to8 = lambda t: (t.flip(0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    wipe_w = imageio.get_writer(out / "demo_wipe.mp4", fps=30, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                                ffmpeg_params=["-crf", str(a.crf), "-preset", "medium"])
    err_w = imageio.get_writer(out / "demo_error_view.mp4", fps=30, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                               ffmpeg_params=["-crf", str(a.crf), "-preset", "medium"])
    flip_now, flip_sum, flip_n, err_rgb, log = float("nan"), 0.0, 0, None, []
    for f in range(a.frames):
        cam = path_cam(f / (a.frames - 1))
        o_ref = r.render(s_ref, cam, W, H, frame=f, jitter=jitter_at(f), ambient=a.ambient)
        o_neu = r.render(s_neu, cam, W, H, frame=f, jitter=jitter_at(f), ambient=a.ambient)
        d_ref = to_display(taa["ref"].resolve(o_ref["rgb"], o_ref["gb"], cam, W, H)).clamp(0, 1)
        d_neu = to_display(taa["neu"].resolve(o_neu["rgb"], o_neu["gb"], cam, W, H)).clamp(0, 1)
        gb = o_ref["gb"]
        mask = gb["valid"] & fitted[gb["mat"].clamp(min=0)]
        share = float(mask.sum()) / max(float(gb["valid"].sum()), 1.0)
        if f % a.flip_every == 0 and f >= 8:  # let TAA history settle
            e, _, _ = flip.evaluate(d_ref.cpu().numpy(), d_neu.cpu().numpy(), "LDR", applyMagma=False)
            e = torch.from_numpy(np.asarray(e)[..., 0] if np.asarray(e).ndim == 3 else np.asarray(e)).to(dev)
            if mask.any():
                flip_now = float(e[mask].mean())
                flip_sum, flip_n = flip_sum + flip_now, flip_n + 1
            err_rgb = torch.from_numpy(cm(e.clamp(0, 0.2).cpu().numpy() / 0.2)[..., :3]).float().to(dev)
            err_rgb[~mask] = 0.5
        i_ref, i_neu = to8(d_ref), to8(d_neu)
        run = flip_sum / flip_n if flip_n else float("nan")
        log.append(dict(frame=f, flip_neural_pixels=flip_now, flip_running_mean=run, neural_pixel_share=share))

        # ---- wipe video (1280x720) with a metrics strip
        x = int(W * (0.5 + 0.42 * math.sin(2 * math.pi * f / a.frames * 3)))  # three sweeps over the whole path
        w = np.concatenate([i_ref[:, :x], i_neu[:, x:]], 1).copy()
        w[:, max(x - 1, 0) : x + 1] = 255
        im = Image.fromarray(w)
        tag(im, "Original textures (BC)", (16, 14))
        tag(im, "Ours: NVFP4 neural textures", (W - 400, 14))
        d = ImageDraw.Draw(im)
        d.rectangle((0, H - 74, W, H), fill=(0, 0, 0))
        f28, f22 = font(26), font(21)
        bw = 330
        d.text((16, H - 66), f"BC: {bc_mib:.1f} MiB", fill=(255, 255, 255), font=f22)
        d.rectangle((190, H - 62, 190 + bw, H - 46), fill=(150, 150, 150))
        d.text((16, H - 38), f"Ours: {our_mib:.1f} MiB", fill=(255, 255, 255), font=f22)
        d.rectangle((190, H - 34, 190 + bw * our_mib / bc_mib, H - 18), fill=(70, 130, 210))
        d.text((560, H - 66), "10 of 71 materials are neural", fill=(200, 200, 200), font=f22)
        d.text((560, H - 38), f"{share * 100:.0f} % of pixels in view", fill=(200, 200, 200), font=f22)
        d.text((920, H - 66), "FLIP, neural pixels", fill=(200, 200, 200), font=f22)
        d.text((920, H - 38), "..." if flip_n == 0 else f"{flip_now:.4f}  (mean {run:.4f})", fill=(255, 255, 255), font=f22)
        wipe_w.append_data(np.asarray(im))

        # ---- error-view video (1920x1080): original | ours / FLIP map | numbers
        cw, ch = 960, 540
        half = lambda arr: Image.fromarray(arr).resize((cw, ch), Image.LANCZOS)
        canvas = Image.new("RGB", (2 * cw, 2 * ch), (255, 255, 255))
        canvas.paste(half(i_ref), (0, 0))
        canvas.paste(half(i_neu), (cw, 0))
        if err_rgb is not None:
            emap = to8(err_rgb)
        else:
            emap = np.full((H, W, 3), 128, np.uint8)
        canvas.paste(half(emap), (0, ch))
        tag(canvas, "Original (decoded BC textures)", (14, 12), 26)
        tag(canvas, "Ours (NVFP4, CUDA decoder v2)", (cw + 14, 12), 26)
        tag(canvas, "FLIP error (gray: not neural)", (14, ch + 12), 26)
        d = ImageDraw.Draw(canvas)
        fb, fm, fs = font(40), font(30), font(24)
        x0, y0 = cw + 40, ch + 30
        d.text((x0, y0), "Live numbers", fill=(0, 0, 0), font=fb)
        d.text((x0, y0 + 70), "FLIP, neural pixels, this frame", fill=(90, 90, 90), font=fs)
        d.text((x0, y0 + 100), "..." if flip_n == 0 else f"{flip_now:.4f}", fill=(0, 0, 0), font=fb)
        d.text((x0, y0 + 160), "FLIP, mean over the video so far", fill=(90, 90, 90), font=fs)
        d.text((x0, y0 + 190), "..." if flip_n == 0 else f"{run:.4f}", fill=(0, 0, 0), font=fb)
        d.text((x0, y0 + 250), f"Memory of the 10 neural materials: {our_mib:.1f} MiB vs {bc_mib:.1f} MiB BC", fill=(0, 0, 0), font=fs)
        d.text((x0, y0 + 285), f"Neural pixels in view: {share * 100:.0f} %", fill=(0, 0, 0), font=fs)
        grad = (cm(np.linspace(0, 1, 500))[:, :3] * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(np.repeat(grad[None], 26, 0)), (x0, y0 + 345))
        d.text((x0, y0 + 376), "0 (identical)", fill=(0, 0, 0), font=font(20))
        d.text((x0 + 410, y0 + 376), "0.2", fill=(0, 0, 0), font=font(20))
        err_w.append_data(np.asarray(canvas))
        if f % 100 == 0:
            print(f"frame {f}/{a.frames}  flip {flip_now:.4f}  running {run:.4f}  neural share {share:.2f}", flush=True)
    wipe_w.close()
    err_w.close()
    vals = [l["flip_neural_pixels"] for l in log if l["flip_neural_pixels"] == l["flip_neural_pixels"]]
    summary = dict(frames=a.frames, size=[W, H], flip_every=a.flip_every, flip_samples=len(vals), flip_mean=float(np.mean(vals)),
                   flip_min=float(np.min(vals)), flip_max=float(np.max(vals)), bc_mib=bc_mib, ours_mib=our_mib,
                   neural_share_mean=float(np.mean([l["neural_pixel_share"] for l in log])), per_frame=log)
    (out / "demo_metrics.json").write_text(json.dumps(summary))
    for n in ("demo_wipe.mp4", "demo_error_view.mp4"):
        print(n, round((out / n).stat().st_size / 1e6, 1), "MB")
    print({k: v for k, v in summary.items() if k != "per_frame"})


if __name__ == "__main__":
    main()
