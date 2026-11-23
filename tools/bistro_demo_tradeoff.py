"""Demo video of the speed-vs-quality trade-off: original | one-term FP4-direct | two-term FP4-direct on one Bistro fly-through.

    pixi run python tools/bistro_demo_tradeoff.py --fit-one out/bistro-v2 --fit-two out/bistro-v2x2 --out results/renderer/demo-tradeoff

All three renders share the camera path, jitter, tap seeds and TAA. The one-term variant stores the layer-1 weights as one NVFP4
term (fastest decoder); the two-term variant stores them as two (W ~ q(W) + q(W - q(W)), two block-scaled MMAs per latent grid),
which recovers quality at a smaller speedup. Layout (1920x1080): row 1 the three images, row 2 the FLIP maps of the two variants
(gray: pixel not neural) and a legend, bottom strip: the numbers. FLIP is over neural pixels only, evaluated every
`--flip-every` frames. Speedups are from results/decoder/w1_modes.json (idle GPU, 1080p decode), the castle_brick_07 costs from
results/w1-weights/results.json (two seeds); only the live FLIP numbers are measured on this video.
"""

import argparse
import json
import os
import pathlib
import sys

import flip_evaluator as flip
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bistro_demo import font, path_cam, tag  # noqa: E402
from sixteen.decoder import CudaDecoderV2  # noqa: E402
from sixteen.render.neural_ref import MixedSourceV2, OverrideStore, TensorSet, load_all_fits  # noqa: E402
from sixteen.render.renderer import TAA, Renderer, jitter_at  # noqa: E402
from sixteen.render.scene import load_scene  # noqa: E402
from sixteen.render.shade import to_display  # noqa: E402
from sixteen.render.sources import AtlasSource  # noqa: E402
from sixteen.render.textures import TextureAtlas  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-one", default="out/bistro-v2")
    ap.add_argument("--fit-two", default="out/bistro-v2x2")
    ap.add_argument("--out", default="results/renderer/demo-tradeoff")
    ap.add_argument("--frames", type=int, default=1350)
    ap.add_argument("--ambient", type=float, default=0.3)
    ap.add_argument("--flip-every", type=int, default=5)
    ap.add_argument("--crf", type=int, default=28)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev, W, H = "cuda", 1280, 720
    w1 = json.load(open("results/decoder/w1_modes.json"))["speedup_over_fp16"]
    cost = json.load(open("results/w1-weights/results.json"))["flip_vs_bilinear_pct"]
    sp = dict(one=w1["fp4"], two=w1["fp4x2"])
    cb = dict(one=cost["concat4_w1nvfp4"], two=cost["concat4_w1nvfp4x2"])

    scene = load_scene("BistroInterior")
    r = Renderer(scene)
    fits1, fits2 = load_all_fits(scene, a.fit_one, dev), load_all_fits(scene, a.fit_two, dev)
    stems = sorted({f["stem"] for f in fits1} & {f["stem"] for f in fits2})
    fits1 = [f for f in fits1 if f["stem"] in stems]
    fits2 = [f for f in fits2 if f["stem"] in stems]
    print(f"{len(stems)} materials fitted in both variants", flush=True)
    atlas = TextureAtlas(OverrideStore(r.store, {f["index"]: TensorSet(f["target"], dev) for f in fits1}))
    mk = lambda fits: MixedSourceV2(AtlasSource(atlas, "stf"), {f["index"]: CudaDecoderV2.from_checkpoint(f["ckpt"]) for f in fits})
    src = dict(ref=AtlasSource(atlas, "stf"), one=mk(fits1), two=mk(fits2))
    fitted = torch.zeros(len(scene.materials), dtype=torch.bool, device=dev)
    fitted[[f["index"] for f in fits1]] = True
    cmp_ = json.load(open("results/renderer/neural-compare-v2/neural_compare.json"))
    bc_mib, our_mib = cmp_["bc_bytes_top_level_total"] / 2**20, cmp_["neural_bytes_total"] / 2**20

    cm = matplotlib.colormaps["magma"]
    taa = {k: TAA(alpha=0.1) for k in src}
    to8 = lambda t: (t.flip(0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    wr = imageio.get_writer(out / "demo_tradeoff.mp4", fps=30, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                            ffmpeg_params=["-crf", str(a.crf), "-preset", "medium"])
    now = dict(one=float("nan"), two=float("nan"))
    ssum, sn, emap, log = dict(one=0.0, two=0.0), 0, dict(one=None, two=None), []
    cw, ch = 640, 360
    for f in range(a.frames):
        cam = path_cam(f / (a.frames - 1))
        d, gbr = {}, None
        for k, s in src.items():
            o = r.render(s, cam, W, H, frame=f, jitter=jitter_at(f), ambient=a.ambient)
            d[k] = to_display(taa[k].resolve(o["rgb"], o["gb"], cam, W, H)).clamp(0, 1)
            if k == "ref":
                gbr = o["gb"]
        mask = gbr["valid"] & fitted[gbr["mat"].clamp(min=0)]
        share = float(mask.sum()) / max(float(gbr["valid"].sum()), 1.0)
        if f % a.flip_every == 0 and f >= 8 and mask.any():
            sn += 1
            for k in ("one", "two"):
                e, _, _ = flip.evaluate(d["ref"].cpu().numpy(), d[k].cpu().numpy(), "LDR", applyMagma=False)
                e = torch.from_numpy(np.asarray(e)[..., 0]).to(dev)
                now[k] = float(e[mask].mean())
                ssum[k] += now[k]
                rgb = torch.from_numpy(cm(e.clamp(0, 0.2).cpu().numpy() / 0.2)[..., :3]).float().to(dev)
                rgb[~mask] = 0.5
                emap[k] = rgb
        run = {k: (ssum[k] / sn if sn else float("nan")) for k in ("one", "two")}
        log.append(dict(frame=f, flip_one=now["one"], flip_two=now["two"], run_one=run["one"], run_two=run["two"], neural_share=share))

        half = lambda t: Image.fromarray(to8(t)).resize((cw, ch), Image.LANCZOS)
        canvas = Image.new("RGB", (3 * cw, 3 * ch), (255, 255, 255))
        canvas.paste(half(d["ref"]), (0, 0))
        canvas.paste(half(d["one"]), (cw, 0))
        canvas.paste(half(d["two"]), (2 * cw, 0))
        blank = torch.full((H, W, 3), 0.5, device=dev)
        canvas.paste(half(emap["one"] if emap["one"] is not None else blank), (cw, ch))
        canvas.paste(half(emap["two"] if emap["two"] is not None else blank), (2 * cw, ch))
        tag(canvas, "Original (BC textures)", (10, 8), 22)
        tag(canvas, "One-term FP4 weights: fastest", (cw + 10, 8), 22)
        tag(canvas, "Two-term FP4 weights: accurate", (2 * cw + 10, 8), 22)
        tag(canvas, "FLIP error, one-term", (cw + 10, ch + 8), 22)
        tag(canvas, "FLIP error, two-term", (2 * cw + 10, ch + 8), 22)
        dr = ImageDraw.Draw(canvas)
        f20, f26, f34 = font(20), font(26), font(36)
        grad = (cm(np.linspace(0, 1, cw - 80))[:, :3] * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(np.repeat(grad[None], 24, 0)), (40, ch + 130))
        dr.text((40, ch + 160), "FLIP error: 0 (identical)", fill=(0, 0, 0), font=f20)
        dr.text((cw - 120, ch + 160), "0.2", fill=(0, 0, 0), font=f20)
        dr.rectangle((40, ch + 206, 64, ch + 230), fill=(128, 128, 128))
        dr.text((74, ch + 204), "pixel not neural (not measured)", fill=(0, 0, 0), font=f20)
        dr.text((40, ch + 20), "Same camera, jitter, tap seeds", fill=(60, 60, 60), font=f20)
        dr.text((40, ch + 50), f"10 neural materials: {our_mib:.1f} MiB vs {bc_mib:.1f} MiB BC", fill=(60, 60, 60), font=f20)
        dr.text((40, ch + 80), f"Neural pixels in view: {share * 100:.0f} %", fill=(60, 60, 60), font=f20)
        y0 = 2 * ch + 20
        for i, (k, title) in enumerate((("one", "One-term FP4 weights"), ("two", "Two-term FP4 weights"))):
            x0 = cw + 20 + i * cw
            dr.text((x0, y0), title, fill=(0, 0, 0), font=f26)
            dr.text((x0, y0 + 44), f"Decode {sp[k]:.2f}x faster than FP16-unpack", fill=(0, 0, 0), font=f20)
            dr.text((x0, y0 + 74), "(1080p, idle GPU, castle_brick_07 model)", fill=(110, 110, 110), font=font(16))
            dr.text((x0, y0 + 104), f"FLIP vs bilinear model: +{cb[k]:.1f} %", fill=(0, 0, 0), font=f20)
            dr.text((x0, y0 + 134), "(castle_brick_07, 2 seeds)", fill=(110, 110, 110), font=font(16))
            dr.text((x0, y0 + 168), "FLIP on this video:", fill=(0, 0, 0), font=f20)
            dr.text((x0, y0 + 196), "..." if sn == 0 else f"{now[k]:.4f}   mean {run[k]:.4f}", fill=(0, 0, 0), font=f34)
        dr.text((20, y0), "Speed vs quality", fill=(0, 0, 0), font=f34)
        dr.text((20, y0 + 56), "The weights of layer 1 are stored", fill=(60, 60, 60), font=f20)
        dr.text((20, y0 + 84), "as one or two NVFP4 terms.", fill=(60, 60, 60), font=f20)
        dr.text((20, y0 + 124), "Two terms: more accurate, less speedup.", fill=(60, 60, 60), font=f20)
        wr.append_data(np.asarray(canvas))
        if f % 100 == 0:
            print(f"frame {f}/{a.frames}  one {now['one']:.4f} two {now['two']:.4f}  share {share:.2f}", flush=True)
    wr.close()
    v = lambda k: [l[f"flip_{k}"] for l in log if l[f"flip_{k}"] == l[f"flip_{k}"]]
    summ = dict(frames=a.frames, materials=stems, speedup=sp, castle_brick_flip_cost_pct=cb, flip_every=a.flip_every,
                flip_mean_one=float(np.mean(v("one"))), flip_mean_two=float(np.mean(v("two"))),
                flip_range_one=[float(np.min(v("one"))), float(np.max(v("one")))], flip_range_two=[float(np.min(v("two"))), float(np.max(v("two")))],
                neural_share_mean=float(np.mean([l["neural_share"] for l in log])), per_frame=log)
    (out / "demo_tradeoff_metrics.json").write_text(json.dumps(summ))
    print(round((out / "demo_tradeoff.mp4").stat().st_size / 1e6, 1), "MB", {k: v for k, v in summ.items() if k not in ("per_frame", "materials")})


if __name__ == "__main__":
    main()
