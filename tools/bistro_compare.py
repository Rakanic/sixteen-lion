"""Compare our neural textures against the original Bistro textures on a rendered view.

Fits come from `python -m sixteen.sweep configs/bistro-pilot.yaml` (bilinear model, PyTorch decode only) or
`configs/bistro-v2.yaml` (concat4 + NVFP4 layer-1 weights; also rendered through CUDA decoder v2, FP4-direct, and
checked against the PyTorch decode). Both renders use the same camera, the
same jitter and the same stochastic-tap seeds, and the same atlas path; the reference uses the training
target tensors, the neural render uses the decoded fits, so the difference is compression error only.
Materials without a fit use their original textures in both renders.

    pixi run python tools/bistro_compare.py --fit out/bistro-pilot --frames 32 --out results/renderer/neural-compare
"""

import argparse
import json
import os
import pathlib
import sys

import flip_evaluator as flip
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bistro_inventory import dds_info  # noqa: E402
from render_bistro import CAMERAS  # noqa: E402
from sixteen.bistro_data import set_files  # noqa: E402
from sixteen.decoder import CudaDecoderV2  # noqa: E402
from sixteen.render.neural_ref import MixedSourceV2, OverrideStore, TensorSet, load_all_fits  # noqa: E402
from sixteen.render.renderer import Renderer  # noqa: E402
from sixteen.render.scene import load_scene  # noqa: E402
from sixteen.render.shade import to_display  # noqa: E402
from sixteen.render.sources import AtlasSource  # noqa: E402
from sixteen.render.textures import TextureAtlas  # noqa: E402

BPP = {"DXT1": 4, "DXT5": 8, "ATI2": 8}


def psnr(a, b):
    return float(10 * torch.log10(1.0 / (a - b).pow(2).mean().clamp(min=1e-12)))


def bc_bytes(stem):
    """(top level only, with mips) bytes of the shipped BaseColor + Normal + Specular files."""
    top, full = 0, 0
    for p in set_files(stem):
        i = dds_info(p)
        top += i["w"] * i["h"] * BPP[i["fmt"]] // 8
        full += i["bytes"]
    return top, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="BistroInterior")
    ap.add_argument("--camera", default="entrance")
    ap.add_argument("--fit", default="out/bistro-pilot")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--out", default="results/renderer/neural-compare")
    ap.add_argument("--ambient", type=float, default=0.3, help="constant sky term; same for every render here")
    ap.add_argument("--crop", type=lambda t: tuple(map(int, t.split("x"))), default=(960, 540), help="crop window WxH in pixels")
    ap.add_argument("--crop-min-neural", type=float, default=0.8, help="minimum fraction of neural pixels in the crop window")
    a = ap.parse_args()
    W, H = map(int, a.size.split("x"))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    scene = load_scene(a.scene)
    cam = CAMERAS[a.scene][a.camera]
    r = Renderer(scene)
    base = r.store
    fits = load_all_fits(scene, a.fit, dev)
    assert fits, f"no checkpoints under {a.fit}"
    ref_over, neu_over, per = {}, {}, []
    for f in fits:
        stem, i, model, meta, target, dec = f["stem"], f["index"], f["model"], f["meta"], f["target"], f["decoded"]
        # check: the loaded checkpoint decodes to the same quality train() recorded (catches a missing fmt / scales)
        p = psnr(dec, target)
        assert abs(p - meta["psnr_texel"]) < 0.02, (stem, p, meta["psnr_texel"])
        # check: per-pixel forward at texel centres equals the pre-decoded image
        g = torch.Generator(device=dev).manual_seed(0)
        res = target.shape[0]
        idx = torch.randint(0, res * res, (20000,), device=dev, generator=g)
        uv = torch.stack([(idx % res + 0.5) / res, (idx // res + 0.5) / res], -1)
        with torch.no_grad():
            direct = model(uv, model.quantized_latents(model.fmt, hard=True))
        assert (direct.clamp(0, 1) - dec.reshape(-1, 9)[idx]).abs().max() < 2e-5, stem
        bits = model.storage_bits(model.fmt)
        top, full = bc_bytes(stem)
        ref_over[i], neu_over[i] = TensorSet(target, dev), TensorSet(dec, dev)
        per.append(dict(material=stem, scene_index=i, psnr_texel=p, psnr_texel_metrics_json=meta["psnr_texel"],
                        flip_flat_plane=meta["flip"], psnr_shaded_flat_plane=meta["psnr_shaded"],
                        neural_bytes=(bits["latent_bits"] + bits["mlp_bits"]) / 8, bc_bytes_top_level=top, bc_bytes_with_mips=full,
                        bpp_total=meta["bpp_total"], train_s=meta["train_s"]))
        print(f"{stem:45s} psnr {p:5.2f} dB  flat-plane FLIP {meta['flip']:.4f}  neural {per[-1]['neural_bytes'] / 1e6:.2f} MB vs BC {top / 1e6:.2f} MB ({full / 1e6:.2f} with mips)", flush=True)

    atlas_ref = TextureAtlas(OverrideStore(base, ref_over))
    atlas_neu = TextureAtlas(OverrideStore(base, neu_over))
    s_ref, s_neu = AtlasSource(atlas_ref, "stf"), AtlasSource(atlas_neu, "stf")

    # check: pushing the *target* through the neural path gives an identical image (plumbing adds no error)
    atlas_chk = TextureAtlas(OverrideStore(base, ref_over))
    chk = r.accumulate(AtlasSource(atlas_chk, "stf"), cam, W, H, 2, ambient=a.ambient) - r.accumulate(s_ref, cam, W, H, 2, ambient=a.ambient)
    assert chk.abs().max() == 0, "plumbing check failed"
    del atlas_chk, chk

    rgb_ref = r.accumulate(s_ref, cam, W, H, a.frames, ambient=a.ambient)
    rgb_neu = r.accumulate(s_neu, cam, W, H, a.frames, ambient=a.ambient)
    v2 = all(f["model"].filter == "concat4" and f["model"].w1_fmt == "nvfp4" for f in fits)
    cuda = {}
    if v2:
        # CUDA decoder v2 (FP4-direct layer 1) with seeds equal to the screen pixel index, so taps match s_ref/s_neu
        decs = {f["index"]: CudaDecoderV2.from_checkpoint(f["ckpt"]) for f in fits}
        for name, fp4 in (("fp4", True), ("fp16", False)):
            src = MixedSourceV2(AtlasSource(atlas_ref, "stf"), decs, fp4=fp4)
            r.render(src, cam, W, H, ambient=a.ambient)  # warm-up
            src.ms.clear()
            cuda[name] = (r.accumulate(src, cam, W, H, a.frames, ambient=a.ambient), src)
        rgb_pt = rgb_neu
        rgb_neu = cuda["fp4"][0]  # headline render: the real decoder
    gb = r.render(s_ref, cam, W, H)["gb"]
    fitted = torch.zeros(len(scene.materials), dtype=torch.bool, device=dev)
    fitted[list(ref_over)] = True
    mask = gb["valid"] & fitted[gb["mat"].clamp(min=0)]
    cov = {p["material"]: float((gb["mat"] == p["scene_index"]).sum()) / float(gb["valid"].sum()) for p in per}
    for p in per:
        p["pixel_share"] = cov[p["material"]]

    d_ref, d_neu = to_display(rgb_ref).clamp(0, 1), to_display(rgb_neu).clamp(0, 1)
    extra = {}
    if v2:
        d_pt, d_16 = to_display(rgb_pt).clamp(0, 1), to_display(cuda["fp16"][0]).clamp(0, 1)
        dec_ms = lambda k: float(np.median(cuda[k][1].ms))
        extra = dict(
            decode="CUDA decoder v2 (FP4-direct layer 1, output-space STF); PyTorch decode of the same fits reported alongside",
            flip_whole_frame_pytorch=float(flip.evaluate(d_ref.cpu().numpy(), d_pt.cpu().numpy(), "LDR")[1]),
            psnr_whole_frame_pytorch=psnr(d_pt, d_ref),
            cuda_vs_pytorch=dict(max_abs=float((d_neu - d_pt).abs().max()), mean_abs=float((d_neu - d_pt).abs().mean()),
                                 psnr=psnr(d_neu, d_pt), fp16_path_vs_fp4_path_max_abs=float((d_16 - d_neu).abs().max())),
            decoder_ms_per_frame_median=dict(fp4=dec_ms("fp4"), fp16=dec_ms("fp16"), materials=len(fits),
                                             note="all decoder launches of one 1080p frame, GPU shared and busy; each material decodes a full frame"),
        )
    err, mean_flip, _ = flip.evaluate(d_ref.cpu().numpy(), d_neu.cpu().numpy(), "LDR")
    err = np.asarray(err)
    err_t = torch.from_numpy(err[..., 0] if err.ndim == 3 else err).to(dev)  # FLIP's map is replicated over RGB
    res = dict(
        scene=a.scene, camera=a.camera, size=[W, H], frames=a.frames, ambient=a.ambient, materials_fitted=len(per),
        neural_pixel_share=float(mask.sum()) / float(gb["valid"].sum()),
        flip_whole_frame=float(mean_flip), flip_neural_pixels=float(err_t[mask].mean()),
        psnr_whole_frame=psnr(d_neu, d_ref), psnr_neural_pixels=psnr(d_neu[mask], d_ref[mask]),
        neural_bytes_total=sum(p["neural_bytes"] for p in per), bc_bytes_top_level_total=sum(p["bc_bytes_top_level"] for p in per),
        bc_bytes_with_mips_total=sum(p["bc_bytes_with_mips"] for p in per), per_material=per,
        checks=["checkpoint psnr_texel == metrics.json (<0.02 dB), per material", "per-pixel forward == pre-decoded texels, per material",
                "target pushed through the neural path renders identically (max abs diff 0)"],
        note=("CUDA decoder v2 on concat4 fits with NVFP4 latents and layer-1 weights; reference = decoded BC textures; identical jitter and tap seeds."
              if v2 else "PyTorch decode of NVFP4 latents (bilinear model), not the CUDA kernel. Reference = decoded BC textures. Frames use identical jitter and tap seeds."),
        **extra,
    )
    (out / "neural_compare.json").write_text(json.dumps(res, indent=1))

    # images. Tensors are in GL row order (row 0 = bottom); `view` flips to top-down for files.
    import matplotlib
    from PIL import Image

    cm = matplotlib.colormaps["magma"]
    err_rgb = torch.from_numpy(cm(err_t.clamp(0, 0.2).cpu().numpy() / 0.2)[..., :3]).float().to(dev)  # FLIP 0..0.2
    view = lambda t: (t.flip(0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    for name, t in (("reference", d_ref), ("neural", d_neu), ("flip", err_rgb)):
        Image.fromarray(view(t)).resize((1280, 720), Image.LANCZOS).save(out / f"{name}.jpg", quality=90)
    # crop: the window (default 960x540, a quarter of the frame) that is mostly neural pixels and has the largest
    # mean FLIP over its neural pixels. Figure: original | ours | FLIP map (magma, 0..0.2) with a colorbar; pixels
    # of materials we did not fit are gray in the map (both renders use the same original texture there).
    cw, ch = a.crop
    win = (ch, cw)
    nsum = F.avg_pool2d(mask.float()[None, None], win, stride=30)[0, 0]  # fraction of neural pixels per window
    esum = F.avg_pool2d((err_t * mask)[None, None], win, stride=30)[0, 0]
    score = torch.where(nsum >= a.crop_min_neural, esum / nsum.clamp(min=1e-9), torch.zeros_like(esum))
    yi, xi = divmod(int(score.argmax()), score.shape[1])
    y0, x0 = yi * 30, xi * 30
    sl = lambda t: t[y0 : y0 + ch, x0 : x0 + cw]
    m_c = sl(mask)
    flip_rgb = err_rgb.clone()
    flip_rgb[~mask] = 0.5  # gray: not neural
    panels = [view(sl(d_ref)), view(sl(d_neu)), view(sl(flip_rgb))]
    from PIL import ImageDraw, ImageFont

    # 2x2 grid: original | ours / FLIP map | legend (colorbar, gray = not neural)
    canvas = Image.new("RGB", (2 * cw + 8, 2 * ch + 8), (255, 255, 255))
    for i, pnl in enumerate(panels):
        canvas.paste(Image.fromarray(pnl), ((i % 2) * (cw + 8), (i // 2) * (ch + 8)))
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=34)
    lx, ly, bw = cw + 8 + 60, ch + 8 + ch // 2 - 70, cw - 120
    grad = (cm(np.linspace(0, 1, bw))[:, :3] * 255).astype(np.uint8)
    canvas.paste(Image.fromarray(np.repeat(grad[None], 40, 0)), (lx, ly))
    d.text((lx, ly + 50), "0 (identical)", fill=(0, 0, 0), font=font)
    d.text((lx + bw - 170, ly + 50), "0.2 FLIP", fill=(0, 0, 0), font=font)
    d.rectangle((lx, ly + 120, lx + 40, ly + 160), fill=(128, 128, 128))
    d.text((lx + 56, ly + 122), "pixel not neural (not measured)", fill=(0, 0, 0), font=font)
    canvas.save(out / "crop.jpg", quality=93)
    json.dump({"crop_origin_gl_row_col": [y0, x0], "crop_size_hw": list(win), "neural_fraction_in_crop": float(m_c.float().mean()),
               "mean_flip_neural_pixels_in_crop": float(sl(err_t)[m_c].mean()), "mean_flip_all_pixels_in_crop": float(sl(err_t).mean()),
               "selection": "highest mean FLIP over neural pixels among windows with at least crop_min_neural neural pixels", "crop_min_neural": a.crop_min_neural},
              open(out / "crop.json", "w"))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_material", "checks", "note")}, indent=1))
    if v2:
        res["checks"].append("CUDA v2 render vs PyTorch decode of the same fits: see cuda_vs_pytorch")
        (out / "neural_compare.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
