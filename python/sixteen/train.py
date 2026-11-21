"""Train one neural texture with QAT and evaluate it.

A run = (material, latent format, grid configuration, loss). Training runs in
full precision for the first `qat_start` fraction of steps, then with fake
quantization of the latents (straight-through estimator). Evaluation decodes
the full texture with hard quantization and reports texel PSNR plus PSNR and
FLIP of shaded images (sixteen.shading.EVAL_SETUPS) against the same shading of
the reference textures.
"""

import json
import math
import pathlib
import time

import imageio.v3 as iio
import numpy as np
import torch

from .data import GROUPS, load_material
from .model import NeuralTexture
from .shading import EVAL_SETUPS, ggx_radiance, random_hemisphere, shade_image, tonemap

DEFAULTS = dict(
    steps=20000,
    qat_start=0.7,
    batch=1 << 18,
    lr_latent=1e-2,
    lr_mlp=2e-3,
    hidden=64,
    layers=2,
    w_texel=1.0,
    w_shading=0.0,
    seed=0,
    res="2k",
    crop=(768, 768, 256),  # y, x, size of the saved image crops
)


def psnr(a, b):
    mse = (a - b).pow(2).mean().item()
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def train(material, fmt, grids, out_dir=None, device="cuda", **kw):
    cfg = {**DEFAULTS, **kw}
    torch.manual_seed(cfg["seed"])
    ref = load_material(material, cfg["res"]).permute(1, 2, 0).contiguous().to(device)  # [H, W, 9]
    H, W, C = ref.shape
    flat = ref.view(-1, C)

    model = NeuralTexture((H, W), C, grids=grids, hidden=cfg["hidden"], layers=cfg["layers"]).to(device)
    opt = torch.optim.Adam(
        [
            {"params": model.latents.parameters(), "lr": cfg["lr_latent"]},
            {"params": model.mlp.parameters(), "lr": cfg["lr_mlp"]},
        ]
    )
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps, qat_step = cfg["steps"], int(cfg["qat_start"] * cfg["steps"])

    t0 = time.time()
    for step in range(steps):
        if step == qat_step:
            model.fmt = fmt
        # cosine decay within each phase, so QAT starts from a fresh learning rate
        s0, s1 = (0, qat_step) if step < qat_step else (qat_step, steps)
        f = 0.5 * (1 + math.cos(math.pi * (step - s0) / max(1, s1 - s0)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * (0.02 + 0.98 * f) * (0.3 if step >= qat_step else 1.0)

        idx = torch.randint(0, H * W, (cfg["batch"],), device=device)
        uv = torch.stack([(idx % W + 0.5) / W, (idx // W + 0.5) / H], dim=-1)
        target = flat[idx]
        pred = model(uv)
        loss = cfg["w_texel"] * (pred - target).pow(2).mean()
        if cfg["w_shading"] > 0:
            l = random_hemisphere(idx.shape[0], device)
            v = random_hemisphere(idx.shape[0], device)
            loss = loss + cfg["w_shading"] * (
                tonemap(ggx_radiance(pred, l, v)) - tonemap(ggx_radiance(target, l, v))
            ).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    train_s = time.time() - t0

    metrics = evaluate(model, ref, fmt, cfg, out_dir)
    bits = model.storage_bits(fmt)
    metrics.update(
        material=material,
        fmt=fmt,
        grids=[list(g) for g in grids],
        loss="texel" if cfg["w_shading"] == 0 else f"texel+{cfg['w_shading']:g}*shading",
        bpp_latent=bits["latent_bits"] / (H * W),
        bpp_total=(bits["latent_bits"] + bits["mlp_bits"]) / (H * W),
        train_s=train_s,
        final_loss=loss.item(),
        cfg={k: v for k, v in cfg.items() if k != "crop"},
    )
    if out_dir is not None:
        (pathlib.Path(out_dir) / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


@torch.no_grad()
def evaluate(model, ref, fmt, cfg, out_dir=None):
    import flip_evaluator as flip

    H, W, _ = ref.shape
    dec = model.decode_image(H, W, fmt).clamp(0, 1)
    m = {"psnr_texel": psnr(dec, ref)}
    for g, sl in GROUPS.items():
        m[f"psnr_{g}"] = psnr(dec[..., sl], ref[..., sl])

    shaded = {}
    for s in EVAL_SETUPS:
        a, b = shade_image(ref, s), shade_image(dec, s)
        shaded[s] = (a, b)
        m[f"psnr_shaded_{s}"] = psnr(a, b)
        _, mean_flip, _ = flip.evaluate(a.cpu().numpy(), b.cpu().numpy(), "LDR")
        m[f"flip_{s}"] = float(mean_flip)
    m["psnr_shaded"] = float(np.mean([m[f"psnr_shaded_{s}"] for s in EVAL_SETUPS]))
    m["flip"] = float(np.mean([m[f"flip_{s}"] for s in EVAL_SETUPS]))

    if out_dir is not None:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        y, x, n = cfg["crop"]
        to8 = lambda t: (t[y : y + n, x : x + n].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        for g, sl in GROUPS.items():
            iio.imwrite(out / f"{g}.png", to8(dec[..., sl]))
        for s, (_, b) in shaded.items():
            iio.imwrite(out / f"shaded_{s}.png", to8(b))
    return m


def save_reference_crops(material, out_dir, res="2k", crop=DEFAULTS["crop"]):
    ref = load_material(material, res).permute(1, 2, 0).contiguous()
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y, x, n = crop
    to8 = lambda t: (t[y : y + n, x : x + n].clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)
    for g, sl in GROUPS.items():
        iio.imwrite(out / f"{g}.png", to8(ref[..., sl]))
    for s in EVAL_SETUPS:
        iio.imwrite(out / f"shaded_{s}.png", to8(shade_image(ref, s)))
