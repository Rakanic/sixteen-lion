"""Choose the rounding direction of near-boundary latents by loss (AdaRound-style).

Starts from a trained MXFP4 or NVFP4 checkpoint (scale_mode=max). Block scales stay as
the spec rule sets them and the MLP is frozen. Latents within `band` of a
rounding boundary get a learned up/down choice h in [0, 1] (rectified
sigmoid, annealed regularizer pushes it to 0 or 1); all others round to
nearest. Arms:

    nearest   plain round-to-nearest (what QAT ends with)
    more-qat  control: the same number of extra STE-QAT steps on the latents
    texel     rounding choices optimized for texel error
    shading   rounding choices optimized for texel + shading error

    pixi run python tools/shading_rounding.py out/profile/castle_brick_07_mxfp4_shade4
"""

import argparse
import json
import pathlib

import torch

from sixteen.data import load_material
from sixteen.model import NeuralTexture
from sixteen.quant import E2M1, E4M3, round_minifloat
from sixteen.shading import ggx_radiance, random_hemisphere, tonemap
from sixteen.train import evaluate

GRID = torch.tensor([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6.0])


def block_scale(x, fmt):
    """Per-value block scale exactly as the format's quantizer sets it."""
    if fmt == "mxfp4":
        b = x.reshape(-1, 32)
        amax = b.abs().amax(1, keepdim=True).clamp(min=2.0**-127)
        return torch.exp2(torch.floor(torch.log2(amax)) - E2M1.emax).expand_as(b).reshape(x.shape)
    b = x.reshape(-1, 16)  # nvfp4: E4M3 block scale times an fp32 tensor scale
    gscale = b.abs().amax().clamp(min=1e-12) / (E2M1.max * E4M3.max)
    s = round_minifloat(b.abs().amax(1, keepdim=True) / E2M1.max / gscale, E4M3) * gscale
    s = torch.where(s > 0, s, torch.ones_like(s))
    return s.expand_as(b).reshape(x.shape)


def neighbours(v):
    """Grid points just below and above v (v already clipped to [-6, 6])."""
    g = GRID.to(v.device)
    i = torch.bucketize(v, g).clamp(1, len(g) - 1)
    lo, hi = g[i - 1], g[i]
    on = (v == hi)
    lo = torch.where(on, hi, lo)
    return lo, hi


class Rounder(torch.nn.Module):
    def __init__(self, x, band, fmt):
        super().__init__()
        self.s = block_scale(x, fmt)
        v = (x / self.s).clamp(-6, 6)
        self.lo, self.hi = neighbours(v)
        gap = (self.hi - self.lo).clamp(min=1e-12)
        t = ((v - self.lo) / gap).clamp(0, 1)
        self.free = (self.hi > self.lo) & ((t - 0.5).abs() < band)
        self.nearest_h = (round_minifloat(v, E2M1) == self.hi).float()
        # init: h = 0.7 / 0.3 toward the nearest choice. Not 0/1: the rectified
        # sigmoid is flat there, so the choice could never move.
        h0 = 0.3 + 0.4 * self.nearest_h
        self.a = torch.nn.Parameter(torch.log((h0 + 0.1) / (1.1 - h0)))

    def h(self):
        soft = (torch.sigmoid(self.a) * 1.2 - 0.1).clamp(0, 1)
        return torch.where(self.free, soft, self.nearest_h)

    def forward(self, hard=False):
        h = self.h()
        if hard:
            h = torch.where(self.free, (h > 0.5).float(), self.nearest_h)
        return (self.lo + h * (self.hi - self.lo)) * self.s

    def reg(self, beta):
        h = self.h()[self.free]
        return (1 - (2 * h - 1).abs().pow(beta)).mean()


def batch_loss(model, latents, flat, H, W, batch, w_shading, device):
    idx = torch.randint(0, H * W, (batch,), device=device)
    uv = torch.stack([(idx % W + 0.5) / W, (idx // W + 0.5) / H], dim=-1)
    pred, target = model(uv, latents), flat[idx]
    loss = (pred - target).pow(2).mean()
    if w_shading > 0:
        l, v = random_hemisphere(batch, device), random_hemisphere(batch, device)
        loss = loss + w_shading * (tonemap(ggx_radiance(pred, l, v)) - tonemap(ggx_radiance(target, l, v))).pow(2).mean()
    return loss


def load(run, device):
    ck = torch.load(pathlib.Path(run) / "model.pt", weights_only=False, map_location=device)
    cfg = ck["cfg"]
    mat = json.loads((pathlib.Path(run) / "metrics.json").read_text())["material"]
    ref = load_material(mat, cfg["res"]).permute(1, 2, 0).contiguous().to(device)
    H, W, C = ref.shape
    model = NeuralTexture((H, W), C, grids=[tuple(g) for g in ck["grids"]], hidden=cfg["hidden"], layers=cfg["layers"], filter=cfg.get("filter", "bilinear")).to(device)
    model.load_state_dict(ck["state"])
    model.fmt = ck["fmt"]
    assert ck["fmt"] in ("mxfp4", "nvfp4") and all(c is None for c in ck["scale_choice"]), "needs a 4-bit scale_mode=max run"
    for p in model.mlp.parameters():
        p.requires_grad_(False)
    return model, ref, cfg, mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--w-shading", type=float, default=4.0)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=1 << 20)
    ap.add_argument("--arms", default="nearest,texel,shading")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"
    out = pathlib.Path(args.run) / "rounding"
    out.mkdir(exist_ok=True)
    results = {}
    for arm in args.arms.split(","):
        torch.manual_seed(args.seed)
        model, ref, cfg, mat = load(args.run, dev)
        fmt = model.fmt
        H, W, C = ref.shape
        flat = ref.view(-1, C)
        ecfg = {**cfg, "crop": (768, 768, 256)}
        if arm == "nearest":
            pass
        elif arm == "more-qat":
            opt = torch.optim.Adam(model.latents.parameters(), lr=1e-2 * 0.3 * 0.02)  # QAT's final lr
            for _ in range(args.steps):
                loss = batch_loss(model, model.quantized_latents(), flat, H, W, args.batch, args.w_shading, dev)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        else:
            ws = args.w_shading if arm == "shading" else 0.0
            rounders = torch.nn.ModuleList([Rounder(g.detach(), args.band, fmt) for g in model.latents]).to(dev)
            opt = torch.optim.Adam(rounders.parameters(), lr=1e-2)
            for step in range(args.steps):
                beta = 20 - 18 * step / args.steps  # anneal: allow movement first, then force 0/1
                lat = [r() for r in rounders]
                loss = batch_loss(model, lat, flat, H, W, args.batch, ws, dev)
                reg = sum(r.reg(beta) for r in rounders) / len(rounders)
                warm = args.steps // 10  # no regularizer first, then ramp it up over half the run
                lam = args.lam * min(1.0, max(0.0, (step - warm) / (args.steps // 2)))
                opt.zero_grad(set_to_none=True)
                (loss + lam * reg).backward()
                opt.step()
            with torch.no_grad():
                free = sum(r.free.sum().item() for r in rounders)
                total = sum(r.free.numel() for r in rounders)
                flipped = sum(((r.h() > 0.5).float() != r.nearest_h)[r.free].sum().item() for r in rounders)
                for g, r in zip(model.latents, rounders):
                    g.copy_(r(hard=True))
            # The hardened values are exact values of the format under the
            # stored block scales; decode them as they are (no re-quantization,
            # which would recompute NVFP4's tensor and block scales).
            model.fmt = None
            results.setdefault("info", {})[arm] = {"free_frac": free / total, "flipped_frac_of_free": flipped / max(free, 1)}
        m = evaluate(model, ref, "fp32" if model.fmt is None else fmt, ecfg, out / arm)
        results[arm] = {k: v for k, v in m.items() if k.startswith(("psnr", "flip"))}
        print(arm, {k: round(results[arm][k], 4) for k in ("psnr_shaded", "flip", "psnr_unseen", "flip_unseen")}, flush=True)
    meta = json.loads((pathlib.Path(args.run) / "metrics.json").read_text())
    results.update(material=mat, fmt=meta["fmt"], loss=meta["loss"], train_seed=meta["cfg"]["seed"], round_seed=args.seed)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results.get("info", {}), indent=1))


if __name__ == "__main__":
    main()
