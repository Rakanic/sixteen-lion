"""Profile the latent values of one trained neural texture.

Trains one run (or reuses its checkpoint) and plots where the latent values
fall: raw histogram before and after QAT, per-channel ranges, where values
land on the 4-bit grid after MX / NVFP4 block scaling (how many round to 0,
how many blocks clip), in-block dynamic range, and per-value loss sensitivity.

    pixi run python tools/profile_latents.py --out <plots dir>
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sixteen.figures import INK, MUTED, style_axes
from sixteen.quant import E2M1, E4M3, round_minifloat
from sixteen.train import train

ROOT = pathlib.Path(__file__).resolve().parent.parent
BLUE, ORANGE = "#2a78d6", "#eb6834"
E2M1_GRID = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def mx_ratios(x, block=32):
    """|x| / block scale under the MX spec rule, and the block max ratio."""
    b = x.reshape(-1, block).abs()
    amax = b.amax(1, keepdim=True).clamp(min=2.0**-127)
    scale = torch.exp2(torch.floor(torch.log2(amax)) - E2M1.emax)
    return (b / scale).flatten(), (amax / scale).flatten()


def nv_ratios(x, block=16):
    b = x.reshape(-1, block).abs()
    gscale = b.amax().clamp(min=1e-12) / (E2M1.max * E4M3.max)
    bscale = round_minifloat(b.amax(1, keepdim=True) / E2M1.max / gscale, E4M3) * gscale
    bscale = torch.where(bscale > 0, bscale, torch.ones_like(bscale))
    return (b / bscale).flatten(), (b.amax(1, keepdim=True) / bscale).flatten()


def level_fractions(r):
    """Fraction of values that round to each E2M1 magnitude (saturating at 6)."""
    q = round_minifloat(r, E2M1)
    return {f"{g:g}": (q == g).float().mean().item() for g in E2M1_GRID}


def fig_save(fig, outs, name):
    for o in outs:
        fig.savefig(o / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="castle_brick_07")
    ap.add_argument("--fmt", default="mxfp4")
    ap.add_argument("--w-shading", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", help="extra output directory for the figures")
    args = ap.parse_args()

    run = ROOT / "out" / "profile" / f"{args.material}_{args.fmt}_shade{args.w_shading:g}"
    if not (run / "model.pt").exists():
        train(args.material, args.fmt, [(4, 16), (8, 16)], out_dir=run, steps=args.steps, w_shading=args.w_shading)
    ck = torch.load(run / "model.pt", weights_only=False, map_location="cpu")
    fl = torch.load(run / "model_float.pt", weights_only=False, map_location="cpu")
    names = [k for k in ck["state"] if k.startswith("latents.")]
    outs = [run]
    if args.out:
        outs.append(pathlib.Path(args.out))
        outs[-1].mkdir(parents=True, exist_ok=True)

    summary = {}
    # 1. raw value histogram, before vs after QAT, per grid
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 2.8), squeeze=False, facecolor="#fcfcfb")
    for ax, n in zip(axes[0], names):
        style_axes(ax)
        a, b = fl[n].flatten().cpu().numpy(), ck["state"][n].flatten().cpu().numpy()
        lim = np.percentile(np.abs(np.concatenate([a, b])), 99.9)
        bins = np.linspace(-lim, lim, 121)
        ax.hist(a, bins, color=BLUE, alpha=0.55, label="end of fp32 phase")
        ax.hist(b, bins, histtype="step", color=ORANGE, lw=1.6, label="after QAT")
        ax.set_yscale("log")
        h, w, c = ck["state"][n].shape
        ax.set_title(f"grid {n.split('.')[-1]}: {h}x{w}x{c}", fontsize=9, color=INK)
        ax.set_xlabel("latent value", fontsize=8, color=MUTED)
        for tag, t in (("float", fl[n]), ("qat", ck["state"][n])):
            summary[f"{n}_{tag}"] = {
                "mean": t.mean().item(), "std": t.std().item(),
                "p0.1": t.quantile(0.001).item() if t.numel() < 2**24 else None,
                "min": t.min().item(), "max": t.max().item(),
                "abs_p50": t.abs().median().item(),
            }
    axes[0][0].set_ylabel("count (log)", fontsize=8, color=MUTED)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig_save(fig, outs, "latent_hist.png")

    # 2. per-channel spread (after QAT): 1st-99th percentile bars + median
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 2.6), squeeze=False, facecolor="#fcfcfb")
    for ax, n in zip(axes[0], names):
        style_axes(ax)
        t = ck["state"][n].reshape(-1, ck["state"][n].shape[-1]).float()
        idx = torch.randperm(t.shape[0])[:200000]
        q = torch.quantile(t[idx], torch.tensor([0.01, 0.5, 0.99]), dim=0).numpy()
        xs = np.arange(t.shape[1])
        ax.vlines(xs, q[0], q[2], color=BLUE, lw=3)
        ax.plot(xs, q[1], "o", color=INK, ms=3)
        ax.set_title(f"grid {n.split('.')[-1]}: per-channel 1-99% range", fontsize=9, color=INK)
        ax.set_xlabel("channel", fontsize=8, color=MUTED)
        ax.set_xticks(xs)
        ax.tick_params(axis="x", labelsize=6)
        summary[f"{n}_channel_absmax_ratio"] = (t.abs().amax(0).max() / t.abs().amax(0).min()).item()
    fig.tight_layout()
    fig_save(fig, outs, "latent_channels.png")

    # 3. where values land on the E2M1 grid after block scaling (fine grid, after QAT)
    x = ck["state"][names[0]].float()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8), facecolor="#fcfcfb")
    for ax, (fmt, fn) in zip(axes, (("MXFP4 (32/block, pow2 scale)", mx_ratios), ("NVFP4 (16/block, E4M3 scale)", nv_ratios))):
        style_axes(ax)
        r, rmax = fn(x)
        ax.hist(r.numpy(), np.linspace(0, 8, 161), color=BLUE)
        for g in E2M1_GRID:
            ax.axvline(g, color=MUTED, lw=0.6, ls=":")
        ax.axvspan(6.25, 8, color=ORANGE, alpha=0.12)
        ax.text(7.1, ax.get_ylim()[1] * 0.85, "clipped", fontsize=7, color=MUTED, ha="center")
        ax.set_title(fmt, fontsize=9, color=INK)
        ax.set_xlabel("|value| / block scale  (dotted = E2M1 levels)", fontsize=8, color=MUTED)
        key = fmt.split()[0].lower()
        summary[key] = {
            "frac_values_round_to_0": level_fractions(r)["0"],
            "level_fractions": level_fractions(r),
            "frac_values_clipped": (r > 6.0).float().mean().item(),
            "frac_blocks_clipped": (rmax > 6.0).float().mean().item(),
            "mean_block_max_ratio": rmax.mean().item(),
        }
    axes[0].set_ylabel("count", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig_save(fig, outs, "latent_on_e2m1_grid.png")

    # 4. in-block dynamic range: max |x| / median |x| per 32-block
    b = x.reshape(-1, 32).abs()
    dr = (b.amax(1) / b.median(1).values.clamp(min=1e-8)).numpy()
    # 5. sensitivity (accumulated grad^2 of the training loss over the last window)
    s = ck["sensitivity"][0].float().flatten()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8), facecolor="#fcfcfb")
    style_axes(axes[0])
    axes[0].hist(np.log2(np.clip(dr, 1, None)), 80, color=BLUE)
    axes[0].set_xlabel("log2(max |x| / median |x|) per 32-value block", fontsize=8, color=MUTED)
    axes[0].set_ylabel("blocks", fontsize=8, color=MUTED)
    axes[0].set_title("in-block dynamic range", fontsize=9, color=INK)
    style_axes(axes[1])
    if s.sum() > 0:
        sn = (s / s.mean()).numpy()
        axes[1].hist(np.log10(sn[sn > 0]), 100, color=BLUE)
        axes[1].set_xlabel("log10(sensitivity / mean)  (grad^2 of training loss)", fontsize=8, color=MUTED)
        axes[1].set_title("per-value loss sensitivity", fontsize=9, color=INK)
        top = np.sort(sn)[::-1]
        summary["sensitivity_share_top10pct"] = float(top[: len(top) // 10].sum() / top.sum())
        summary["sensitivity_vs_abs_corr"] = float(np.corrcoef(sn, x.abs().flatten().numpy())[0, 1])
        ch = (s.view(-1, x.shape[-1]).mean(0) / s.mean()).numpy()
        summary["sensitivity_per_channel"] = ch.round(3).tolist()
    else:
        axes[1].text(0.5, 0.5, "no sensitivity recorded\n(scale_mode=max run)", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=8, color=MUTED)
    summary["block_dynamic_range_median"] = float(np.median(dr))
    fig.tight_layout()
    fig_save(fig, outs, "latent_blocks_sensitivity.png")

    for o in outs:
        (o / "profile_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
