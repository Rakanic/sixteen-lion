"""Figures and tables from a sweep's out/<name>/ directory.

    python -m sixteen.figures out/milestone --out <plots dir>

Writes results.csv, rate-distortion plots (quality vs bits per texel, one
panel per material, one line per latent format across grid sizes), a
side-by-side crop grid per material, and for sweeps with several losses a
grouped bar chart comparing them.
"""

import argparse
import json
import pathlib

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Quantized formats take categorical slots 1-7 in fixed order (the dataviz
# reference palette); the unquantized references are neutral gray.
FORMAT_ORDER = ["fp32", "fp16", "int8", "int4", "mxint8", "mxfp8", "mxfp6", "mxfp4", "nvfp4"]
FORMAT_STYLE = {
    "fp32": dict(color="#8a8984", ls="--"),
    "fp16": dict(color="#8a8984", ls=":"),
    "int8": dict(color="#2a78d6"),
    "int4": dict(color="#eb6834"),
    "mxint8": dict(color="#1baf7a"),
    "mxfp8": dict(color="#eda100"),
    "mxfp6": dict(color="#e87ba4"),
    "mxfp4": dict(color="#008300"),
    "nvfp4": dict(color="#6250d6"),
}
LOSS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
METRICS = {
    "psnr_shaded": ("PSNR after shading (dB)", True),
    "flip": ("FLIP after shading (lower is better)", False),
    "psnr_texel": ("texel PSNR (dB)", True),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style_axes(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def load(sweep_dir):
    rows = []
    for p in sorted(pathlib.Path(sweep_dir).rglob("metrics.json")):
        m = json.loads(p.read_text())
        m.pop("cfg", None)
        m["grid"] = p.parts[-4]
        m["loss_name"] = p.parts[-3]
        m["grids"] = str(m["grids"])
        rows.append(m)
    return pd.DataFrame(rows)


def rd_plot(df, metric, out):
    label, _ = METRICS[metric]
    mats = sorted(df.material.unique())
    fig, axes = plt.subplots(1, len(mats), figsize=(3.4 * len(mats), 3.0), squeeze=False, facecolor="#fcfcfb")
    for ax, mat in zip(axes[0], mats):
        style_axes(ax)
        d = df[(df.material == mat) & (df.loss_name == "texel")]
        for fmt in [f for f in FORMAT_ORDER if f in set(d.fmt)]:
            s = d[d.fmt == fmt].sort_values("bpp_total")
            ax.plot(s.bpp_total, s[metric], marker="o", ms=4, lw=1.6, label=fmt,
                    markeredgecolor="#fcfcfb", markeredgewidth=0.8, **FORMAT_STYLE[fmt])
        ax.set_xscale("log", base=2)
        ax.set_title(mat.replace("_", " "), fontsize=9, color=INK)
        ax.set_xlabel("bits per texel (9 channels, incl. MLP)", fontsize=8, color=MUTED)
    axes[0][0].set_ylabel(label, fontsize=8, color=MUTED)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def crop_grid(sweep_dir, df, material, grid, out, fmts=("fp32", "mxfp8", "mxfp6", "mxfp4", "nvfp4", "int4")):
    root = pathlib.Path(sweep_dir) / material
    cols = ["albedo", "normal", "arm", "shaded_grazing", "shaded_specular"]
    rows = [("reference", root / "reference")]
    for f in fmts:
        d = root / grid / "texel" / f
        if (d / "metrics.json").exists():
            rows.append((f, d))
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.6 * len(cols), 1.6 * len(rows)), squeeze=False)
    for r, (name, d) in enumerate(rows):
        for c, col in enumerate(cols):
            ax = axes[r][c]
            ax.imshow(iio.imread(d / f"{col}.png"))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(col.replace("shaded_", "shaded: "), fontsize=7, color=INK)
            if c == 0:
                txt = name
                if name != "reference":
                    m = df[(df.material == material) & (df.grid == grid) & (df.fmt == name) & (df.loss_name == "texel")].iloc[0]
                    txt = f"{name}\n{m.bpp_total:.1f} bpt\nFLIP {m.flip:.3f}"
                ax.set_ylabel(txt, fontsize=7, color=INK, rotation=0, ha="right", va="center")
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def loss_bars(df, metric, out):
    label, _ = METRICS[metric]
    losses = list(dict.fromkeys(df.loss_name))
    mats = sorted(df.material.unique())
    fmts = [f for f in FORMAT_ORDER if f in set(df.fmt)]
    fig, axes = plt.subplots(1, len(mats), figsize=(3.4 * len(mats), 2.8), squeeze=False, facecolor="#fcfcfb")
    w = 0.8 / len(losses)
    for ax, mat in zip(axes[0], mats):
        style_axes(ax)
        ax.grid(axis="x", visible=False)
        for i, loss in enumerate(losses):
            d = df[(df.material == mat) & (df.loss_name == loss)].set_index("fmt").reindex(fmts)
            ax.bar(np.arange(len(fmts)) + (i - (len(losses) - 1) / 2) * w, d[metric], w * 0.92,
                   color=LOSS_COLORS[i], label=loss)
        ax.set_xticks(np.arange(len(fmts)), fmts, fontsize=8)
        ax.set_title(mat.replace("_", " "), fontsize=9, color=INK)
        lo = df[metric].min()
        if metric.startswith("psnr"):
            ax.set_ylim(np.floor(lo - 1), None)
    axes[0][0].set_ylabel(label, fontsize=8, color=MUTED)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--out", help="output directory (default: <sweep_dir>/figures)")
    args = ap.parse_args()
    out = pathlib.Path(args.out or pathlib.Path(args.sweep_dir) / "figures")
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.sweep_dir)
    if df.empty:
        print("no finished runs")
        return
    cols = ["material", "grid", "loss_name", "fmt", "bpp_total", "bpp_latent", "psnr_texel", "psnr_albedo",
            "psnr_normal", "psnr_arm", "psnr_shaded", "flip", "train_s"]
    df.sort_values(["material", "grid", "loss_name", "fmt"])[cols + [c for c in df.columns if c not in cols]].to_csv(
        out / "results.csv", index=False)
    multi_loss = df.loss_name.nunique() > 1
    for metric in METRICS:
        if df.grid.nunique() > 1:
            rd_plot(df, metric, out / f"rd_{metric}.png")
        if multi_loss:
            loss_bars(df, metric, out / f"loss_{metric}.png")
    for mat in df.material.unique():
        for grid in sorted(df[df.material == mat].grid.unique()):
            crop_grid(args.sweep_dir, df, mat, grid, out / f"crops_{mat}_{grid}.png")
    print(f"{len(df)} runs -> {out}")


if __name__ == "__main__":
    main()
