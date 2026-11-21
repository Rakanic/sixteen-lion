"""Summarize the rendering-aware rounding experiment.

Reads <sweep>/**/rounding/results.json (tools/shading_rounding.py) and writes
a CSV of every run, a markdown summary (win counts, mean FLIP change on the
training-like and unseen lighting, per material), a bar chart, and the gain
expressed as equivalent memory using the milestone quality-vs-bits curves.

    pixi run python tools/rounding_report.py out/rounding-base --out results/rounding
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sixteen.figures import INK, MUTED, style_axes

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARM_COLORS = {"texel": "#2a78d6", "shading": "#eb6834"}


def load(sweep):
    rows = []
    for p in sorted(pathlib.Path(sweep).rglob("rounding/results.json")):
        r = json.loads(p.read_text())
        for arm in ("nearest", "texel", "shading"):
            if arm not in r:
                continue
            rows.append({"material": r["material"], "fmt": r["fmt"], "train_loss": "texel" if r["loss"] == "texel" else "shading",
                         "seed": r["train_seed"], "arm": arm, **{k: v for k, v in r[arm].items()}})
    return pd.DataFrame(rows)


def equivalent_memory(gain_frac, fmt):
    """Extra latent memory that buys the same relative FLIP drop on the milestone
    curve (texel-loss training, mid -> large grid, log-linear in bits)."""
    csv = ROOT / "results" / "milestone" / "results.csv"
    if not csv.exists():
        return None
    d = pd.read_csv(csv)
    d = d[(d.fmt == fmt) & (d.loss_name == "texel")].groupby("grid")[["flip", "bpp_total"]].mean()
    if not {"m", "l"} <= set(d.index):
        return None
    # FLIP falls by a factor (flip_l / flip_m) when bits grow by (bpp_l / bpp_m);
    # assume flip ~ bits^-k between those two points.
    k = -np.log(d.flip["l"] / d.flip["m"]) / np.log(d.bpp_total["l"] / d.bpp_total["m"])
    return (1 / (1 - gain_frac)) ** (1 / k) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--out", default=str(ROOT / "results" / "rounding"))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(args.sweep)
    if d.empty:
        print("no finished runs")
        return
    d.to_csv(out / "runs.csv", index=False)
    key = ["material", "fmt", "train_loss", "seed"]
    w = d.pivot_table(index=key, columns="arm", values=["flip", "flip_unseen", "psnr_shaded", "psnr_unseen"])
    w = w.dropna()
    n = len(w)
    lines = [f"# Rendering-aware rounding: {n} trained models\n",
             "Decoder frozen, block scales unchanged, same storage. FLIP change vs round-to-nearest (negative = better).\n"]

    def rel(metric, arm):
        return 100 * (w[(metric, arm)] / w[(metric, "nearest")] - 1)

    summary = []
    for (fmt, tl), g in w.groupby(level=["fmt", "train_loss"]):
        row = {"format": fmt, "trained with": tl, "models": len(g)}
        for metric, tag in (("flip", "test lights"), ("flip_unseen", "unseen lights")):
            for arm in ("texel", "shading"):
                r = 100 * (g[(metric, arm)] / g[(metric, "nearest")] - 1)
                row[f"{arm}-rounding, {tag}"] = f"{r.mean():+.1f} %"
        row["shading beats texel (unseen)"] = f"{int((g[('flip_unseen', 'shading')] < g[('flip_unseen', 'texel')]).sum())}/{len(g)}"
        summary.append(row)
    lines.append(pd.DataFrame(summary).to_markdown(index=False))

    lines.append("\n## Win counts over all models\n")
    for metric, tag in (("flip", "test lights"), ("flip_unseen", "unseen lights")):
        a = int((w[(metric, "shading")] < w[(metric, "nearest")]).sum())
        b = int((w[(metric, "shading")] < w[(metric, "texel")]).sum())
        c = int((w[(metric, "texel")] < w[(metric, "nearest")]).sum())
        lines.append(f"- {tag}: shading-rounding beats nearest in {a}/{n}, beats texel-rounding in {b}/{n}; texel-rounding beats nearest in {c}/{n}")

    lines.append("\n## Per material (mean over formats, training losses, seeds)\n")
    pm = pd.DataFrame({
        "texel-rounding, test": rel("flip", "texel"), "shading-rounding, test": rel("flip", "shading"),
        "texel-rounding, unseen": rel("flip_unseen", "texel"), "shading-rounding, unseen": rel("flip_unseen", "shading"),
    }).groupby(level="material").mean().round(1)
    lines.append(pm.to_markdown())

    lines.append("\n## Seed spread (shading-rounding FLIP change, test lights)\n")
    sd = rel("flip", "shading").groupby(level=["material", "fmt", "train_loss"]).agg(["min", "max"]).round(1)
    lines.append(f"- largest gap between the two seeds: {(sd['max'] - sd['min']).max():.1f} percentage points")

    lines.append("\n## Equivalent memory (texel-trained models, from the milestone curves)\n")
    for fmt in sorted(w.index.get_level_values("fmt").unique()):
        g = w.xs(("texel",), level=["train_loss"])
        g = g[g.index.get_level_values("fmt") == fmt]
        if len(g) == 0:
            continue
        gain = 1 - (g[("flip", "shading")] / g[("flip", "nearest")]).mean()
        em = equivalent_memory(gain, fmt)
        if em is not None:
            lines.append(f"- {fmt}: FLIP {-100 * gain:+.1f} % equals about {100 * em:.0f} % more latent memory at plain rounding")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # bar chart: mean relative FLIP change per (format, training loss), test vs unseen lights
    groups = [(f, t) for f, t in w.index.droplevel(["material", "seed"]).unique()]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8), sharey=True, facecolor="#fcfcfb")
    for ax, (metric, tag) in zip(axes, (("flip", "test lighting"), ("flip_unseen", "unseen lighting"))):
        style_axes(ax)
        ax.grid(axis="x", visible=False)
        x = np.arange(len(groups))
        for i, arm in enumerate(("texel", "shading")):
            vals = [rel(metric, arm).xs((f, t), level=["fmt", "train_loss"]).mean() for f, t in groups]
            ax.bar(x + (i - 0.5) * 0.38, vals, 0.35, color=ARM_COLORS[arm], label=f"{arm}-aware rounding")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, [f"{f}\n{t} loss" for f, t in groups], fontsize=7)
        ax.set_title(tag, fontsize=9, color=INK)
    axes[0].set_ylabel("FLIP change vs nearest (%)", fontsize=8, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(out / "rounding_flip.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
