"""Bar chart of decode time per 1080p frame for every decoder in v2_baselines.json.

    pixi run python tools/plot_decoder_baselines.py   # -> results/decoder/speedup.{png,pdf}
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch

from sixteen.figures import INK, MUTED, style_axes

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAY, BLUE, ORANGE = "#8a8984", "#2a78d6", "#eb6834"
LABELS = {  # data key -> (axis label, color)
    "pytorch eager fp32 (CUDA cores)": ("PyTorch, fp32", GRAY),
    "pytorch eager fp32, TF32 Tensor Cores": ("PyTorch, fp32 (TF32 Tensor Cores)", GRAY),
    "pytorch eager fp16 (Tensor Cores)": ("PyTorch, fp16 (Tensor Cores)", GRAY),
    "pytorch compiled fp32": ("PyTorch compiled, fp32", GRAY),
    "pytorch compiled fp16 (Tensor Cores)": ("PyTorch compiled, fp16 (Tensor Cores)", GRAY),
    "CUDA FP16-unpack (ours)": ("Ours, fused: FP16-unpack", BLUE),
    "CUDA FP4-direct (ours)": ("Ours, fused: FP4-direct", ORANGE),
}


def main():
    r = json.loads((ROOT / "results/decoder/v2_baselines.json").read_text())
    ms = r["ms"]
    fp4 = ms["CUDA FP4-direct (ours)"]
    keys = sorted(ms, key=lambda k: ms[k], reverse=True)  # slowest on top
    fig, ax = plt.subplots(figsize=(6.4, 3.0), facecolor="#fcfcfb")
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    y = range(len(keys))
    ax.barh(list(y), [ms[k] for k in keys], height=0.62, color=[LABELS[k][1] for k in keys])
    ax.set_xscale("log")
    ax.set_xlim(0.1, 80)
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.invert_yaxis()
    ax.set_yticks(list(y), [LABELS[k][0] for k in keys], fontsize=8, color=INK)
    for i, k in enumerate(keys):
        txt = f"{ms[k]:.3f} ms" if ms[k] < 1 else f"{ms[k]:.1f} ms"
        if k != "CUDA FP4-direct (ours)":
            s = ms[k] / fp4
            txt += f"   {s:.2f}x" if s < 10 else f"   {s:.1f}x"
        ax.text(ms[k] * 1.12, i, txt, va="center", fontsize=7.5, color=INK)
    ax.set_xlabel("decode time per 1920x1080 frame (ms, log scale); labels: time and FP4-direct speedup",
                  fontsize=7.5, color=MUTED)
    ax.legend(handles=[Patch(color=GRAY, label="PyTorch, same work"), Patch(color=BLUE, label="ours, FP16 Tensor Cores"),
                       Patch(color=ORANGE, label="ours, FP4 Tensor Cores")],
              loc="lower right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"results/decoder/speedup.{ext}", dpi=220, bbox_inches="tight")
    print("wrote results/decoder/speedup.{png,pdf}")


if __name__ == "__main__":
    main()
