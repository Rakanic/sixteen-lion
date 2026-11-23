"""Check the normal-map green convention against the albedo (reproduces the claim in docs/bistro-inventory.md).

If albedo and normal map describe one surface (dark grout low, stones high), the normal's x and y
components correlate with the negative albedo gradient with the SAME sign when y is OpenGL (green up).
Over Bistro exterior materials with 2048^2 maps and a visible relation (|corr_x| > 0.15), this counts how
often y agrees with x after our DirectX -> OpenGL flip. Last run: 42 of 48 (unflipped would be 6 of 48).
It assumes u increases to the right, and it is a proxy: foliage, where albedo is a poor height cue,
accounts for several of the disagreements.

    pixi run python tools/check_normal_convention.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bistro_inventory import dds_info  # noqa: E402
from sixteen.render.scene import load_scene  # noqa: E402
from sixteen.render.textures import _read, decode_normal_dx_bc5  # noqa: E402


def corr(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1])


def main():
    s = load_scene("BistroExterior")
    td = s.texture_dir
    rows = []
    for m in s.materials:
        if not (m.base_color and m.normal):
            continue
        if dds_info(td / m.base_color)["w"] < 2048 or dds_info(td / m.normal)["w"] < 2048:
            continue
        base = _read(td / m.base_color)[..., :3].astype(np.float32) / 255
        lum = torch.from_numpy(base @ np.array([0.299, 0.587, 0.114], np.float32))
        nm = decode_normal_dx_bc5(_read(td / m.normal)) * 2 - 1  # OpenGL after our flip
        pool = lambda t: F.avg_pool2d(t[None, None], 8)[0, 0]
        lum, nx, ny = pool(lum), pool(nm[..., 0]), pool(nm[..., 1])
        d_dx = (lum[:, 2:] - lum[:, :-2])[1:-1]
        d_dup = -(lum[2:] - lum[:-2])[:, 1:-1]  # image rows run top-down, so "up" is decreasing row
        rows.append((m.name, corr(nx[1:-1, 1:-1], -d_dx), corr(ny[1:-1, 1:-1], -d_dup)))
    rows = [r for r in rows if abs(r[1]) > 0.15]
    agree = sum(1 for r in rows if r[1] * r[2] > 0)
    print(f"textures with |corr_x| > 0.15: {len(rows)}")
    print(f"y agrees with x after the flip: {agree} of {len(rows)} (unflipped would agree in {len(rows) - agree})")
    for r in sorted(rows, key=lambda r: -abs(r[1]))[:10]:
        print("  %-45s corr_x %+.2f  corr_y %+.2f" % r)


if __name__ == "__main__":
    main()
