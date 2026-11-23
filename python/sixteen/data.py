"""PBR materials as one [C, H, W] tensor in [0, 1].

Channel layout (CHANNELS): albedo RGB (sRGB-encoded, as stored), tangent-space
normal XYZ (encoded as n * 0.5 + 0.5, OpenGL convention), AO, roughness,
metalness.
"""

import pathlib

import imageio.v3 as iio
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2] / "data" / "polyhaven"
CHANNELS = ["albedo_r", "albedo_g", "albedo_b", "normal_x", "normal_y", "normal_z", "ao", "rough", "metal"]
GROUPS = {"albedo": slice(0, 3), "normal": slice(3, 6), "arm": slice(6, 9)}


def _read(path):
    img = iio.imread(path)
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    img = img.astype(np.float32) / scale
    if img.ndim == 2:
        img = img[..., None]
    return img[..., :3]


def load_material(name, res="2k", root=ROOT):
    if str(name).startswith("bistro/"):  # Bistro sets live in sixteen.bistro_data
        from .bistro_data import load_bistro_material

        return load_bistro_material(name.split("/", 1)[1])
    d = pathlib.Path(root) / name
    maps = [_read(d / f"{m}_{res}.png") for m in ("diffuse", "normal", "arm")]
    return torch.from_numpy(np.concatenate(maps, axis=-1)).permute(2, 0, 1).contiguous()
