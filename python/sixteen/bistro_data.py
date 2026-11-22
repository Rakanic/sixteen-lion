"""Bistro (ORCA v5.2) texture sets as 9-channel compressor targets.

Same layout as sixteen.data (albedo sRGB-encoded, OpenGL normal encoded n*0.5+0.5, AO, roughness,
metalness) at 2048x2048. Bistro stores its maps at different sizes (the AO/roughness/metalness map is
usually 16x16, some normal maps 1x1); they are resampled with the same tiling bilinear lookup the
renderer uses for texture lookups, so the target equals what the original-texture path returns at
texel centres. The source data is block-compressed (BC1/BC3/BC5), so this target is the decoded BC data.

The occlusion channel (R of the specular map) is zero in every Bistro map, i.e. unused; it is
set to 1 (no occlusion), as for materials without a specular map in the renderer.

Names are "bistro/<set>", e.g. "bistro/Paris_Cashregister", which sixteen.data.load_material dispatches here.
"""

import pathlib

import torch

from .render.sources import bilinear_wrap_h
from .render.textures import _read, decode_normal_dx_bc5

TEXTURES = pathlib.Path(__file__).resolve().parents[2] / "data" / "bistro-orca" / "Bistro_v5_2" / "Textures"
RES = 2048


def _to_res(t, res):
    """[h, w, c] float -> [res, res, c], tiling bilinear at texel centres (identity if already res x res)."""
    if t.shape[0] == res and t.shape[1] == res:
        return t
    idx = torch.arange(res, dtype=torch.float32)
    ys, xs = torch.meshgrid(idx, idx, indexing="ij")
    uv = torch.stack([(xs + 0.5) / res, (ys + 0.5) / res], -1).reshape(-1, 2)
    return bilinear_wrap_h(t.half(), uv).reshape(res, res, -1)


def set_files(stem, root=TEXTURES):
    return [root / f"{stem}_{k}.dds" for k in ("BaseColor", "Normal", "Specular")]


def load_bistro_material(stem, res=RES, root=TEXTURES):
    """-> [9, res, res] float32 in [0, 1]. Needs all three maps."""
    base, normal, spec = set_files(stem, root)
    for p in (base, normal, spec):
        if not p.exists():
            raise FileNotFoundError(p)
    albedo = torch.from_numpy(_read(base)[..., :3].astype("float32") / 255.0)
    nrm = decode_normal_dx_bc5(_read(normal))
    arm = torch.from_numpy(_read(spec)[..., :3].astype("float32") / 255.0)
    if float(arm[..., 0].max()) == 0.0:  # the occlusion channel is 0 in all 201 Bistro maps: unused, not black
        arm[..., 0] = 1.0
    t = torch.cat([_to_res(albedo, res), _to_res(nrm, res), _to_res(arm, res)], -1)
    return t.permute(2, 0, 1).contiguous().float()
