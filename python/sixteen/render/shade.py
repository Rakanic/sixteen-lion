"""Per-pixel GGX shading of 9-channel material texels, using sixteen.shading.ggx_radiance.

Same BRDF, same constant sky term, same tone map as the compressor's training loss
and FLIP evaluation, so renderer numbers are comparable with results/. The
tangent frame comes from the mesh; light and view are rotated into it because
ggx_radiance works in tangent space.
"""

import torch
import torch.nn.functional as F

from ..shading import AMBIENT, ggx_radiance, tonemap, unpack

SUN = (0.35, 0.85, 0.40)


def tangent_frame(gb, view_dir):
    """Orthonormal (T, B, N) per pixel; back faces are flipped toward the viewer."""
    n = F.normalize(gb["nrm"], dim=-1)
    t = F.normalize(gb["tan"] - n * (n * gb["tan"]).sum(-1, keepdim=True), dim=-1)
    b = torch.linalg.cross(n, t)
    b = b * torch.sign((b * gb["bit"]).sum(-1, keepdim=True) + 1e-12)
    back = (n * view_dir).sum(-1, keepdim=True) < 0
    return t, torch.where(back, -b, b), torch.where(back, -n, n)


def shade_pixels(tex9, gb_sel, eye, sun=SUN, intensity=3.0, ambient=AMBIENT):
    """tex9 [N, 9] with gb_sel = dict of per-pixel [N, .] tensors -> linear radiance [N, 3]."""
    v = F.normalize(eye - gb_sel["pos"], dim=-1)
    t, b, n = tangent_frame(gb_sel, v)
    l = F.normalize(torch.tensor(sun, device=v.device), dim=0).expand_as(v)
    ts = lambda x: torch.stack([(x * t).sum(-1), (x * b).sum(-1), (x * n).sum(-1)], -1)
    rad = ggx_radiance(tex9, ts(l), ts(v), light=intensity, ambient=False)
    albedo, _, ao, _, _ = unpack(tex9)
    return rad + ambient * albedo * ao


def to_display(linear_rgb):
    return tonemap(linear_rgb)
