"""Differentiable GGX shading of decoded materials.

Used two ways: as the shading-aware training loss (random light and view per
sample) and for evaluation (full images under fixed light/view setups, scored
with PSNR and FLIP against the same shading of the uncompressed textures).

Inputs are material texels in the CHANNELS layout of sixteen.data, shape
[..., 9]. The surface is a flat plane, so the tangent-space normal is the
shading normal.
"""

import math

import torch

AMBIENT = 0.03  # constant sky term, scaled by albedo and AO


def srgb_to_linear(c):
    c = c.clamp(0, 1)
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = c.clamp(0, 1)
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def unpack(tex):
    albedo = srgb_to_linear(tex[..., 0:3])
    n = tex[..., 3:6] * 2 - 1
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    ao = tex[..., 6:7].clamp(0, 1)
    rough = tex[..., 7:8].clamp(0.045, 1)
    metal = tex[..., 8:9].clamp(0, 1)
    return albedo, n, ao, rough, metal


def ggx_radiance(tex, l, v, light=3.0, ambient=True, brdf="ggx"):
    """Outgoing radiance (linear RGB) for unit directions l, v of shape [..., 3].

    brdf="phong" swaps the GGX distribution and Smith term for normalized
    Blinn-Phong (exponent 2/alpha^2 - 2, implicit visibility); it is only used
    to evaluate under a shading model the training never saw.
    """
    albedo, n, ao, rough, metal = unpack(tex)
    h = l + v
    h = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    ndl = (n * l).sum(-1, keepdim=True).clamp(min=0)
    ndv = (n * v).sum(-1, keepdim=True).clamp(min=1e-4)
    ndh = (n * h).sum(-1, keepdim=True).clamp(min=0)
    vdh = (v * h).sum(-1, keepdim=True).clamp(min=0)
    a2 = (rough * rough) ** 2
    f0 = 0.04 * (1 - metal) + albedo * metal
    f = f0 + (1 - f0) * (1 - vdh) ** 5
    if brdf == "ggx":
        d = a2 / (math.pi * (ndh * ndh * (a2 - 1) + 1) ** 2)
        k = (rough + 1) ** 2 / 8
        g = (ndl / (ndl * (1 - k) + k)) * (ndv / (ndv * (1 - k) + k))
        spec = d * g * f / (4 * ndl.clamp(min=1e-4) * ndv)
    else:
        e = (2 / a2.clamp(min=1e-4) - 2).clamp(1, 1e4)
        spec = (e + 8) / (8 * math.pi) * ndh.clamp(min=1e-6) ** e * f
    diff = albedo * (1 - metal) / math.pi
    out = (diff + spec) * ndl * light
    return out + AMBIENT * albedo * ao if ambient else out


def random_hemisphere(n, device, min_z=0.1):
    """Unit vectors on the upper hemisphere (z >= min_z), uniform in solid angle."""
    z = torch.rand(n, 1, device=device) * (1 - min_z) + min_z
    phi = torch.rand(n, 1, device=device) * 2 * math.pi
    r = (1 - z * z).sqrt()
    return torch.cat([r * phi.cos(), r * phi.sin(), z], dim=-1)


def tonemap(x):
    """Reinhard then sRGB: the display image that FLIP and PSNR see."""
    return linear_to_srgb(x / (1 + x))


def _dir(elev_deg, azim_deg):
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    return torch.tensor([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


# Fixed evaluation setups: (light elevation, light azimuth, view elevation, view azimuth).
# Grazing light exposes normal error; the near-mirror setup exposes roughness error.
EVAL_SETUPS = {
    "overhead": (70, 30, 90, 0),
    "grazing": (15, 0, 60, 180),
    "specular": (45, 0, 45, 180),
}


def shade_image(tex_hwc, setup):
    le, la, ve, va = EVAL_SETUPS[setup]
    l = _dir(le, la).to(tex_hwc)
    v = _dir(ve, va).to(tex_hwc)
    return tonemap(ggx_radiance(tex_hwc, l.expand_as(tex_hwc[..., :3]), v.expand_as(tex_hwc[..., :3])))


# Lighting the training never sees (it samples one directional light of
# intensity 3 under GGX): many lights at once, 4x the light, and another BRDF.
UNSEEN_SETUPS = {
    "sky": dict(view=(50, 135)),  # 64 directional lights, brighter toward the zenith
    "bright": dict(light=(30, 90), view=(55, 270), intensity=12.0),
    "phong": dict(light=(40, 200), view=(70, 20), brdf="phong"),
}


def _sky_lights(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.rand(n, 1, generator=g) * 0.95 + 0.05
    phi = torch.rand(n, 1, generator=g) * 2 * math.pi
    r = (1 - z * z).sqrt()
    dirs = torch.cat([r * phi.cos(), r * phi.sin(), z], dim=-1)
    w = 0.3 + 0.7 * z[:, 0]
    return dirs, 3.0 * w / w.sum()


def shade_image_unseen(tex_hwc, setup):
    cfg = UNSEEN_SETUPS[setup]
    v = _dir(*cfg["view"]).to(tex_hwc).expand_as(tex_hwc[..., :3])
    if setup == "sky":
        dirs, ws = _sky_lights()
        out = None
        for d, w in zip(dirs.to(tex_hwc), ws.tolist()):
            r = ggx_radiance(tex_hwc, d.expand_as(v), v, light=w, ambient=False)
            out = r if out is None else out + r
        albedo, _, ao, _, _ = unpack(tex_hwc)
        return tonemap(out + AMBIENT * albedo * ao)
    l = _dir(*cfg["light"]).to(tex_hwc).expand_as(v)
    return tonemap(ggx_radiance(tex_hwc, l, v, light=cfg.get("intensity", 3.0), brdf=cfg.get("brdf", "ggx")))
