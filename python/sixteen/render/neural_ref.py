"""Neural-texture reference source: our compressor's output decoded in PyTorch and fed through the renderer.

A fitted material is decoded once, at every texel centre (the only place train() constrains the model),
into a [H, W, 9] tensor, and that tensor goes through the same atlas path as the original textures
(AtlasSource, "stf": pick one of the four nearest texels with bilinear weights, take its value). The
reference render uses the training target tensor through the identical path, so the two renders differ
only by compression error. This is the PyTorch decode of the stored (fake-quantized, hard-rounded)
latents, not the CUDA kernel.
"""

import json
import pathlib

import torch

from ..model import NeuralTexture
from .textures import pyramid


class TensorSet:
    """Duck-types TextureSet from a [H, W, 9] tensor (albedo, normal, arm); no alpha."""

    def __init__(self, t9, device="cuda"):
        t9 = t9.to(device)
        self.albedo = [l for l in pyramid(t9[..., 0:3])]
        self.normal = [l for l in pyramid(t9[..., 3:6])]
        self.arm = [l for l in pyramid(t9[..., 6:9])]
        self.alpha = None


class OverrideStore:
    """TextureStore look-alike: overridden materials come from `overrides`, the rest from `base`.

    Keeps its own cache so building several atlases from one base store loads each original once
    (TextureAtlas clears `.sets`, which here is only a dummy).
    """

    def __init__(self, base, overrides):
        self.scene, self.base, self.overrides, self.sets = base.scene, base, overrides, {}

    def get(self, i):
        return self.overrides[i] if i in self.overrides else self.base.get(i)


def load_fit(ckpt, device="cuda"):
    """-> (model, meta dict from metrics.json next to the checkpoint)."""
    ckpt = pathlib.Path(ckpt)
    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    res = {"1k": 1024, "2k": 2048, "4k": 4096}[cfg["res"]]
    model = NeuralTexture((res, res), 9, grids=[tuple(g) for g in ck["grids"]], hidden=cfg["hidden"],
                          layers=cfg["layers"], filter=cfg.get("filter", "bilinear"), w1_fmt=cfg.get("w1_fmt"))
    model.load_state_dict(ck["state"])
    model.fmt = ck["fmt"]  # decode with the stored format's rounding, as train.evaluate does
    model.scale_choice = [None] * len(model.latents) if cfg.get("scale_mode", "max") == "max" else ck["scale_choice"]
    meta = json.loads((ckpt.parent / "metrics.json").read_text())
    return model.to(device).eval(), meta


@torch.no_grad()
def decode_texels(model, res):
    """Decoded material at every texel centre: [res, res, 9] in [0, 1]."""
    return model.decode_image(res, res, model.fmt).clamp(0, 1)


def stem_of(m):
    return m.base_color[: -len("_BaseColor.dds")]


def load_all_fits(scene, fit_dir, device="cuda"):
    """Every fitted Bistro material under `fit_dir`: dicts with stem, scene index, model, meta, target [H,W,9], decoded [H,W,9]."""
    import glob

    from ..bistro_data import load_bistro_material

    fit_dir = pathlib.Path(fit_dir)
    by_stem = {stem_of(m): i for i, m in enumerate(scene.materials) if m.base_color}
    fits = []
    for ck in sorted(glob.glob(str(fit_dir / "bistro/*/m/shade4/nvfp4/model.pt"))):
        stem = pathlib.Path(ck).relative_to(fit_dir).parts[1]
        model, meta = load_fit(ck, device)
        target = load_bistro_material(stem).permute(1, 2, 0).contiguous().to(device)
        fits.append(dict(stem=stem, index=by_stem[stem], ckpt=ck, model=model, meta=meta, target=target, decoded=decode_texels(model, target.shape[0])))
    return fits


class MixedSource:
    """Original-atlas lookup for every pixel, then overwrite the fitted materials with the CUDA v1 decoder.

    Decoder v1 picks one random latent tap per grid (latent-space stochastic filtering) and runs the MLP on the
    stored codes; averaged over frames it stays ~20 dB from the correct decode (results/decoder). Its stochastic
    seeds come from the pixel's position in the per-material batch, not its screen index, so seeds repeat across
    materials; fine for a visual, not for a seed-matched comparison.
    """

    def __init__(self, atlas_source, decoders):
        self.src, self.decoders = atlas_source, decoders

    def sample(self, uv, duv, mat, pixel_id, frame=0):
        tex, alpha = self.src.sample(uv, duv, mat, pixel_id, frame)
        uvt = torch.stack([uv[:, 0], 1.0 - uv[:, 1]], -1)  # texture rows run top-down
        for m, dec in self.decoders.items():
            idx = (mat == m).nonzero()[:, 0]
            if idx.numel():
                tex[idx] = dec(uvt[idx].contiguous(), int(frame)).clamp(0, 1)
        return tex, alpha


class MixedSourceV2:
    """Original-atlas lookup for every pixel, then overwrite the fitted materials with CUDA decoder v2.

    v2 does output-space stochastic filtering: each pixel picks one full-resolution texel with bilinear
    weights, decodes it from the stored FP4 codes (layer 1 as two block-scaled FP4 MMAs when `fp4`), and
    TAA averages the colours. Its seeds come from the pixel's index in the batch; to make them equal the
    screen pixel index (so the taps match `AtlasSource(..., "stf")` exactly), uvs are scattered into a
    full-frame buffer and the whole frame is decoded for each material (about 0.25 ms at 1080p), then the
    material's pixels are read back. `ms` accumulates the CUDA time of all decoder launches.
    """

    def __init__(self, atlas_source, decoders, fp4=True):
        self.src, self.decoders, self.fp4 = atlas_source, decoders, fp4
        self.ms = []

    def sample(self, uv, duv, mat, pixel_id, frame=0):
        tex, alpha = self.src.sample(uv, duv, mat, pixel_id, frame)
        uvt = torch.stack([uv[:, 0], 1.0 - uv[:, 1]], -1)  # texture rows run top-down
        n = int(pixel_id.max()) + 1
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for m, dec in self.decoders.items():
            idx = (mat == m).nonzero()[:, 0]
            if idx.numel():
                full = torch.full((n, 2), 0.5, device=uv.device)
                full[pixel_id[idx]] = uvt[idx]
                tex[idx] = dec(full.contiguous(), int(frame), fp4=self.fp4)[pixel_id[idx]].clamp(0, 1)
        t1.record()
        torch.cuda.synchronize()
        self.ms.append(t0.elapsed_time(t1))
        return tex, alpha
