"""Deferred renderer: G-buffer (nvdiffrast) -> texture source -> GGX shading -> optional TAA.

    r = Renderer(load_scene("BistroInterior"))
    out = r.render(OriginalSource(r.store, "trilinear"), cam, 1920, 1080)
    out["rgb"]   # [H, W, 3] linear radiance, rows bottom-up
    out["ms"]    # per-stage wall time (synchronized), ms
"""

import time

import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from .gbuffer import SceneGPU, proj_matrix, rasterize_gbuffer, view_matrix
from .shade import shade_pixels
from .textures import TextureStore

SKY = (0.35, 0.45, 0.60)  # linear background radiance


def halton(i, base):
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def jitter_at(frame):
    """Subpixel camera jitter in [-0.5, 0.5)^2 (Halton 2, 3)."""
    return halton(frame + 1, 2) - 0.5, halton(frame + 1, 3) - 0.5


class Renderer:
    def __init__(self, scene, device="cuda"):
        self.scene, self.device = scene, device
        self.sg = SceneGPU(scene, device)
        self.ctx = dr.RasterizeCudaContext(device)
        self.store = TextureStore(scene, device)

    @torch.no_grad()
    def render(self, source, cam, W, H, frame=0, jitter=(0.0, 0.0), **shade_kw):
        ms = {}
        sync = torch.cuda.synchronize
        t = time.perf_counter()
        sync()
        gb = rasterize_gbuffer(self.ctx, self.sg, cam, W, H, jitter)
        sync()
        ms["raster"], t = (time.perf_counter() - t) * 1e3, time.perf_counter()

        sel = gb["valid"].reshape(-1).nonzero()[:, 0]  # pixel ids in raster order: also the STF seeds
        pick = lambda k: gb[k].reshape(H * W, -1)[sel]
        flat = {k: pick(k) for k in ("pos", "nrm", "tan", "bit")}
        mat = gb["mat"].reshape(-1)[sel]
        tex, alpha = source.sample(pick("uv"), pick("duv"), mat, sel, frame)
        sync()
        ms["texture"], t = (time.perf_counter() - t) * 1e3, time.perf_counter()

        eye = torch.tensor(cam.eye, dtype=torch.float32, device=self.device)
        rad = shade_pixels(tex, flat, eye, **shade_kw)
        img = torch.tensor(SKY, device=self.device).repeat(H * W, 1)
        img[sel] = rad
        sync()
        ms["shade"] = (time.perf_counter() - t) * 1e3
        ms["total"] = sum(ms.values())
        return dict(rgb=img.view(H, W, 3), gb=gb, alpha=alpha, ms=ms)

    def accumulate(self, source, cam, W, H, n, start=0, **kw):
        """Mean of n jittered frames (static camera): the converged reference for a source.

        `start` offsets the frame index (jitter sequence and STF seeds), so two accumulations with
        different starts use independent samples.
        """
        acc = None
        for f in range(start, start + n):
            rgb = self.render(source, cam, W, H, frame=f, jitter=jitter_at(f), **kw)["rgb"]
            acc = rgb if acc is None else acc + rgb
        return acc / n


def pixel_center_world(gb, cam, W, H):
    """World position of each pixel *centre* at the depth of the pixel's (jittered) hit point.

    Reprojecting the jittered hit position would resample the history at a fractional pixel
    offset every frame and blur it; TAA reprojects the pixel centre, as motion vectors do.
    """
    dev = gb["pos"].device
    view = view_matrix(cam, dev)
    p = proj_matrix(cam, W, H, dev)
    hit = torch.cat([gb["pos"], torch.ones_like(gb["pos"][..., :1])], -1) @ view.T
    d = -hit[..., 2]
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    nx, ny = 2 * (xs + 0.5) / W - 1, 2 * (ys + 0.5) / H - 1
    v = torch.stack([nx * d / p[0, 0], ny * d / p[1, 1], -d, torch.ones_like(d)], -1)
    return v @ torch.linalg.inv(view).T


class TAA:
    """Temporal accumulation: reproject history by pixel-centre world position, clamp to the 3x3 neighbourhood, blend."""

    def __init__(self, alpha=0.1, clamp=True):
        self.alpha, self.clamp, self.hist, self.prev_vp = alpha, clamp, None, None

    @torch.no_grad()
    def resolve(self, rgb, gb, cam, W, H):
        dev = rgb.device
        vp = proj_matrix(cam, W, H, dev) @ view_matrix(cam, dev)  # unjittered
        out = rgb
        if self.hist is not None:
            world = pixel_center_world(gb, cam, W, H)
            p = world @ self.prev_vp.T
            ndc = p[..., :2] / p[..., 3:4].clamp(min=1e-6)
            ok = gb["valid"] & (p[..., 3] > 0) & (ndc.abs() < 1).all(-1)
            hist = F.grid_sample(self.hist.permute(2, 0, 1)[None], ndc[None], mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            if self.clamp:
                x = rgb.permute(2, 0, 1)[None]
                lo, hi = -F.max_pool2d(-x, 3, 1, 1), F.max_pool2d(x, 3, 1, 1)
                hist = torch.minimum(torch.maximum(hist, lo[0].permute(1, 2, 0)), hi[0].permute(1, 2, 0))
            out = torch.where(ok[..., None], hist * (1 - self.alpha) + rgb * self.alpha, rgb)
        self.hist, self.prev_vp = out, vp
        return out
