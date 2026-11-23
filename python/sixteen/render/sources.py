"""Texture sources: uv + footprint + material id -> 9 material channels (+ alpha).

`original`      trilinear (isotropic) lookup in the mip pyramids; deterministic.
`original-stf`  stochastic bilinear tap at level 0 (same hash and tap rule as the
                CUDA decoder); one texel per pixel per frame, averaged over
                frames by TAA / accumulation. With a jittered camera this
                converges to the box-filtered pixel footprint.
"""

import torch

from ..decoder import M32, _pcg, stf_texels


def bilinear_wrap_h(grid, uv):
    """grid [h, w, c] float16, uv [N, 2] (tiling) -> [N, c] float32; only the 4 taps are converted."""
    h, w, c = grid.shape
    x = uv[:, 0] * w - 0.5
    y = uv[:, 1] * h - 0.5
    x0, y0 = torch.floor(x), torch.floor(y)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0, y0 = x0.long() % w, y0.long() % h
    x1, y1 = (x0 + 1) % w, (y0 + 1) % h
    flat = grid.reshape(-1, c)
    g = lambda yy, xx: flat[yy * w + xx].float()
    top = g(y0, x0) * (1 - fx) + g(y0, x1) * fx
    bot = g(y1, x0) * (1 - fx) + g(y1, x1) * fx
    return top * (1 - fy) + bot * fy


def sample_trilinear(levels, uv, duv):
    """levels: pyramid list; uv [N, 2]; duv [N, 4] = (du/dx, du/dy, dv/dx, dv/dy) in uv units per pixel."""
    h, w = levels[0].shape[:2]
    rho = torch.maximum(torch.hypot(duv[:, 0] * w, duv[:, 2] * h), torch.hypot(duv[:, 1] * w, duv[:, 3] * h))
    lod = torch.log2(rho.clamp(min=1.0)).clamp(0, len(levels) - 1)
    l0 = torch.floor(lod).long()
    fr = (lod - l0)[:, None]
    l1 = (l0 + 1).clamp(max=len(levels) - 1)
    out = torch.zeros(uv.shape[0], levels[0].shape[-1], device=uv.device)
    for l in torch.unique(torch.cat([l0, l1])).tolist():
        for sel, wgt in (((l0 == l), 1 - fr), ((l1 == l) & (l0 != l), fr)):
            idx = sel.nonzero()[:, 0]
            if idx.numel():
                out[idx] += wgt[idx] * bilinear_wrap_h(levels[l], uv[idx])
    return out


def sample_stf(levels, uv, seed):
    h, w, c = levels[0].shape
    return levels[0].reshape(-1, c)[stf_texels(uv, w, h, seed)].float()


def stf_seed(pixel_id, frame):
    """Per-pixel seed; identical to decoder.pixel_seeds for pixel_id = arange(n)."""
    return _pcg(((pixel_id.long() * 2654435761) & M32) ^ (frame & M32))


class OriginalSource:
    def __init__(self, store, mode="trilinear"):
        assert mode in ("trilinear", "stf")
        self.store, self.mode = store, mode

    def sample(self, uv, duv, mat, pixel_id, frame=0):
        """uv [N, 2] in scene convention (v up); returns tex9 [N, 9] and alpha [N] (1 if opaque)."""
        uv = torch.stack([uv[:, 0], 1.0 - uv[:, 1]], -1)  # texture rows run top-down
        duv = torch.stack([duv[:, 0], duv[:, 1], -duv[:, 2], -duv[:, 3]], -1)
        n = uv.shape[0]
        tex = torch.empty(n, 9, device=uv.device)
        alpha = torch.ones(n, device=uv.device)
        seed = stf_seed(pixel_id, frame) if self.mode == "stf" else None
        order = torch.argsort(mat)
        counts = torch.bincount(mat, minlength=int(mat.max()) + 1).tolist()
        start = 0
        for m, cnt in enumerate(counts):
            if cnt == 0:
                continue
            idx = order[start : start + cnt]
            start += cnt
            ts = self.store.get(m)
            if self.mode == "trilinear":
                s = lambda lv: sample_trilinear(lv, uv[idx], duv[idx])
            else:
                s = lambda lv: sample_stf(lv, uv[idx], seed[idx])
            tex[idx, 0:3] = s(ts.albedo)
            tex[idx, 3:6] = s(ts.normal)
            tex[idx, 6:9] = s(ts.arm)
            if ts.alpha is not None:
                alpha[idx] = s(ts.alpha)[:, 0]
        return tex, alpha


def _bilinear_pp(flat, off, w, h, uv):
    """Bilinear tiling lookup with per-pixel (offset, width, height) into a flat texel buffer."""
    x = uv[:, 0] * w - 0.5
    y = uv[:, 1] * h - 0.5
    x0, y0 = torch.floor(x), torch.floor(y)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0, y0 = x0.long(), y0.long()
    xa, xb, ya, yb = x0 % w, (x0 + 1) % w, y0 % h, (y0 + 1) % h
    g = lambda yy, xx: flat[off + yy * w + xx].float()
    top = g(ya, xa) * (1 - fx) + g(ya, xb) * fx
    bot = g(yb, xa) * (1 - fx) + g(yb, xb) * fx
    return top * (1 - fy) + bot * fy


class AtlasSource:
    """OriginalSource semantics (trilinear / stf) as one vectorized gather over a TextureAtlas."""

    def __init__(self, atlas, mode="trilinear"):
        assert mode in ("trilinear", "stf")
        self.atlas, self.mode = atlas, mode

    def _map(self, name, mat, uv, duv, seed):
        a = self.atlas
        flat, desc, nlev = a.flat[name], a.desc[name], a.nlev[name]
        d0 = desc[mat, 0]
        if self.mode == "stf":
            w, h = d0[:, 1], d0[:, 2]
            t = stf_texels(uv, w, h, seed)
            return flat[d0[:, 0] + t].float()
        w0, h0 = d0[:, 1].float(), d0[:, 2].float()
        rho = torch.maximum(torch.hypot(duv[:, 0] * w0, duv[:, 2] * h0), torch.hypot(duv[:, 1] * w0, duv[:, 3] * h0))
        top = (nlev[mat] - 1)
        lod = torch.minimum(torch.log2(rho.clamp(min=1.0)), top.float())
        l0 = torch.floor(lod).long()
        fr = (lod - l0)[:, None]
        l1 = torch.minimum(l0 + 1, top)
        d_a, d_b = desc[mat, l0], desc[mat, l1]
        lo = _bilinear_pp(flat, d_a[:, 0], d_a[:, 1], d_a[:, 2], uv)
        hi = _bilinear_pp(flat, d_b[:, 0], d_b[:, 1], d_b[:, 2], uv)
        return lo * (1 - fr) + hi * fr

    def sample(self, uv, duv, mat, pixel_id, frame=0):
        uv = torch.stack([uv[:, 0], 1.0 - uv[:, 1]], -1)
        duv = torch.stack([duv[:, 0], duv[:, 1], -duv[:, 2], -duv[:, 3]], -1)
        seed = stf_seed(pixel_id, frame) if self.mode == "stf" else None
        tex = torch.cat([self._map(n, mat, uv, duv, seed) for n in ("albedo", "normal", "arm")], -1)
        alpha = self._map("alpha", mat, uv, duv, seed)[:, 0]
        return tex, alpha
