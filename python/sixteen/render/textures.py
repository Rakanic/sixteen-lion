"""Original Bistro textures as float16 mip pyramids, one pyramid per map.

Maps stay at their native resolution (Bistro's specular maps are mostly 16x16,
its base colour and normal maps 2048x2048), each with its own pyramid. Layouts
follow sixteen.data: albedo sRGB-encoded as stored, normal XYZ encoded n*0.5+0.5
in OpenGL (green up) convention, ARM = AO, roughness, metalness.

Bistro stores normals as DirectX BC5 (x, y; z reconstructed), so green is flipped.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _read(path):
    im = Image.open(path)
    im.load()
    return np.asarray(im)


def pyramid(t):
    """[H, W, C] float tensor -> list of float16 levels down to 1x1 (2x2 box filter)."""
    levels = [t.half()]
    x = t.permute(2, 0, 1)[None].float()
    while x.shape[-1] > 1 or x.shape[-2] > 1:
        x = F.avg_pool2d(x, (2 if x.shape[-2] > 1 else 1, 2 if x.shape[-1] > 1 else 1))
        levels.append(x[0].permute(1, 2, 0).contiguous().half())
    return levels


def decode_normal_dx_bc5(arr):
    """uint8 [H, W, >=2] DirectX BC5 normal -> [H, W, 3] encoded OpenGL normal in [0, 1]."""
    xy = torch.from_numpy(arr[..., :2].astype(np.float32) / 255.0) * 2 - 1
    nx, ny = xy[..., 0], -xy[..., 1]  # DirectX -> OpenGL: flip green
    nz = (1 - nx * nx - ny * ny).clamp(min=0).sqrt()
    n = torch.stack([nx, ny, nz], -1)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return n * 0.5 + 0.5


class TextureSet:
    """Pyramids on `device`: albedo [.,.,3], normal [.,.,3], arm [.,.,3], alpha [.,.,1] or None."""

    def __init__(self, texture_dir, mat, device="cuda"):
        d = texture_dir
        base = _read(d / mat.base_color)
        albedo = torch.from_numpy(base[..., :3].astype(np.float32) / 255.0)
        alpha = None
        if base.ndim == 3 and base.shape[-1] == 4 and base[..., 3].min() < 255:
            alpha = torch.from_numpy(base[..., 3:4].astype(np.float32) / 255.0)
        normal = decode_normal_dx_bc5(_read(d / mat.normal)) if mat.normal else torch.tensor([[[0.5, 0.5, 1.0]]])
        if mat.specular:
            arm = torch.from_numpy(_read(d / mat.specular)[..., :3].astype(np.float32) / 255.0)
        else:
            arm = torch.tensor([[[1.0, mat.roughness_factor, 0.0]]])
        self.albedo = [l.to(device) for l in pyramid(albedo)]
        self.normal = [l.to(device) for l in pyramid(normal)]
        self.arm = [l.to(device) for l in pyramid(arm)]
        self.alpha = [l.to(device) for l in pyramid(alpha)] if alpha is not None else None

    def nbytes(self):
        return sum(l.numel() * 2 for maps in (self.albedo, self.normal, self.arm, self.alpha or []) for l in maps)


class TextureStore:
    """Lazy per-material cache of TextureSets."""

    def __init__(self, scene, device="cuda"):
        self.scene, self.device, self.sets = scene, device, {}

    def get(self, i):
        if i not in self.sets:
            self.sets[i] = TextureSet(self.scene.texture_dir, self.scene.materials[i], self.device)
        return self.sets[i]


class TextureAtlas:
    """All materials' pyramids packed into one flat fp16 buffer per map, with a descriptor table.

    desc[map][material, level] = (offset, width, height) in texels (levels past the
    last repeat it); nlev[map][material] = number of levels. Lets one vectorized
    gather serve every pixel regardless of material (no Python loop per material).
    """

    MAPS = ("albedo", "normal", "arm", "alpha")

    def __init__(self, store, device="cuda"):
        n = len(store.scene.materials)
        self.flat, self.desc, self.nlev = {}, {}, {}
        for name in self.MAPS:
            chunks, descs, nl, off = [], [], [], 0
            for m in range(n):
                levels = getattr(store.get(m), name)
                if levels is None:  # opaque
                    levels = [torch.ones(1, 1, 1, dtype=torch.float16, device=device)]
                d = []
                for l in levels:
                    h, w, c = l.shape
                    d.append((off, w, h))
                    chunks.append(l.reshape(-1, c))
                    off += h * w
                descs.append(d)
                nl.append(len(levels))
            lmax = max(nl)
            descs = [d + [d[-1]] * (lmax - len(d)) for d in descs]
            self.flat[name] = torch.cat(chunks)
            self.desc[name] = torch.tensor(descs, dtype=torch.long, device=device)
            self.nlev[name] = torch.tensor(nl, dtype=torch.long, device=device)
        store.sets.clear()  # the atlas owns the texels now
        torch.cuda.empty_cache()

    def nbytes(self):
        return sum(f.numel() * 2 for f in self.flat.values())
