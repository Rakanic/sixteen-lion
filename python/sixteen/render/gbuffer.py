"""Camera, rasterization into a G-buffer (nvdiffrast CUDA rasterizer, no display needed).

Rows are in OpenGL order (row 0 = bottom); flip when writing images.
"""

import dataclasses
import math

import nvdiffrast.torch as dr
import torch


@dataclasses.dataclass
class Camera:
    eye: tuple
    target: tuple
    up: tuple = (0.0, 1.0, 0.0)
    fovy: float = 60.0  # degrees
    near: float = 0.05
    far: float = 500.0


def view_matrix(cam, device):
    eye, tgt, up = (torch.tensor(v, dtype=torch.float32, device=device) for v in (cam.eye, cam.target, cam.up))
    f = torch.nn.functional.normalize(tgt - eye, dim=0)
    s = torch.nn.functional.normalize(torch.linalg.cross(f, up), dim=0)
    u = torch.linalg.cross(s, f)
    m = torch.eye(4, device=device)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -torch.stack([s @ eye, u @ eye, -(f @ eye)])
    return m


def proj_matrix(cam, W, H, device, jitter=(0.0, 0.0)):
    """OpenGL perspective; jitter (pixels) shifts the image by that much in x/y."""
    f = 1.0 / math.tan(math.radians(cam.fovy) / 2)
    n, fa = cam.near, cam.far
    p = torch.zeros(4, 4, device=device)
    p[0, 0], p[1, 1] = f * H / W, f
    p[2, 2], p[2, 3], p[3, 2] = (fa + n) / (n - fa), 2 * fa * n / (n - fa), -1.0
    p[0, 2] -= 2 * jitter[0] / W  # x_ndc = P00*x/(-z) - P02
    p[1, 2] -= 2 * jitter[1] / H
    return p


class SceneGPU:
    def __init__(self, scene, device="cuda"):
        t = lambda a, dt=torch.float32: torch.from_numpy(a).to(device=device, dtype=dt)
        self.pos = t(scene.pos)
        self.attrs = torch.cat([t(scene.pos), t(scene.normal), t(scene.tangent), t(scene.bitangent), t(scene.uv)], 1).contiguous()
        self.tri = t(scene.tri, torch.int32).contiguous()
        self.tri_mat = t(scene.tri_material, torch.int32)
        self.device = device


@torch.no_grad()
def rasterize_gbuffer(ctx, sg, cam, W, H, jitter=(0.0, 0.0)):
    dev = sg.device
    vp = proj_matrix(cam, W, H, dev, jitter) @ view_matrix(cam, dev)
    clip = torch.cat([sg.pos, torch.ones_like(sg.pos[:, :1])], 1) @ vp.T
    rast, db = dr.rasterize(ctx, clip[None].contiguous(), sg.tri, (H, W), grad_db=False)
    out, da = dr.interpolate(sg.attrs[None], rast, sg.tri, rast_db=db, diff_attrs=[12, 13])
    tri_id = rast[0, ..., 3].long() - 1
    valid = tri_id >= 0
    mat = sg.tri_mat[tri_id.clamp(min=0)].long()
    mat = torch.where(valid, mat, torch.full_like(mat, -1))
    return dict(
        valid=valid, pos=out[0, ..., 0:3], nrm=out[0, ..., 3:6], tan=out[0, ..., 6:9], bit=out[0, ..., 9:12],
        uv=out[0, ..., 12:14], duv=da[0], mat=mat, tri=tri_id, depth=rast[0, ..., 2],
    )
