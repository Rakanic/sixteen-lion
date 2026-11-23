"""Scene loading: Bistro FBX -> flat triangle arrays + a material table.

assimp (via pyassimp) flattens the node hierarchy (aiProcess_PreTransformVertices)
and computes tangent frames. The result is cached as .npz + .json next to the
FBX because the FBX import takes 20-60 s.

Conventions kept from the FBX: uv origin is bottom-left (v up). Texture images
are top-down, so texture row = (1 - v) * H (see textures.py). Tangent and
bitangent come from assimp with these uvs, so tangent-space normals use the
OpenGL (green up) convention of sixteen.data after textures.py flips DirectX maps.
"""

import dataclasses
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
BISTRO = ROOT / "data" / "bistro-orca" / "Bistro_v5_2"


@dataclasses.dataclass
class Material:
    name: str
    base_color: str | None  # file names relative to the scene's Textures directory
    normal: str | None
    specular: str | None  # R occlusion, G roughness, B metalness
    emissive: str | None
    roughness_factor: float = 1.0


@dataclasses.dataclass
class Scene:
    pos: np.ndarray  # [V, 3] float32, world space
    normal: np.ndarray  # [V, 3]
    tangent: np.ndarray  # [V, 3]
    bitangent: np.ndarray  # [V, 3]
    uv: np.ndarray  # [V, 2], v up
    tri: np.ndarray  # [T, 3] int32
    tri_material: np.ndarray  # [T] int32
    materials: list
    texture_dir: pathlib.Path

    @property
    def bounds(self):
        return self.pos.min(0), self.pos.max(0)


def _prop(props, key, default=None):
    # pyassimp's PropertyGetter stores (key, semantic) tuples; FBX texture slots use semantic 18
    # (aiTextureType_UNKNOWN), so match on the key alone
    for k, v in dict.items(props):
        if k[0] == key:
            return v
    return default


def _tex_name(mat, *keys):
    for k in keys:
        v = _prop(mat.properties, k)
        if v:
            return os.path.basename(str(v).replace("\\", "/"))
    return None


def rebase_uv(pos, nrm, tan, bit, uv, tri, tri_mat, limit=8.0):
    """Give triangles with huge uvs their own vertices, shifted by an integer.

    Tiled surfaces have uvs up to +-5e4. The rasterizer interpolates in float32,
    where 5e4 leaves ~1e-2 of uv resolution (tens of texels). Subtracting the
    floor of a triangle's smallest uv keeps uv mod 1 unchanged (textures wrap with
    period 1) and makes the interpolated values small. Triangles within +-limit keep
    their shared vertices.
    """
    big = np.abs(uv[tri]).max(axis=(1, 2)) > limit
    if not big.any():
        return pos, nrm, tan, bit, uv, tri, tri_mat
    bt = tri[big]  # [B, 3]
    flat = bt.reshape(-1)
    shift = np.floor(uv[bt].min(axis=1)).astype(np.float32)  # [B, 2]
    new_uv = uv[flat] - np.repeat(shift, 3, axis=0)
    base = len(pos)
    new_tri = base + np.arange(len(flat), dtype=np.int32).reshape(-1, 3)
    out_tri = tri.copy()
    out_tri[big] = new_tri
    cat = lambda a, b: np.concatenate([a, b], 0)
    return (cat(pos, pos[flat]), cat(nrm, nrm[flat]), cat(tan, tan[flat]), cat(bit, bit[flat]), cat(uv, new_uv), out_tri, tri_mat)


def _import(fbx):
    # pyassimp only searches LD_LIBRARY_PATH and system dirs; the conda env's lib dir is neither
    os.environ["LD_LIBRARY_PATH"] = f"{sys.prefix}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    import pyassimp
    from pyassimp import postprocess as pp

    flags = pp.aiProcess_Triangulate | pp.aiProcess_PreTransformVertices | pp.aiProcess_CalcTangentSpace | pp.aiProcess_GenSmoothNormals
    pos, nrm, tan, bit, uv, tri, tri_mat, off = [], [], [], [], [], [], [], 0
    with pyassimp.load(str(fbx), processing=flags) as sc:
        mats = []
        for m in sc.materials:
            p = m.properties
            mats.append(
                Material(
                    name=str(_prop(p, "name", "")),
                    base_color=_tex_name(m, "DiffuseColor|file"),
                    normal=_tex_name(m, "NormalMap|file"),
                    specular=_tex_name(m, "SpecularColor|file"),
                    emissive=_tex_name(m, "EmissiveColor|file"),
                    roughness_factor=float(_prop(p, "roughnessFactor", 1.0)),
                )
            )
        for mesh in sc.meshes:
            n = len(mesh.vertices)
            pos.append(np.asarray(mesh.vertices, np.float32))
            nrm.append(np.asarray(mesh.normals, np.float32))
            tan.append(np.asarray(mesh.tangents, np.float32))
            bit.append(np.asarray(mesh.bitangents, np.float32))
            uv.append(np.asarray(mesh.texturecoords[0][:, :2], np.float32))
            f = np.asarray(mesh.faces, np.int32) + off
            tri.append(f)
            tri_mat.append(np.full(len(f), mesh.materialindex, np.int32))
            off += n
    cat = lambda xs: np.concatenate(xs, 0)
    return (*rebase_uv(cat(pos), cat(nrm), cat(tan), cat(bit), cat(uv), cat(tri), cat(tri_mat)), mats)


def load_scene(name="BistroInterior", root=BISTRO, cache=True):
    root = pathlib.Path(root)
    npz, js = root / f"{name}.cache.npz", root / f"{name}.cache.json"
    if cache and npz.exists() and js.exists():
        z = np.load(npz)
        mats = [Material(**m) for m in json.load(open(js))]
        return Scene(z["pos"], z["normal"], z["tangent"], z["bitangent"], z["uv"], z["tri"], z["tri_material"], mats, root / "Textures")
    arrs = _import(root / f"{name}.fbx")
    pos, nrm, tan, bit, uv, tri, tri_mat, mats = arrs
    if cache:
        np.savez(npz, pos=pos, normal=nrm, tangent=tan, bitangent=bit, uv=uv, tri=tri, tri_material=tri_mat)
        json.dump([dataclasses.asdict(m) for m in mats], open(js, "w"))
    return Scene(pos, nrm, tan, bit, uv, tri, tri_mat, mats, root / "Textures")
