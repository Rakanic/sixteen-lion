"""Renderer tests. GPU tests use a synthetic scene (no Bistro needed); Bistro tests skip if the data is absent."""

import math

import numpy as np
import pytest
import torch

pytest.importorskip("nvdiffrast")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from sixteen.render.gbuffer import Camera, SceneGPU, rasterize_gbuffer
from sixteen.render.renderer import TAA, Renderer, jitter_at
from sixteen.render.scene import BISTRO, Material, Scene, rebase_uv
from sixteen.render.shade import shade_pixels
from sixteen.render.sources import AtlasSource, OriginalSource, bilinear_wrap_h
from sixteen.render.textures import TextureAtlas, pyramid
from sixteen.shading import AMBIENT, ggx_radiance, unpack
from sixteen.decoder import pixel_seeds
from sixteen.render.sources import stf_seed


def quad_scene():
    """Two unit-ish quads in the z = 0 plane, x in [-1, 0] (material 0) and [0, 1] (material 1), uv 0..1 on each."""
    pos, uv, tri, mat = [], [], [], []
    for i, x0 in enumerate((-1.0, 0.0)):
        b = len(pos)
        pos += [[x0, -1, 0], [x0 + 1, -1, 0], [x0 + 1, 1, 0], [x0, 1, 0]]
        uv += [[0, 0], [1, 0], [1, 1], [0, 1]]
        tri += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
        mat += [i, i]
    n = len(pos)
    f = lambda a: np.asarray(a, np.float32)
    mats = [Material(f"m{i}", None, None, None, None) for i in range(2)]
    return Scene(
        f(pos), np.tile(f([0, 0, 1]), (n, 1)), np.tile(f([1, 0, 0]), (n, 1)), np.tile(f([0, 1, 0]), (n, 1)), f(uv),
        np.asarray(tri, np.int32), np.asarray(mat, np.int32), mats, None,
    )


CAM = Camera(eye=(0, 0, 2), target=(0, 0, 0), fovy=90, near=0.1, far=10)


@cuda
def test_gbuffer_uv_derivatives_and_materials():
    r = Renderer(quad_scene())
    W = H = 256
    gb = rasterize_gbuffer(r.ctx, r.sg, CAM, W, H)
    # fov 90, eye at distance 2: the 2-unit wide pair of quads spans 1/2 of the NDC width = 128 px, i.e. 64 px per quad
    row = H // 2
    assert gb["valid"][row, W // 2 - 60 : W // 2 + 60].all()
    assert not gb["valid"][row, 10]
    left, right = gb["mat"][row, W // 2 - 32], gb["mat"][row, W // 2 + 32]
    assert (left.item(), right.item()) == (0, 1)
    # each quad is 1 unit wide (64 px) and 2 units tall (128 px): du/dx = 1/64, dv/dy = 1/128; cross terms vanish
    d = gb["duv"][row, W // 2 - 32]
    assert d[0].item() == pytest.approx(1 / 64, rel=1e-3)
    assert d[3].item() == pytest.approx(1 / 128, rel=1e-3)
    assert abs(d[1].item()) < 1e-6 and abs(d[2].item()) < 1e-6
    # pixel centres: the pixel left of the seam has u just under 1 on quad 0
    uv = gb["uv"][row, W // 2 - 1]
    assert uv[0].item() == pytest.approx(1 - 0.5 / 64, abs=2e-3)
    assert uv[1].item() == pytest.approx(0.5, abs=1 / 128)


@cuda
def test_jitter_shifts_uv_by_the_jitter_amount():
    r = Renderer(quad_scene())
    W = H = 256
    a = rasterize_gbuffer(r.ctx, r.sg, CAM, W, H, jitter=(0.0, 0.0))["uv"][H // 2, W // 2 + 32]
    b = rasterize_gbuffer(r.ctx, r.sg, CAM, W, H, jitter=(0.5, 0.0))["uv"][H // 2, W // 2 + 32]
    # jitter shifts the *image* by +0.5 px in x (documented in proj_matrix), so the surface seen at a fixed
    # pixel is 0.5 px further left: u drops by 0.5 * du/dx = 0.5/64
    assert (b[0] - a[0]).item() == pytest.approx(-0.5 / 64, rel=2e-2)
    assert (b[1] - a[1]).item() == pytest.approx(0.0, abs=1e-5)


def _rot_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


@cuda
def test_shading_is_rotation_invariant_and_matches_ggx_radiance():
    torch.manual_seed(0)
    dev = "cuda"
    n = 64
    tex = torch.rand(n, 9, device=dev)
    pos = torch.randn(n, 3, device=dev)
    frame = lambda R: dict(pos=pos @ R.T, nrm=torch.tensor([0.0, 0, 1], device=dev).expand(n, 3) @ R.T,
                           tan=torch.tensor([1.0, 0, 0], device=dev).expand(n, 3) @ R.T,
                           bit=torch.tensor([0.0, 1, 0], device=dev).expand(n, 3) @ R.T)
    eye = torch.tensor([0.3, -0.2, 3.0], device=dev)
    sun = (0.35, 0.85, 0.4)
    base = shade_pixels(tex, frame(torch.eye(3, device=dev)), eye, sun=sun)
    R = _rot_y(70).to(dev)
    rot = shade_pixels(tex, frame(R), eye @ R.T, sun=tuple((torch.tensor(sun, device=dev) @ R.T).tolist()))
    assert torch.allclose(base, rot, atol=1e-5)

    # identity frame: tangent space is world space, so it must equal ggx_radiance on the same vectors
    v = torch.nn.functional.normalize(eye - pos, dim=-1)
    l = torch.nn.functional.normalize(torch.tensor(sun, device=dev), dim=0).expand(n, 3)
    albedo, _, ao, _, _ = unpack(tex)
    ref = ggx_radiance(tex, l, v, light=3.0, ambient=False) + AMBIENT * albedo * ao
    assert torch.allclose(base, ref, atol=1e-5)


def _fake_store(device="cuda", seed=0):
    g = torch.Generator().manual_seed(seed)
    sizes = [(32, 32, 8, 4), (16, 16, 16, 1)]  # (albedo res, normal res, arm res, ...)

    class Set:
        pass

    sets = {}
    for m, (a, nr, ar, _) in enumerate(sizes):
        s = Set()
        mk = lambda r, c: pyramid(torch.rand(r, r, c, generator=g))
        s.albedo = [l.to(device) for l in mk(a, 3)]
        s.normal = [l.to(device) for l in mk(nr, 3)]
        s.arm = [l.to(device) for l in mk(ar, 3)]
        s.alpha = None if m == 0 else [l.to(device) for l in mk(16, 1)]
        sets[m] = s

    class Store:
        pass

    st = Store()
    st.scene = type("S", (), {"materials": [None, None]})()
    st.sets = sets
    st.get = lambda m: sets[m]
    return st, dict(sets)


@cuda
@pytest.mark.parametrize("mode", ["trilinear", "stf"])
def test_atlas_source_matches_per_material_loop(mode):
    st, sets = _fake_store()
    atlas = TextureAtlas(st)
    keep = type("K", (), {})()
    keep.get = lambda m: sets[m]
    n = 4000
    g = torch.Generator(device="cuda").manual_seed(1)
    uv = torch.rand(n, 2, device="cuda", generator=g) * 3 - 1  # includes tiling and negatives
    duv = torch.rand(n, 4, device="cuda", generator=g) * 0.2 * torch.tensor([1, 1, 1, 1], device="cuda")
    mat = torch.randint(0, 2, (n,), device="cuda", generator=g)
    pid = torch.arange(n, device="cuda")
    t_a, a_a = AtlasSource(atlas, mode).sample(uv, duv, mat, pid, frame=3)
    t_l, a_l = OriginalSource(keep, mode).sample(uv, duv, mat, pid, frame=3)
    assert torch.allclose(t_a, t_l, atol=1e-3), (t_a - t_l).abs().max()
    assert torch.allclose(a_a, a_l, atol=1e-3)


@cuda
def test_stf_average_converges_to_bilinear():
    st, sets = _fake_store()
    atlas = TextureAtlas(st)
    n = 512
    g = torch.Generator(device="cuda").manual_seed(2)
    uv = torch.rand(n, 2, device="cuda", generator=g)
    mat = torch.zeros(n, dtype=torch.long, device="cuda")
    duv = torch.zeros(n, 4, device="cuda")
    pid = torch.arange(n, device="cuda")
    stf = AtlasSource(atlas, "stf")
    acc = sum(stf.sample(uv, duv, mat, pid, frame=f)[0] for f in range(2048)) / 2048
    ref = bilinear_wrap_h(sets[0].albedo[0], torch.stack([uv[:, 0], 1 - uv[:, 1]], -1))
    # the unbiasedness claim in docs/renderer-spec.md D4: mean of stochastic taps = bilinear interpolation
    assert (acc[:, :3] - ref).abs().max() < 0.03


def test_stf_seed_matches_decoder_seeds():
    pid = torch.arange(1000)
    s0, _ = pixel_seeds(1000, 7, "cpu")
    assert torch.equal(stf_seed(pid, 7), s0)


def test_rebase_uv_preserves_uv_mod_one_and_shrinks_range():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(6, 3)).astype(np.float32)
    uv = np.array([[0.2, 0.3], [0.8, 0.3], [0.5, 0.9], [50000.25, -30000.5], [50003.75, -30000.5], [50001.5, -29996.0]], np.float32)
    tri = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
    out = rebase_uv(pos, pos, pos, pos, uv, tri, np.zeros(2, np.int32), limit=8.0)
    o_uv, o_tri = out[4], out[5]
    assert np.abs(o_uv[o_tri[1]]).max() < 8  # the big triangle now has small uvs
    assert np.array_equal(o_tri[0], tri[0])  # the small one keeps its shared vertices
    diff = o_uv[o_tri[1]] - uv[tri[1]]
    assert np.allclose(diff, np.round(diff), atol=0)  # shifted by integers only


@cuda
def test_taa_static_camera_is_identity():
    r = Renderer(quad_scene())
    W = H = 128
    gb = rasterize_gbuffer(r.ctx, r.sg, CAM, W, H)
    img = torch.rand(H, W, 3, device="cuda")
    taa = TAA(alpha=0.1)
    out0 = taa.resolve(img, gb, CAM, W, H)
    out1 = taa.resolve(img, gb, CAM, W, H)
    assert torch.allclose(out1, img, atol=1e-5)  # history == current, so any blend returns it
    assert torch.equal(out0, img)


def test_halton_jitter_is_low_discrepancy_and_in_range():
    j = np.array([jitter_at(f) for f in range(64)])
    assert j.min() >= -0.5 and j.max() < 0.5
    assert abs(j.mean(0)).max() < 0.02


bistro = pytest.mark.skipif(not (BISTRO / "BistroInterior.cache.npz").exists(), reason="Bistro not downloaded / imported")


@bistro
def test_bistro_interior_matches_readme_counts_and_all_textures_exist():
    from sixteen.render.scene import load_scene

    s = load_scene("BistroInterior")
    assert len(s.tri) == 1_046_609  # README.txt: "Interior: 1,046,609 total"
    assert len(s.materials) == 71
    for m in s.materials:
        assert (s.texture_dir / m.base_color).exists() and (s.texture_dir / m.normal).exists()
        assert m.specular is None or (s.texture_dir / m.specular).exists()
    # rebasing leaves only triangles that genuinely span many tiles (largest here: 39), well inside the
    # float32 budget (|uv| < ~400 keeps interpolation error under 0.1 texel at 2048)
    assert np.abs(s.uv[s.tri]).max(axis=(1, 2)).max() < 100


@cuda
def test_v_axis_points_up_in_texture_space():
    """Scene uv has v up; image rows run top-down. v near 1 must read the top row of the image."""
    from sixteen.render.sources import AtlasSource as A

    st, sets = _fake_store()
    grad = torch.linspace(0, 1, 32, device="cuda")[:, None, None].expand(32, 32, 3).contiguous()  # row 0 = 0, last row = 1
    sets[0].albedo = [l.to("cuda") for l in pyramid(grad)]
    atlas = TextureAtlas(st)
    uv = torch.tensor([[0.5, 0.97], [0.5, 0.03]], device="cuda")
    tex, _ = A(atlas, "trilinear").sample(uv, torch.zeros(2, 4, device="cuda"), torch.zeros(2, dtype=torch.long, device="cuda"),
                                         torch.arange(2, device="cuda"))
    assert tex[0, 0].item() < 0.1 and tex[1, 0].item() > 0.9


@cuda
def test_taa_reprojection_follows_camera_motion():
    """History from camera A, warped to camera B, must match B's own image (uv is a smooth surface signal)."""
    r = Renderer(quad_scene())
    W = H = 256
    cam_a = Camera(eye=(0, 0, 2), target=(0, 0, 0), fovy=90, near=0.1, far=10)
    cam_b = Camera(eye=(0.15, 0.05, 2), target=(0.0, 0, 0), fovy=90, near=0.1, far=10)  # moves ~10 px
    gb_a = rasterize_gbuffer(r.ctx, r.sg, cam_a, W, H)
    gb_b = rasterize_gbuffer(r.ctx, r.sg, cam_b, W, H)
    img = lambda gb: torch.cat([gb["uv"], gb["uv"][..., :1]], -1)
    taa = TAA(alpha=0.0)  # output = clamped, reprojected history only
    taa.resolve(img(gb_a), gb_a, cam_a, W, H)
    out = taa.resolve(img(gb_b), gb_b, cam_b, W, H)
    # away from silhouettes and the uv seam between the quads (the bilinear history fetch blends across those
    # discontinuities; real shading relies on the neighbourhood clamp there)
    import torch.nn.functional as F

    mat = gb_b["mat"]
    seam = torch.zeros_like(gb_b["valid"])
    seam[:, 1:] |= mat[:, 1:] != mat[:, :-1]
    seam[:, :-1] |= mat[:, 1:] != mat[:, :-1]
    near = lambda m: F.max_pool2d(m.float()[None, None], 7, 1, 3)[0, 0] > 0
    inner = gb_b["valid"] & gb_a["valid"] & ~near(~gb_b["valid"]) & ~near(~gb_a["valid"]) & ~near(seam)
    assert inner.sum() > 5000
    err = (out - img(gb_b)).abs().amax(-1)[inner]
    naive = (img(gb_a) - img(gb_b)).abs().amax(-1)[inner]
    assert err.max() < 1e-4, err.max()  # reprojection is exact for a smooth signal (measured ~2e-7)
    assert naive.mean() > 100 * err.mean()  # and the warp matters: without it the error is ~1e-2


@cuda
def test_taa_with_jitter_converges_to_the_pixel_footprint_mean():
    """Static camera, jittered frames of a 4 px-period signal: TAA must approach the mean over jitter.

    Regression test: reprojecting the jittered hit position (instead of the pixel centre) resamples
    history at a fractional offset each frame and blurs it; that variant measured 0.21 here, a single
    frame 0.24, the pixel-centre version 0.011.
    """
    r = Renderer(quad_scene())
    W = H = 256
    sig = lambda gb: (0.5 + 0.5 * torch.sin(2 * math.pi * 16 * gb["uv"][..., :1])).expand(-1, -1, 3).contiguous()
    frames = [rasterize_gbuffer(r.ctx, r.sg, CAM, W, H, jitter_at(f)) for f in range(64)]
    mean = sum(sig(g) for g in frames) / 64
    valid = frames[0]["valid"].clone()
    valid[:, :40] = valid[:, -40:] = False
    valid[:40] = valid[-40:] = False
    taa = TAA(alpha=0.1)
    for g in frames:
        out = taa.resolve(sig(g), g, CAM, W, H)
    err = (out - mean).abs().amax(-1)[valid].mean().item()
    single = (sig(frames[-1]) - mean).abs().amax(-1)[valid].mean().item()
    assert single > 0.15  # the signal really is noisy per frame
    assert err < 0.03, err


def test_dx_normal_decode_flips_green_and_rebuilds_z():
    """Bistro normals are DirectX BC5 (x, y; green points down). sixteen.data wants OpenGL (green up), n * 0.5 + 0.5."""
    from sixteen.render.textures import decode_normal_dx_bc5

    flat = decode_normal_dx_bc5(np.full((2, 2, 3), 128, np.uint8))
    assert torch.allclose(flat.reshape(-1, 3)[0], torch.tensor([0.502, 0.502, 1.0]), atol=0.01)  # +z
    px = np.full((1, 1, 3), 128, np.uint8)
    px[..., 0], px[..., 1] = 200, 200  # DirectX: +x, and +y meaning "down"
    n = decode_normal_dx_bc5(px)[0, 0] * 2 - 1
    assert n[0] > 0.4  # x unchanged
    assert n[1] < -0.4  # green flipped: DirectX +y (down) becomes OpenGL -y
    assert n[2] > 0 and abs(n.norm().item() - 1) < 1e-4
