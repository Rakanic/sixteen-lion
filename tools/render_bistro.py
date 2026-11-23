"""Render Bistro (or any scene in sixteen.render) headless to PNG with a chosen texture source.

    pixi run python tools/render_bistro.py --scene BistroInterior --camera entrance --out out/render/interior.png
    pixi run python tools/render_bistro.py --source original-stf --frames 64 ...   # accumulate 64 jittered frames

Our compressed textures (fits from `python -m sixteen.sweep configs/bistro-v2.yaml`; fitted materials come from
the compressor, the rest from the originals; use --frames 16 or more, the taps are stochastic):
    pixi run python tools/render_bistro.py --source neural-v2 --fit out/bistro-v2 --frames 32 --ambient 0.3
    --source neural-ref   PyTorch decode of the fits    --source neural-v1  CUDA decoder v1 (bilinear fits, --fit out/bistro-pilot)
    --source neural-v2    CUDA decoder v2, FP4-direct   --source neural-v2-fp16   v2 with the FP16-unpack layer 1

Needs LD_LIBRARY_PATH=$CONDA_PREFIX/lib the first time a scene is imported from FBX (pyassimp).
"""

import argparse
import pathlib

import imageio.v3 as iio
import torch

from sixteen.render.gbuffer import Camera
from sixteen.render.neural_ref import NEURAL_KINDS, neural_source
from sixteen.render.renderer import Renderer, jitter_at
from sixteen.render.scene import load_scene
from sixteen.render.shade import to_display
from sixteen.render.sources import AtlasSource
from sixteen.render.textures import TextureAtlas

CAMERAS = {
    "BistroInterior": {
        "entrance": Camera(eye=(6.5, 1.7, 4.5), target=(4.5, 1.6, -8.0), fovy=70, far=100),
        "bar": Camera(eye=(1.0, 1.7, -2.0), target=(9.0, 1.5, -9.0), fovy=70, far=100),
    },
    "BistroExterior": {
        "street": Camera(eye=(0.0, 2.0, 20.0), target=(0.0, 3.0, -20.0), fovy=65, far=1000),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="BistroInterior")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--source", default="original", choices=["original", "original-stf", *NEURAL_KINDS])
    ap.add_argument("--fit", default="out/bistro-v2", help="fit directory for the neural sources")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--ambient", type=float, default=0.03)
    ap.add_argument("--out", default="out/render/frame.png")
    args = ap.parse_args()
    W, H = map(int, args.size.split("x"))

    scene = load_scene(args.scene)
    cam = CAMERAS[args.scene][args.camera or next(iter(CAMERAS[args.scene]))]
    r = Renderer(scene)
    if args.source in NEURAL_KINDS:
        src, fits = neural_source(r, scene, args.fit, args.source)
        print(f"{len(fits)} fitted materials from {args.fit}: {', '.join(f['stem'] for f in fits)}")
    else:
        atlas = TextureAtlas(r.store)
        print(f"atlas {atlas.nbytes() / 2**20:.0f} MiB for {len(scene.materials)} materials")
        src = AtlasSource(atlas, "trilinear" if args.source == "original" else "stf")
    r.render(src, cam, W, H, ambient=args.ambient)  # warm-up
    torch.cuda.synchronize()
    if args.frames == 1:
        out = r.render(src, cam, W, H, ambient=args.ambient)
        rgb, ms = out["rgb"], out["ms"]
    else:
        rgb, ms = r.accumulate(src, cam, W, H, args.frames, ambient=args.ambient), None
    img = (to_display(rgb).flip(0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, img)
    print(f"wrote {path}  ms {ms}")


if __name__ == "__main__":
    main()
