"""Download Poly Haven PBR materials (CC0) into data/polyhaven/<id>/.

Each material gets three maps: diffuse (sRGB albedo), nor_gl (OpenGL
tangent-space normal) and arm (AO, roughness, metalness in R, G, B).

    pixi run fetch                       # the default materials at 2k
    pixi run fetch -- --res 4k metal_plate
"""

import argparse
import hashlib
import json
import pathlib
import urllib.request

DEFAULT = ["metal_plate", "rough_wood", "castle_brick_07"]
MAPS = {"diffuse": "Diffuse", "normal": "nor_gl", "arm": "arm"}
UA = {"User-Agent": "sixteen-wide-fetch"}
ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "polyhaven"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r:
        return r.read()


def fetch(asset, res, fmt):
    files = json.loads(get(f"https://api.polyhaven.com/files/{asset}"))
    out = ROOT / asset
    out.mkdir(parents=True, exist_ok=True)
    for name, key in MAPS.items():
        entry = files[key][res][fmt]
        path = out / f"{name}_{res}.{fmt}"
        if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == entry["md5"]:
            print(f"  {path.relative_to(ROOT.parent.parent)} (cached)")
            continue
        blob = get(entry["url"])
        if hashlib.md5(blob).hexdigest() != entry["md5"]:
            raise RuntimeError(f"md5 mismatch for {entry['url']}")
        path.write_bytes(blob)
        print(f"  {path.relative_to(ROOT.parent.parent)} ({len(blob) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("assets", nargs="*", default=DEFAULT)
    ap.add_argument("--res", default="2k")
    ap.add_argument("--fmt", default="png")
    args = ap.parse_args()
    for a in args.assets:
        print(a)
        fetch(a, args.res, args.fmt)


if __name__ == "__main__":
    main()
