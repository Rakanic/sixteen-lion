"""Inventory of the ORCA Amazon Lumberyard Bistro textures (data/bistro-orca/Bistro_v5_2).

Reads DDS headers only (no decoding): format, size, mip count, bytes. Groups
files into material sets by name (<stem>_BaseColor / _Normal / _Specular, and
_Emissive) and reports which sets are complete for the 9-channel layout of
sixteen.data.

    python tools/bistro_inventory.py [--root data/bistro-orca/Bistro_v5_2]
"""

import argparse
import collections
import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "bistro-orca" / "Bistro_v5_2"
DXGI = {71: "BC1_UNORM", 72: "BC1_SRGB", 74: "BC2_UNORM", 77: "BC3_UNORM", 78: "BC3_SRGB", 80: "BC4_UNORM",
        83: "BC5_UNORM", 84: "BC5_SNORM", 95: "BC6H_UF16", 98: "BC7_UNORM", 99: "BC7_SRGB", 28: "RGBA8", 87: "BGRA8"}
KINDS = ("BaseColor", "Normal", "Specular", "Emissive")


def dds_info(path):
    with open(path, "rb") as f:
        head = f.read(148)
    assert head[:4] == b"DDS ", path
    (h,) = struct.unpack_from("<I", head, 12)
    (w,) = struct.unpack_from("<I", head, 16)
    (mips,) = struct.unpack_from("<I", head, 28)
    fourcc = head[84:88]
    if fourcc == b"DX10":
        fmt = DXGI.get(struct.unpack_from("<I", head, 128)[0], f"DXGI_{struct.unpack_from('<I', head, 128)[0]}")
    else:
        fmt = fourcc.decode("ascii", "replace").strip("\0") or "uncompressed"
    return dict(w=w, h=h, mips=max(mips, 1), fmt=fmt, bytes=path.stat().st_size)


def kind_of(stem):
    for k in KINDS:
        if stem.endswith("_" + k):
            return k, stem[: -len(k) - 1]
    return None, stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    tex = pathlib.Path(args.root) / "Textures"
    sets = collections.defaultdict(dict)
    other = []
    for p in sorted(tex.iterdir()):
        if p.suffix.lower() == ".dds":
            k, stem = kind_of(p.stem)
            if k is None:
                other.append(p.name)
                continue
            sets[stem][k] = dds_info(p)
        else:
            other.append(p.name)

    n = len(sets)
    full = [s for s, d in sets.items() if all(k in d for k in KINDS[:3])]
    print(f"{n} texture sets; {len(full)} have BaseColor + Normal + Specular; {len(other)} other files")
    for k in KINDS:
        infos = [d[k] for d in sets.values() if k in d]
        fmts = collections.Counter(i["fmt"] for i in infos)
        res = collections.Counter(f"{i['w']}x{i['h']}" for i in infos)
        mb = sum(i["bytes"] for i in infos) / 2**20
        print(f"{k:10s} {len(infos):4d} files {mb:8.1f} MiB  formats {dict(fmts)}  sizes {dict(res.most_common(6))}")
    mism = [s for s in full if len({(sets[s][k]["w"], sets[s][k]["h"]) for k in KINDS[:3]}) > 1]
    print(f"sets whose three maps differ in resolution: {len(mism)}")
    print("not matched to a map kind:", other[:12], "..." if len(other) > 12 else "")
    total = sum(i["bytes"] for d in sets.values() for i in d.values()) / 2**20
    print(f"total texture bytes (with mips): {total:.1f} MiB")


if __name__ == "__main__":
    main()
