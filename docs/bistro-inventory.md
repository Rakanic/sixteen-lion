# Bistro inventory

Measured 2026-11-21 from the downloaded assets with `tools/bistro_inventory.py` and
`python/sixteen/render/scene.py`; nothing here is from memory. Source: NVIDIA ORCA,
`Bistro_v5_2.zip` (894 MB; `https://developer.nvidia.com/bistro` redirects to the download),
unpacked to `data/bistro-orca/Bistro_v5_2/` (git-ignored). Licence: CC BY 4.0
(`LICENSE.txt`); the README gives the citation to use. Amazon Lumberyard Bistro, ORCA, 2017.

The McGuire-archive copy of Bistro (OBJ) has **no textures** in its zips and uses an older
texture set (`*_diff.png`, `*_spec.png`, `*_ddna.png`); it is not used.

## Geometry

| scene | triangles | materials | extent (m, X × Y × Z) | uv range | note |
|---|---|---|---|---|---|
| BistroInterior | 1,046,609 | 71 | 15.6 × 7.3 × 19.4 | −32 … 46 | matches the README count |
| BistroExterior | 2,832,120 | 132 | 173.8 × 51.1 × 184.4 | ±6e4 | matches the README count |
| BistroInterior_Wine | (README: 1,293,691) | not loaded | — | — | not used |

Units are metres, Y up. Tangents and bitangents come from assimp; the FBX embeds a camera path
and lights, which the loader does not read.

**UV precision.** About 27 % of triangles have uvs outside [0, 1] (tiled surfaces), and the
exterior reaches ±6e4, where float32 interpolation leaves tens of texels of error. The loader
re-bases each triangle with |uv| > 8 by an integer (tiling is preserved). Afterwards the
interior's largest remaining per-triangle uv value is 39. In the exterior 136 triangles
still exceed 400: all are `MASTER_Grain_Metal`, at most 1.6 cm² each, with uvs implying
about 4e7 tiles per m² (effectively noise). A mip-mapped lookup averages them out, so they
are left alone.

## Textures

All 203 texture sets are in `Textures/`. The README states the packing, which matches our
9-channel layout in `sixteen/data.py` except the normal convention:

| map | content | formats here | resolutions |
|---|---|---|---|
| `_BaseColor` | RGB base colour, **alpha = opacity** | BC1 (142), BC3 (61) | 2048² (157), 16² (46) |
| `_Normal` | **DirectX** tangent-space normal, x y only (z rebuilt) | BC5 (203) | 2048² (170), 1² (33) |
| `_Specular` | R occlusion, G roughness, B metalness | BC3 (198), BC1 (3) | **16² (198)**, 2048² (3) |
| `_Emissive` | emissive colour | BC1 (15) | 64² (8), 2048² (7) |

Plus 11 TGA files (street lights, string lights, statue glass, spotlight glass) not matched to
a set. Total texture data with mips: 1,392 MiB.

Consequences:

- **The originals are already block-compressed.** There is no uncompressed Bistro. The
  "original" in every comparison is the decoded BC1/BC3/BC5/BC3 data, and our neural textures
  are trained on that. The paper must say so.
- **The BC baseline already exists in the scene.** Bistro ships BaseColor BC1 (4 bits) or BC3
  (8 bits) + Normal BC5 (8 bits) + a mostly 16² Specular map. Per full-resolution texel that
  is 12 bits where the base colour is BC1 (142 of 203 sets) and 16 where it is BC3 (61 sets;
  only 4 interior and 12 exterior materials actually use their alpha). The ARM map is
  negligible. This is not the log's "BC1+BC5+BC1 = 16", which assumes a full-size BC1 ARM map.
  Compare at equal bytes per material, not equal bits per texel.
- **The ARM map is nearly constant.** 198 of 201 specular maps are 16², so for almost all
  materials AO/roughness/metalness carry little detail and the memory is in albedo and normals
  (normals alone are 907 MiB of the 1,392). Training them as full-resolution neural channels
  spends bits on nothing; decide between upsampling them (simple) and storing per-material
  constants.
- **The "occlusion" channel is unused.** The README says Specular R is occlusion, but R is 0 in all
  201 specular maps (checked 2026-11-22). Read as ambient occlusion it zeroes the ambient term
  everywhere. The renderer and the compressor targets treat an all-zero R as 1 (no occlusion).
  Renders and the stage-1 measurements made before this fix had no ambient light at all, and the
  `--ambient` option did nothing.
- **The normal map is DirectX** (README, and measured). `textures.py` flips green and rebuilds z
  so it matches `sixteen/data.py` (OpenGL). `tools/check_normal_convention.py` confirms the
  flip: against the albedo gradient, the normal's y agrees in sign with x in 42 of 48
  textures after flipping, 6 without.
- Three maps of a set usually have different resolutions (198 of 201 complete sets), so the
  renderer keeps each map at its native size and the compressor will have to resample to one.

## Material coverage (for the compress-everything plan)

| | Interior | Exterior |
|---|---|---|
| materials | 71 | 132 |
| detailed (albedo or normal ≥ 1024²) | **64** (76 % of triangles) | **121** (99 % of triangles) |
| albedo 2048² / normal 2048² | 54 / 62 | 107 / 112 |
| with real alpha (opacity < 1 somewhere) | 4 (3.8 % of triangles) | 12 (10.1 % of triangles) |
| with an emissive map | 4 | 10 |
| without a specular map | 1 | 1 |

Training cost at the milestone's mid grid: median 326 s per fit (27 runs, GPUs shared with
other jobs, `results/milestone/results.csv`). That is about 5.8 GPU-hours for the interior's 64
detailed materials and about 11 for the exterior's 121, before any change to the model.

## What the 9-channel pipeline cannot represent

- **Opacity.** 4 interior and 12 exterior materials need alpha for cut-outs (leaves, grates).
  The renderer currently draws them opaque. The decoder's output is padded to 16 channels, so a
  10th channel is free in the kernel, but the compressor and data loader assume 9.
- **Emissive.** Lamps and string lights use emissive maps. They render as base colour only.
- **Glass.** Renders opaque grey (no transparency).

Until these are decided, report coverage: what fraction of pixels and triangles use materials
that are original-only.
