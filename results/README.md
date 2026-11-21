# Results

Finished experiments only; regenerate with `pixi run sweep <config>` then
`pixi run figures out/<name>`. All runs: 2k Poly Haven materials (9 channels),
grids (downsample, channels), 20k steps (70 % fp32, 30 % QAT), seed 0.
Quality is measured against the uncompressed textures shaded the same way
(GGX, 3 fixed light/view setups); FLIP is the mean over those setups.

| dir | config | question | answer |
|---|---|---|---|
| `milestone/` | `configs/milestone.yaml` (81 runs) | quality vs bits per texel for 9 latent formats, 3 grid sizes | all block-scaled formats lie on one frontier; MXFP6 slightly best; per-channel INT4 worst; fp32/fp16 ~4x the memory for equal FLIP |
| `shading-pilot/` | `configs/shading-pilot.yaml` (27 runs) | does adding a GGX shading loss help at 4/8 bits? | yes: FLIP -10 to -33 %, shaded PSNR +1.3 to +3.1 dB at equal memory; bits move from AO/rough/metal to normals |
| `shading-control/` | `configs/shading-control.yaml` (18 runs) | is the gain larger at low bits? | no: FLIP gain 23 % at fp32/fp16/MXFP8/MXFP6, 17 % NVFP4, 15 % MXFP4 — a general loss improvement, not quantization-specific |
| `profile/` | `tools/profile_latents.py` (castle_brick_07, MXFP4, shading loss) | where do latent values fall? | std ~0.7; 9 % round to 0; 27 % of MX blocks clip their max; sensitivity spans ~3 decades, top 10 % of values carry 54 %, uncorrelated with magnitude (r = 0.04); after QAT 31-33 % of values sit within 1 % of a rounding boundary (4.4 % before QAT) |

Shading-control FLIP gain (texel loss -> texel + 4 x shading loss), mean of 3 materials:

| latents | bits/value | FLIP gain | shaded PSNR gain |
|---|---|---|---|
| fp32 | 32 | 23.1 % | +2.33 dB |
| fp16 | 16 | 23.1 % | +2.33 dB |
| mxfp8 | 8.25 | 22.7 % | +2.30 dB |
| mxfp6 | 6.25 | 22.8 % | +2.30 dB |
| nvfp4 | 4.5 | 17.2 % | +1.96 dB |
| mxfp4 | 4.25 | 15.4 % | +1.82 dB |

The 4- and 8-bit rows combine `shading-pilot/` (shade4 arm) and `shading-control/`.
Crop grids are JPEG (quality 92); the per-run PNGs stay in `out/`.
