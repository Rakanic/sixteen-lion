# Answers for the paper agent (updated 2026-11-22, with idle-GPU timings)

Latest state. Supersedes the earlier version of this file (commit 2d499e5).

## To fix in `docs/paper-brief.md` (please edit, commit and push)

1. C3 row: replace "FLIP −3 to −10 %" with "mean FLIP −7.0 to −8.1 % (texel-error rounding −2.6 to −3.1 %)".
2. Writing rules: delete "Do not claim an FP4 speedup until `results/decoder/` has v2 timings ..."; v2 timings exist (FP4-direct 1.18–1.21x over FP16-unpack, `results/decoder/v2_profile.md`).

## For the milestone tonight

| topic | what to write | source |
|---|---|---|
| rendering-aware rounding | **final**: beats round-to-nearest and texel-aware rounding in **48/48** models (6 materials x 2 seeds x MXFP4/NVFP4 x texel/shading training), under test and unseen lighting; mean FLIP −7.0 to −8.1 % (texel-aware −2.6 to −3.1 %); ≈ 20–24 % more latent memory would be needed for the same gain (estimate); seeds agree (median gap 0.35 points) | `results/rounding/summary.md`, `results/README.md` step 6 |
| 4-neighbour model | quality cost at equal memory, one material: FLIP 0.0567 vs 0.0554 bilinear (+2.3 %); with layer-1 weights also NVFP4 (needed by the FP4 MMA): 0.0620 (+12 %) | `results/concat4-pilot/results.json` |
| decoder | v1 (bilinear model, FP16 MMA) matches PyTorch, ~1 ms per 1080p frame on a busy GPU. **v2 implemented**: the 4-neighbour model with layer 1 as FP4 block-scaled MMA, exact vs PyTorch (max error 2e-4); an FP16-unpack variant of the same kernel matches too | `results/decoder/`, `cuda/ntc_decode_v2.cu` |
| speed (idle GPU, 1080p frame, one lookup per pixel) | **citable**: FP4-direct 0.227 ms per frame. **FP4-direct is 1.18–1.21x faster than our equally fused FP16-unpack kernel** (each at its best launch setting); this is the FP4-specific result. Against the best PyTorch version of the same work (compiled, fp16 Tensor Cores, 4.22 ms) it is 18.6x faster, but say that this gap is kernel fusion (standard practice), not FP4; do not headline the eager-PyTorch 85x. SASS confirms the FP4 path runs `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`. Explain the modest FP4 gain with the profile (arithmetic bottleneck removed, fetch latency exposed) | `results/decoder/v2_profile.md`, `v2_best_vs_best.json`, `v2_baselines.json` |
| figures | `rd_flip.png`, `crops_castle_brick_07_m.jpg`; **`results/decoder/speedup.pdf`** (all decoders, keep the whole table; caption in the brief's figure manifest); optionally `results/rounding/rounding_flip.png` | |

`results/rounding-interim/` was removed; use `results/rounding/`.

## What changed in the story

- C3 (rendering-aware rounding) is now the most solid result: 48/48, holds
  under unseen lighting, and works on texel-trained models (drop-in step).
- C1 (FP4-direct decoding) works, is exact, and is **1.18–1.21x faster** than
  a fair FP16-unpack baseline (hardware FP4→FP16 conversion, same kernel).
  The gain is modest because decode becomes fetch-latency-bound once the
  arithmetic shrinks (Nsight Compute, `results/decoder/v2_profile.md`); say
  so. Fusion alone gives 18.6x over the best PyTorch version (compiled fp16);
  report it as context, not as a contribution.
- Requiring FP4 layer-1 weights costs quality (+12 % FLIP vs bilinear on one
  material). An alternative (MXFP4 latents with FP8 weights through the
  mixed-precision MMA variant) is not tested yet; do not mention it as a result.

## Mapping details for diagrams (verified on the GPU)

- `mma.m16n8k64 ... kind::mxf4nvf4.block_scale.scale_vec::4X ... ue4m3`:
  M = 16 pixels, N = 8 neurons per instruction, K = 64 = 4 latent texels x 16 channels.
- One grid per instruction (grid 0 and grid 1 each give one K = 64 tile).
- A scales: one UE4M3 per texel (16 values); 4 per pixel row. B (weights) are
  NVFP4 too, one UE4M3 per 16 input values.
- Each grid's MMA has its own FP32 accumulator (different per-tensor scales);
  they are scaled and summed.
- Positional encoding + fractional positions (12 values) use one FP16 MMA;
  layers 2–3 are FP16 MMA.
- Filtering is after decoding: each pixel decodes one randomly chosen
  full-resolution texel per frame.
- Texel order in a K tile: (x0,y0), (x1,y0), (x0,y1), (x1,y1).
