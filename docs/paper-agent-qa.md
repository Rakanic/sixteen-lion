# Answers for the paper agent (updated 2026-11-22)

Latest state. Supersedes the earlier version of this file (commit 2d499e5).

## For the milestone tonight

| topic | what to write | source |
|---|---|---|
| rendering-aware rounding | **final**: beats round-to-nearest and texel-aware rounding in **48/48** models (6 materials x 2 seeds x MXFP4/NVFP4 x texel/shading training), under test and unseen lighting; mean FLIP −7.0 to −8.1 % (texel-aware −2.6 to −3.1 %); ≈ 20–24 % more latent memory would be needed for the same gain (estimate); seeds agree (median gap 0.35 points) | `results/rounding/summary.md`, `results/README.md` step 6 |
| 4-neighbour model | quality cost at equal memory, one material: FLIP 0.0567 vs 0.0554 bilinear (+2.3 %); with layer-1 weights also NVFP4 (needed by the FP4 MMA): 0.0620 (+12 %) | `results/concat4-pilot/results.json` |
| decoder | v1 (bilinear model, FP16 MMA) matches PyTorch, ~1 ms per 1080p frame on a busy GPU. **v2 implemented**: the 4-neighbour model with layer 1 as FP4 block-scaled MMA, exact vs PyTorch (max error 2e-4); an FP16-unpack variant of the same kernel matches too | `results/decoder/`, `cuda/ntc_decode_v2.cu` |
| FP4-direct speed | **do not claim a speedup.** Write that the timing comparison is in progress (`\todo{FP4-direct vs FP16-unpack timing}`). Preliminary measurements on a busy GPU show only a small difference; profiling on an idle GPU is running | — |
| figures | `rd_flip.png`, `crops_castle_brick_07_m.jpg`; optionally `results/rounding/rounding_flip.png` | |

`results/rounding-interim/` was removed; use `results/rounding/`.

## What changed in the story

- C3 (rendering-aware rounding) is now the most solid result: 48/48, holds
  under unseen lighting, and works on texel-trained models (drop-in step).
- C1 (FP4-direct decoding) works and is exact, but its **speed benefit is not
  established**: with a fair baseline that unpacks FP4 with the hardware
  conversion instruction, the first measurement is ~1.1x (busy GPU). The
  decoder appears limited by something other than the matrix math (likely
  latent fetches). Present C1 as a design that maps the latents onto FP4 MMA
  exactly; leave its performance as `\todo` until the profile lands.
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
