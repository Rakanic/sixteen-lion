# Answers for the paper agent (updated 2026-11-22, after eaec7e1)

## Latest: answers to your 6 questions (supersede anything below)

Include the data we have, even where it is not the best yet, and say we are
still improving it (two-part weights, layer-2 FP8, Bistro refit).

1. **Canonical decode times: the new kernel, one consistent run.**
   - File: `results/decoder/v2_baselines.json`; chart: `results/decoder/speedup.{png,pdf}`
     (`tools/bench_baselines.py --x2 ...`, `tools/plot_decoder_baselines.py`).
   - Same run: PyTorch eager/TF32/fp16, compiled fp32/fp16, FP16-unpack, FP4-direct, and FP4-direct with two-part weights.
   - Retire the 0.227 ms / 1.18–1.21x set (`v2_best_vs_best.json`, the timing tables in `v2_profile.md`); keep the
     Nsight Compute analysis in `v2_profile.md` (bottleneck explanation still holds).
   - `results/decoder/w1_modes.json` is the interleaved mode sweep (all register caps, prefetch on/off); use it
     for the claim that launch settings and prefetching do not change the ranking.
   - Speed chart caption: the PyTorch gap is fusion; the FP4-specific gain is FP4-direct vs FP16-unpack.
2. **Two-part weights (W ≈ q(W) + q(W − q(W)), two FP4 MMAs per grid): present it as the quality-preserving option, work in progress.**
   - Quality: `results/w1-weights/results.json` (castle_brick_07, 2 seeds, FLIP vs bilinear; use `flip_vs_bilinear_pct`).
   - Speed: `v2_baselines.json`.
   - Paper framing: a trade-off we are still improving.
     - One-part FP4 weights are fastest but cost quality.
     - Two-part weights recover the quality (on par with unquantized weights) at a smaller speedup.
     - Say we are working on closing the speed gap.
   - Do not call either one final.
   - MXFP8 weights (needs the FP4 x FP8 MMA): report as rejected.
     - Worse quality.
     - Its K = 32 tile shares one scale per 2 texels, which breaks the 4-texel mapping.
     - Measured 1.18x speedup (commit c7a2461), timing only.
   - Bistro: not refit by me. A teammate is running `configs/bistro-v2x2.yaml`.
     - Until that lands, keep 0.0153 vs 0.0106 (`results/renderer/neural-compare*/neural_compare.json`) as the one-part result.
     - Add that the two-part refit is in progress.
     - Per-material flat-plane change (one-part vs bilinear) is in the same files (`per_material[].flip_flat_plane`).
       - Detailed materials (plasters) are close to castle_brick.
       - Smooth ones (ceramic, painted wood) get the largest relative increase from a small base.
3. **Rounding on the FP4-ready (concat4 + NVFP4 W1) model: no results yet.** Write it as future work.
4. **Rendering-aware rounding optimizer (`tools/shading_rounding.py`):**
   - Parameterization:
     - Only latents within band 0.05 of a rounding midpoint are free: |t − 0.5| < 0.05, where t is the position between the two neighbouring E2M1 grid points. All other latents round to nearest.
     - Each free latent has h = rectified sigmoid (sigmoid(a) · 1.2 − 0.1, clamped to [0, 1]); the value is lo + h · (hi − lo).
     - Init h = 0.7 / 0.3 toward the nearest choice.
   - Optimizer: Adam, lr 1e-2, 3000 steps, batch 2^20 texels.
   - Loss:
     - texel loss + 4 x shading loss + λ · mean(1 − |2h − 1|^β).
     - β anneals from 20 to 2.
     - λ = 0 for the first 10 % of steps, then ramps to 0.05 over half the run.
   - Block scales are fixed by the format rule; the MLP is frozen.
   - Final choice: h > 0.5.
   - "more-qat" control: the same number of extra STE-QAT steps at QAT's final lr.
5. **Still open:** Bistro frame time on an idle GPU, and the BC1/BC5 equal-memory baseline. Mark both `\todo`.
6. **Corrections:**
   - Single-term NVFP4 W1 cost on castle_brick_07 is **+10.8 %** FLIP vs bilinear (2 seeds), not +12 % (1 seed).
   - The 4-neighbour layout alone is **+2.5 %** (2 seeds), not +2.3 %.
   - Do not compare flat-plane FLIP (≈0.01–0.06 per material) with room FLIP (≈0.01). They are different measurements; compare within one.
   - Replace 0.227 ms and 1.18–1.21x everywhere with the `v2_baselines.json` numbers.

---


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
