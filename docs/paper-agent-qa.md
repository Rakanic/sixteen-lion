# Answers for the paper agent (state after a3d00ce, 2026-11-21)

## 1. What in the brief or results README is wrong or superseded

Nothing is wrong. Two updates:

- **Both FP4 block-scaled MMA variants are now verified on the RTX PRO 6000
  (sm_120a):** `kind::mxf4` (MXFP4, UE8M0 scale per 32 values, `scale_vec::2X`)
  and `kind::mxf4nvf4` (NVFP4, UE4M3 scale per 16 values, `scale_vec::4X`).
  Probes: `cuda/probe/mma_fp4_probe.cu`, `cuda/probe/mma_nvfp4_probe.cu`.
  The brief's C1 status line ("instruction verified") now covers the NVFP4
  variant the mapping uses.
- **Interim rounding robustness results exist** (`results/rounding-interim/`, see 3).

## 2. concat4 pilot

No numbers yet (training). Milestone: **"in progress"**. You may write that
the decoder design requires feeding the 4 neighbouring latent texels unblended
and that its quality at equal memory is being measured. Do not claim it
matches bilinear.

## 3. Rounding robustness

Not finished: 18 of 48 models (3 of 6 materials, both seeds, both formats,
both training losses). Interim summary: `results/rounding-interim/summary.md`.
Headline so far:

- rendering-aware rounding beats round-to-nearest **and** texel-aware rounding
  in **18/18** models, under the test lights and under unseen lighting (sky of
  64 lights, 4x brighter light, Blinn-Phong instead of GGX);
- mean FLIP change −4.0 to −7.5 % (texel-aware: −0.7 to −2.3 %), the same under
  unseen lighting (−4.1 to −7.1 %);
- the two training seeds differ by at most 0.6 percentage points.

The milestone **may** cite these, labelled "preliminary, 18 of 48 models".
The final numbers will be in `results/rounding/` (I will push when done).

## 4. Decoder v2 / FP4-direct timing

None yet. Keep `\todo{FP4-direct vs FP16-unpack timing}`. The only timing is
v1 (FP16 MMA, latents unpacked): 0.83–1.16 ms per 1080p frame with one lookup
per pixel, measured while both GPUs were 95 % busy (`results/decoder/`).

## 5. Figures and the mapping

Milestone figures: `rd_flip.png` and `crops_castle_brick_07_m.jpg` are enough.
If space allows, add `results/rounding-interim/rounding_flip.png` (labelled
preliminary). The profile figure belongs in the final paper.

Mapping, for your diagrams (planned v2 design; v1 does not do this yet):

- **MMA shape** `m16n8k64`, block-scaled, FP4 (E2M1) A and B, FP32 accumulate.
  **M = 16 pixels** (one per row), **N = 8 hidden neurons** per instruction
  (8 instructions cover the 64 hidden units), **K = 64 = 4 latent texels × 16 channels**.
- **One grid per MMA K-tile.** Grid 0 (1/4 resolution, 16 channels): its 4
  neighbouring texels form one K = 64 tile. Grid 1 (1/8 resolution, 16 channels):
  a second K = 64 tile.
- **Scales.** NVFP4: each 16-value block (= one texel) has one **UE4M3** scale;
  per pixel row the instruction takes **4** A-scales (the 4 texels; `scale_vec::4X`).
  NVFP4 also has one FP32 scale per tensor, which the instruction does not
  take. The two grids (and the weights) have different per-tensor scales, so
  each grid's MMA writes its own FP32 accumulator; the two are multiplied by
  their per-tensor scales and summed (one FMA per output value).
- **B operand (weights) must also be FP4** with UE4M3 scales per 16 K values:
  layer 1's latent weights are quantized to NVFP4 too. Draw them as FP4, and
  note it as a requirement of the instruction.
- **The rest of layer 1** (positional encoding + fractional positions, 12
  values) goes through one FP16 `m16n8k16` MMA into the same accumulators.
  Layers 2 and 3 are FP16 MMA, as in v1.
- **Filtering happens after decoding** (output-space stochastic filtering): each
  pixel decodes one randomly chosen full-resolution texel per frame and TAA
  averages colors. Do not draw a latent-space filter before the MLP; the v1
  measurement (20.6 dB bias) is why.
- **Texel order inside a K-tile:** (x0,y0), (x1,y0), (x0,y1), (x1,y1).
