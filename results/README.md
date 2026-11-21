# Results

What we found, in the order we found it. Each step: the question, the
answer, and what it means for the project. All runs: 2k Poly Haven
materials (9 channels: albedo, normal, AO/roughness/metalness), two latent
grids + MLP decoder, 20k steps (70 % fp32, 30 % QAT), seed 0 unless noted.
**FLIP** is measured on GGX-shaded images (3 fixed light/view setups) against
the same shading of the uncompressed textures; lower is better.

## 1. Which latent format? → `milestone/`

**Question.** Quality vs memory for 9 formats (fp32 … MXFP4, NVFP4) at 3 grid sizes.
**Answer.** All block-scaled formats lie on one quality-vs-bits curve; MXFP6 is
slightly best, per-channel INT4 is clearly worst, fp32/fp16 need ~4x the
memory for the same FLIP. Mid grid at 4 bits: 5.3 bits/texel (BC1+BC5+BC1: 16).
**Meaning.** Block scaling is what makes 4 bits usable, but the bit width
itself is not where quality comes from. Format choice can be driven by
**decode speed on the hardware** (FP4 vs FP6 vs FP8 Tensor Core rates).

## 2. Does a shading loss help at low bits? → `shading-pilot/`

**Question.** Train on GGX-shaded error (random light/view) as well as texel error, equal memory.
**Answer.** Yes, on all 9 material × format pairs: FLIP −10 to −33 %, shaded
PSNR +1.3 to +3.1 dB. Bits move from AO/roughness to normals.
**Meaning.** Use it in every model from now on.

## 3. Is that gain specific to quantization? → `shading-control/`

**Question.** Same comparison without quantization (fp32, fp16) and at 6 bits.
**Answer.** No. FLIP gain is 23 % at fp32, fp16, MXFP8 and MXFP6, and smaller at 4 bits (15–17 %).
**Meaning.** The shading loss is a general training improvement (as in prior
appearance-driven work), so it is a **baseline, not our contribution**.

## 4. Where do the latent values fall? → `profile/`

**Question.** Distribution of one trained MXFP4 model's latents (castle_brick_07).
**Answer.**
- bell-shaped, std ≈ 0.7; channels have similar ranges
- 9 % round to 0; 27 % of 32-value blocks clip their largest value
- loss sensitivity spans ~3 orders of magnitude, top 10 % of values carry 54 %, and it is unrelated to magnitude (r = 0.04)
- after QAT, **31–33 % of values sit within 1 % of a rounding boundary** (4.4 % before QAT): their final rounding is close to random
- grouped codes would not compress further: measured entropy 3.8 bits per 4-bit value

**Meaning.** Rules that look only at values (max-abs, MSE) cannot see what
matters for the image; the near-random rounding of a third of the values is
the clearest thing to fix.

## 5. Choose block scales by rendered error? → `scale-select/`

**Question.** Per-block scale by the spec rule vs MSE-optimal vs sensitivity-weighted, re-chosen during QAT.
**Answer.** No. Both alternatives are worse than the spec rule in all 12 cases (FLIP +0.7 to +4.8 %).
**Meaning.** Negative result. Changing scales during training destabilizes
the decoder; keep the spec rule.

## 6. Choose rounding directions by rendered error? → `rounding-pilot/`

**Question.** After training, decoder frozen, same storage: pick up/down for
near-boundary values by texel error or by rendered error (vs round-to-nearest,
and vs the same number of extra QAT steps).

| FLIP | castle_brick_07 | metal_plate | rough_wood |
|---|---|---|---|
| round to nearest | 0.0592 | 0.0438 | 0.0559 |
| extra QAT (control) | 0.0608 | 0.0445 | 0.0570 |
| texel-aware rounding | 0.0570 (−3.7 %) | 0.0434 (−0.9 %) | 0.0553 (−1.1 %) |
| **rendering-aware rounding** | **0.0532 (−10.1 %)** | **0.0424 (−3.2 %)** | **0.0520 (−7.0 %)** |

**Meaning.** A free, post-training improvement that beats texel-aware
rounding on every material. It is AdaRound with a rendering loss, so it is a
**supporting technique, not the headline**. Robustness check running
(`configs/rounding-base.yaml`: 6 materials, 2 seeds, both 4-bit formats,
unseen lighting); results will land in `rounding/`.

## 7. CUDA decoder v1 → `decoder/`

**Question.** Can a hand-written Tensor Core decoder reproduce the PyTorch model, and how fast is it?
**Answer.** Yes: NVFP4 latents, FP16 MMA, max error 1.3e-3 vs PyTorch on the same
taps; 0.83–1.16 ms per 1080p frame (one lookup per pixel, measured while both GPUs
were 95 % busy), 36–45x faster than PyTorch. The GPU runs block-scaled FP4 MMA
(`m16n8k64 kind::mxf4`, target `sm_120a`, `cuda/probe/`).
Picking one random latent tap (latent-space STF) is biased: averaged over 64 frames
it stays 20.6 dB from the bilinear decode, because the MLP is nonlinear.
**Meaning.** Do not blend latents before the MLP. Feed the 4 neighbouring latent
texels as stored (4 x 16 values = the K = 64 operand and four scale blocks of one
FP4 MMA) and filter the decoded colors instead. This is the paper's main
contribution; model variant `filter: concat4`, kernel v2 in progress.

## Where this leaves the project

1. Quantization tricks give modest gains (3–10 % FLIP); none is strong enough to be the paper.
2. Format choice barely changes quality per bit, so the open question is **speed**: the paper's main contribution moves to a real-time decoder that runs the 4-bit latents directly on Blackwell FP4 Tensor Cores (paper plan: `.claude/project-log.md`).
3. Steps 1–6 become the paper's "what matters" study, with rendering-aware rounding as one technique.

Regenerate a sweep with `pixi run sweep <config>` then `pixi run figures out/<name>`.
Crop grids are JPEG (quality 92); per-run PNGs and checkpoints stay in `out/`.
