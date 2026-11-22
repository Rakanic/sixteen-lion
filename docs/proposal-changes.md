# Changes since the proposal

The proposal (`docs/proposal/index.html`, copied from upstream commit `9d2a942`,
2026-11-12) was accepted with the feedback **"missing graphics novelty"**. This file
records what changed, why, and where the evidence is, so the milestone report, final
paper and webpage can say how we addressed it. Source of truth for status:
the "Now" section of `.claude/project-log.md`, `results/README.md` and
`docs/paper-brief.md` (C1–C3). Last updated 2026-11-21.

## How we addressed the feedback, in three sentences

The proposal's contribution was a quantization scheme (block-scaled formats plus QAT)
with a decoder mapped to Tensor Cores. Our own measurements showed the quantization part
is modest: all block-scaled formats lie on one quality-vs-memory frontier, and the
shading loss that looked like a quantization trick helps 23 % at fp32 too. So the
headline moved to the rendering-side decoder design: the stored 4-bit codes and scales
reach the Blackwell FP4 Tensor Core unpacked, and filtering moves to the decoded colours.

**Status of that claim:** C1 (FP4-direct decoding) is **in progress**. The block-scaled
FP4 MMA is verified on the GPU and decoder v1 (FP16 MMA) works, but the 4-neighbour
model and the FP4-direct kernel have no results yet. The novelty claim also depends on
the literature check (Thread-Efficient Decoding for NTC, NTC with Hypernetworks), which
is still open. Do not state C1 as achieved or as novel until both are done.

## Changes

| # | proposal said | now | why | evidence |
|---|---|---|---|---|
| 1 | Contribution: "a quantization scheme and quantization-aware training procedure" | Headline is **FP4-direct decoding** (C1). Quantization results are a "what matters" study (C2), plus rendering-aware rounding (C3) | Feedback "missing graphics novelty"; formats share one frontier, shading loss is general, scale selection hurts | `results/README.md` steps 1–5; `docs/paper-brief.md` §2 |
| 2 | Decoder "reconstruct[s] each block" with its scale, then FP16 or INT8 Tensor Core arithmetic | Stored FP4 codes and block scales go straight into block-scaled FP4 MMA (`m16n8k64`, NVFP4 `scale_vec::4X`). v1 still unpacks to FP16 | Unpacking is the step the hardware can skip | `cuda/probe/`, `docs/paper-agent-qa.md` §5; v2 kernel in progress |
| 3 | "texture hardware cannot filter block scaled values, so our decoder must filter them"; "filter across neighboring values" | **Output-space stochastic filtering**: decode one random texel per pixel per frame, TAA averages colours. Model variant `concat4` feeds the four neighbouring latent texels unblended | Measured: averaging one random latent tap over 64 frames stays 20.6 dB from the bilinear decode, because the MLP is nonlinear | `results/decoder/v1_castle_brick_07_nvfp4.json`; log entry 2026-11-21 |
| 4 | Headline metric: PSNR (texel and rendered frames) | **FLIP on shaded images** first, PSNR second | Texel PSNR reads as an ML-compression result | log entry 2026-11-20 |
| 5 | A Blackwell-class GPU is a stretch goal ("measurements on a second GPU, preferably one with reported native Tensor Core support for 4-bit and 6-bit formats") | Core to the project. Developed on an RTX PRO 6000 Blackwell (`sm_120a`) | The FP4 MMA is the contribution | `cuda/probe/mma_fp4_probe.cu`, `mma_nvfp4_probe.cu` |
| 6 | No shading-aware training | **GGX shading loss** added: FLIP −8.8 to −32.5 % (`shade4` runs). It is a baseline, **not claimed as a contribution**: the gain is 23 % at fp32 too | Quantization error matters only through the BRDF | `results/shading-pilot/`, `results/shading-control/` |
| 7 | Decoder weights quantized to INT8 (research question 3); one shared decoder per scene | Both demoted (P2). The FP4 MMA needs its B operand in FP4 too, so layer-1 weights become NVFP4 as a hardware requirement, not as a study | Priority went to the core mapping | log P2 list; `docs/paper-agent-qa.md` §5 |
| 8 | BC1 baseline at equal memory, with the learned values stored in BC1 "following Belcour and Benyoub" so hardware can filter them | The log now plans conventional BC1 + BC5 + BC1 textures (16 bits/texel, 9 channels). **This is a different baseline**, so it must be stated as a change, and the team has not confirmed which one the paper uses. Not done | — | log P1; `docs/renderer-spec.md` D5 |
| 9 | Week 1: Bistro loaded with original textures | **Not started.** Spec in `docs/renderer-spec.md`; FlightHelmet fallback gate at end of 11/27 | — | log; `docs/renderer-spec.md` §6 |
| 10 | Stretch: "a decoder that returns four neighboring texture values in one evaluation" | Still a P3 stretch in the log (2×2-quad decode); not started. **Not the same as `concat4`**: that decoder takes four latent texels as input and returns one texel | — | log P3; `python/sixteen/model.py` |

## Research questions

| proposal question | where it stands |
|---|---|
| Block-scaled vs uniform vs BC1 at equal memory | Block-scaled vs uniform INT4/INT8 measured (INT4 clearly worst). BC1 not done, and its meaning changed (row 8) |
| Minimum bit width before errors are visible after shading | Partial: FLIP and crops at 4, 6, 8 bits and fp16/fp32. No visibility threshold claimed |
| INT8 decoder: quality and decode time | Not started (see #7) |
| Best decoder update granularity in real time | Not addressed so far; the team needs to decide whether to drop it or tie it to the output-space STF / 2×2-quad work |

## Added after the proposal

- Rendering-aware rounding (C3): FLIP −3 to −10 % on 3 materials, −4.0 to −7.5 % over the
  first 18 of 48 robustness models (preliminary). It is AdaRound with a rendering loss.
- Negative result: choosing block scales by error during training is worse than the spec
  rule in 12 of 12 cases.
- Latent profile: after QAT, 31–33 % of latents sit within 1 % of a rounding boundary.
