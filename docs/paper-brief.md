# Paper brief: Sixteen Wide

Instructions for the agent writing the paper in `paper/` (Overleaf, ACM `acmart`,
`acmtog` format). Read this file, then `results/README.md`. Every number in the
paper must come from a file under `results/`; if a number is not there, write
`\todo{...}` instead of estimating it.

## 1. One-sentence pitch

Neural texture compression on block-scaled matrix hardware: we store each
material's latents in NVFP4 and decode them on Blackwell Tensor Cores
**without unpacking**, because **the four neighbouring latent texels map onto
the four scale blocks of one FP4 MMA**.

## 2. Contributions (use this list in the introduction, in this order)

| # | contribution | status | evidence |
|---|---|---|---|
| C1 | **FP4-direct decoding.** A latent layout and decoder in which each 16-channel latent texel is one NVFP4 block, so the four neighbouring latent texels map onto the four scale blocks (K = 64) of one block-scaled FP4 MMA (`m16n8k64`); stored codes reach the Tensor Core unpacked, and filtering moves to the decoded colors (output-space stochastic filtering). | instruction verified on the GPU; v1 decoder (FP16 MMA) works; 4-neighbour model and FP4 kernel **in progress** | `results/decoder/`, `cuda/` |
| C2 | **What matters for block-scaled neural textures**, a controlled study: (a) all block-scaled formats share one quality-vs-memory frontier, MXFP6 slightly best; (b) a GGX shading loss cuts FLIP by 23 % at every precision, so it is general, not quantization-specific; (c) after QAT a third of latents sit on a rounding boundary; (d) choosing block scales by error during training hurts. | done | `results/milestone/`, `shading-pilot/`, `shading-control/`, `profile/`, `scale-select/` |
| C3 | **Rendering-aware rounding.** After training, choose up/down for near-boundary latents by shaded-image error; same storage, decoder frozen; FLIP −3 to −10 %, beating texel-error rounding on every material. | done on 3 materials; 6-material × 2-seed robustness run with unseen lighting **in progress** | `results/rounding-pilot/`, later `results/rounding/` |

Position C3 honestly: it is AdaRound (Nagel et al. 2020) with a rendering loss;
claim the rendering objective and the measurement, not the optimizer.

## 3. Milestone report (due Sun 2026-11-22, 23:59 PT)

About one page of content in the same template, plus links to the webpage,
slides and video (placeholders: `\todo{URL}`). Four parts, as the course asks:

1. **What we have done.** PyTorch compressor with QAT for 9 latent formats
   (fp32, fp16, int8, int4, MXINT8, MXFP8, MXFP6, MXFP4, NVFP4), GGX shading
   loss, FLIP evaluation on shaded images; 3 Poly Haven materials (6 in the
   running robustness experiment); first CUDA
   Tensor Core decoder; FP4 block-scaled MMA verified on an RTX PRO 6000
   Blackwell.
2. **Preliminary results** (use exactly these):
   - Figure: `results/milestone/rd_flip.png`, FLIP vs bits per texel, 9 formats, 3 materials.
   - Figure: `results/milestone/crops_castle_brick_07_m.jpg`, reference vs 4/6/8-bit crops.
   - Table: shading-loss gain by precision (`results/README.md`, step 3: 23 % at fp32 … 15 % at MXFP4).
   - Table: rendering-aware rounding, 3 materials (`results/README.md`, step 6).
   - One sentence: decoder v1 matches PyTorch (max error 1.3e-3) and decodes a 1080p frame in about 1 ms on a busy GPU (`results/decoder/`).
3. **Progress vs plan.** Compressor ahead of plan; decoder started; Bistro
   integration not started. The contribution changed after measurements
   (state it plainly): the shading loss turned out to be general, so the
   main contribution moved from quantization to FP4-direct decoding.
4. **Updated plan.** 11/23–11/25 4-neighbour model + FP4 kernel on one
   material; 11/26–11/29 Bistro; 11/30–12/3 timing (FP4-direct vs FP16-unpack,
   per format), rounding robustness, BC1/BC5 baseline at equal memory;
   12/1–12/6 paper and video.

## 4. Final paper outline (4–6 pages)

| § | content | figures / tables |
|---|---|---|
| 1 Introduction | NTC meets block-scaled Tensor Cores; contributions C1–C3 | teaser: Bistro with NVFP4 materials (`\todo`) |
| 2 Related work | NTC (Vaidyanathan et al. 2023); neural appearance models (Zeltner et al. 2024); block-compressed neural features (Weinreich et al. 2024); Fujieda & Harada 2024 (verify title/venue); thread-efficient NTC decoding (2026); OCP MX spec, NVFP4; QAT, AdaRound, QAT oscillations (Nagel et al. 2022); appearance-driven simplification (Hasselgren et al. 2021); stochastic texture filtering (Pharr et al. 2024) | — |
| 3 Background | MX / NVFP4 block formats; Blackwell block-scaled MMA | diagram: one texel = one NVFP4 block; 4 texels = one MMA K tile |
| 4 Method | 4.1 compressor (grids, MLP, QAT, shading loss); 4.2 **FP4-direct decode** (the mapping, output-space STF, why latent-space STF is biased: 20.6 dB, `results/decoder/`); 4.3 rendering-aware rounding | decoder dataflow diagram |
| 5 Results | 5.1 formats (`rd_flip.png`); 5.2 shading loss + control; 5.3 latent profile (`profile/latent_on_e2m1_grid.png`); 5.4 rounding (table + crops, unseen lighting); 5.5 decode time per format, FP4-direct vs FP16-unpack, Bistro frame time (`\todo`); 5.6 BC1/BC5 at equal memory (`\todo`) | RD plot, tables, crops, timing bars |
| 6 Limitations and lessons | scale selection during training fails; QAT oscillation; flat-plane shading model; one GPU | — |
| 7 Conclusion | | |

Course rules the text must satisfy: say clearly what we started from (PyTorch,
NVIDIA FLIP evaluator, Poly Haven CC0 textures, Bistro scene) versus what we
built (compressor, quantizers, shading loss, decoder kernels, experiments);
per-member contributions go on the webpage.

## 5. Figure manifest

Copy figures into `paper/figures/` under the same names; regenerate from the repo, never edit pixels.

| file | use | caption draft |
|---|---|---|
| `results/milestone/rd_flip.png` | Fig. RD | FLIP after shading vs bits per texel (9 channels, incl. MLP) for nine latent formats at three grid sizes. All block-scaled formats share one frontier. |
| `results/milestone/crops_<material>_m.jpg` | Fig. crops | Reference and decoded crops at the mid grid; rows give bits per texel and FLIP. |
| `results/shading-control/loss_flip.png`, `results/shading-pilot/loss_flip.png` | Fig./Tab. shading loss | FLIP with texel loss vs texel + shading loss at equal memory. |
| `results/profile/latent_on_e2m1_grid.png`, `latent_hist.png` | Fig. profile | Latent values relative to their block scale; after QAT values pile up on rounding boundaries. |
| `results/rounding-pilot/results.csv`, `castle_shaded_grazing_*.png` | Tab./Fig. rounding | Round-to-nearest vs texel-aware vs rendering-aware rounding, same storage. |
| `results/decoder/v1_castle_brick_07_nvfp4.json` | text, Tab. timing | Decoder accuracy and 1080p time (v1; re-measure on an idle GPU for the final paper). |

## 6. Writing rules

- Lead with the mapping sentence (C1); the quantization study supports it.
- Numbers with units and the file they come from; FLIP to 3 decimals, PSNR to 0.1 dB, percentages to 0.1 %.
- Do not claim an FP4 speedup until `results/decoder/` has v2 timings; until then write `\todo{FP4-direct vs FP16-unpack timing}`.
- Report negative results (scale selection, latent-space STF bias) briefly; they are part of C2 and C1's motivation.
- Short sentences, active voice, no hype words ("novel", "significantly") without a number next to them.
- When `results/README.md` or the project log changes, update the matching section and the tables; keep this brief as the source of the outline.
