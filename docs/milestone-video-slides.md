# Milestone video and slides: plan

Due **Sun 2026-11-22, 23:59 PT** on Gradescope ("[Final Project] Milestone"), no late days. Aim to submit by 22:00.
The PDF report (ACM SIGGRAPH template, `paper/milestone.tex`) must carry **clickable links** to the webpage, the slides and the video.

## What the course asks for

| item | requirement | where it lives |
|---|---|---|
| Video | about 1 minute; **every member appears or speaks**; unlisted YouTube or a Berkeley-visible Drive link | link in report and webpage |
| Slides | 2 to 3 slides | link from the webpage |
| Webpage | persistent (GitHub Pages) page linking report, video, slides | link in report |
| Report | what is done, preliminary results, progress against the plan, updated plan | already drafted in `paper/milestone.tex`; needs the three URLs, a team read and an Overleaf push |

All three should tell the same story as the report's four parts. Say what is ours and what is a starting point (below).

## The one-sentence story

> Textures are most of a scene's memory. We store them as 4-bit block-scaled learned latents (about 5.7 bits per texel, against 12 to 16 for Bistro's block-compressed textures) and decode them on Blackwell's FP4 Tensor Cores; this milestone has the compressor, a working FP4-direct decoder and Bistro rendering through it.

## Numbers we can say (all from files in the repo)

| claim | number | source |
|---|---|---|
| Storage | 5.3 (MXFP4) / 5.7 (NVFP4) bits per texel, 9 channels | `results/milestone/`, log |
| Bistro's own textures | about 12 bits per full-res texel (BC1 base colour; 16 with BC3) | `docs/bistro-inventory.md` |
| Shading-aware training | FLIP after shading 15 % lower at MXFP4, 23 % lower at fp32 | log table |
| Rendering-aware rounding | 7 to 8 % lower FLIP than round-to-nearest, **48 of 48** models, worth about 20 to 24 % more latent memory | `results/rounding/summary.md` |
| Decoder v2 (FP4-direct) | **1.2x** faster than fused FP16-unpack (about 0.22 vs 0.27 ms per 1080p frame), idle GPU | `results/decoder/v2_best_vs_best.json` |
| Bistro, 10 materials through decoder v2 | FLIP **0.0064** on the neural pixels (91 % of the frame); 28.4 MiB against 60.0 MiB of BC for the same materials (2.1x; 2.8x with mips) | `results/renderer/neural-compare-v2/` |

**Do not say**: "beats BC at equal memory" (the reference is decoded BC and we compare bytes, not equal-memory quality); that FP4-direct is free
(concat4 with NVFP4 weights is about 38 % worse in FLIP than the bilinear model on Bistro, single seed; say "costs some quality, recovering it is next");
that the FP4-direct idea is novel (the literature check is still open); any frame-time figure from the renderer (GPUs were busy).

## Video: 60 seconds, four speakers

Format: one screen-share recording over Zoom with cameras on, one take per segment, cut together. Each speaker is on camera as a small inset for their segment, which meets "appears or speaks" twice over.
Speaker assignment below is a guess from the commit history and the log's lanes (compressor 2, decoder and renderer 2); **confirm names**.

| time | speaker | on screen | script (about 35 words) |
|---|---|---|---|
| 0:00 to 0:10 | A, framing | `wipe_original_vs_v2.mp4`, title lower-third with all four names | "Textures dominate a scene's memory. We store them as four-bit learned latents and decode them on Blackwell's FP4 Tensor Cores. On the left, Bistro with its original textures; on the right, ours." |
| 0:10 to 0:25 | B, compressor | `results/milestone/rd_flip.png`, one crop strip (`crops_castle_brick_07_m.jpg`) | "Our compressor trains latent grids and a small decoder with quantization-aware training, in nine formats. Block-scaled formats all land on one quality-versus-memory curve, so we added shading-aware training and rendering-aware rounding: seven to eight percent lower FLIP in all 48 models." |
| 0:25 to 0:40 | C, decoder | `results/decoder/speedup.png`, a v1 vs v2 clip (`v1_vs_v2.mp4`, 6 s) | "Decoder version two feeds the stored FP4 codes and block scales straight into the block-scaled FP4 matrix instruction, no unpacking. It runs 1.2 times faster than the FP16 path and replaces version one's biased latent-space tap with output-space stochastic filtering." |
| 0:40 to 0:52 | D, renderer / Bistro | Bistro wipe at the divider, then `crop.jpg` with the FLIP map | "We load Bistro, a million triangles, headless, and render ten fitted materials through the real decoder. FLIP after shading is 0.006 at half the memory of Bistro's block-compressed textures. Four-bit weights cost some quality; recovering it is next." |
| 0:52 to 1:00 | A, plan | plan strip (slide 3) | "Next: every material, recovering the FP4 quality gap, idle-GPU timing against a BC baseline, and a recorded fly-through. Thanks." |

Editing rules: no b-roll beyond what is listed; captions on every number; end card with the webpage URL and the starting-point credit line.
If it runs over 60 seconds, cut the crop strip in B first, then the v1 clip in C.

### Clips to cut beforehand (assets already exist)

- `results/renderer/neural-compare-v2/wipe_original_vs_v2.mp4`: use the first 10 s, and 5 s around the divider centre for D.
- `v1_vs_v2.mp4`: 6 s on the floor, where v1's speckle is visible.
- Stills: `crop.jpg`, `flip.jpg`, `wipe_poster.jpg`; charts `results/milestone/rd_flip.png`, `results/decoder/speedup.png`.
- Exports at 1920x1080, 30 fps. The renderer videos are 1280x720 (wipe) and 1920x540 (v1 vs v2), so letterbox rather than upscale.

## Slides: 3

**Slide 1, Problem and approach.** Title, four names (and 184/284 marking for the later presentation), the one-sentence story, a 3-box pipeline (compressor, decoder, renderer) and the hook image (`wipe_poster.jpg`).
Footer: starting points, stated plainly: the NTC idea (Vaidyanathan et al., 2023), nvdiffrast, assimp, flip-evaluator, NVIDIA ORCA Bistro (CC BY 4.0). *Confirm the list before recording: the log mentions tiny-cuda-nn and the NTC SDK as examples, but this repo does not import them.*

**Slide 2, What works (the evidence).** Left: quality versus bits (`rd_flip.png`) with the three headline numbers (5.7 bits, -23 % FLIP, rounding 48 of 48).
Right: decoder speedup (`speedup.png`): FP4-direct 1.2x over FP16-unpack, FP16 PyTorch 4.2 ms for scale. One-line caption on what "FP4-direct" means for graphics people: the stored 4-bit codes and scales go to the Tensor Core without being expanded.

**Slide 3, Bistro and the plan.** Left: original | ours crop with the FLIP map and the numbers (0.0064, 2.1x fewer bytes, 91 % of pixels), plus the honest line: "concat4 with FP4 weights costs about a third more FLIP than bilinear; closing this is the main open item".
Right: updated plan to 12/6 as a five-row table, with the 11/27 gate (if all-material Bistro is not working, switch the headline scene to FlightHelmet, as the proposal allowed):

| dates | goal |
|---|---|
| 11/23 to 11/27 | fit all interior materials; binned decode; recover FP4 quality |
| 11/27 | gate: Bistro or FlightHelmet as headline scene |
| 11/28 to 12/3 | idle-GPU timing per format, BC baseline, fly-through video, display decision |
| 12/1 to 12/6 | paper (4 to 6 pages) and final video |

Keep each slide under about 40 words besides figures; the video carries the narration.

## Webpage (GitHub Pages)

One page, same three parts as the slides, plus links to the report PDF, the video, the slides, the repo and PR #1. Embed `wipe_original_vs_v2.mp4` (1 MB) directly. Include the starting-points line and, for the final, per-member contributions.
The repo already has `docs/proposal/` as a Pages-style page to copy the styling from.

## Schedule for today (no late days)

| by | who | task |
|---|---|---|
| +1 h | all | read this plan; confirm speaker assignment, starting-point list, and that the "do not say" list is respected |
| +1.5 h | I can do now | cut the clips and export the stills at 1080p into one folder; draft the three slides as a deck; draft the webpage |
| +2.5 h | all | Zoom recording, one take per segment; each re-records only on a factual slip |
| +4 h | video owner | edit, caption, export; upload unlisted; check the link from a logged-out window |
| +4.5 h | slides, webpage owners | export slides to PDF, publish the page, check every link |
| +5.5 h | report owner | put the three URLs in `milestone.tex`, team read, push to Overleaf, download the PDF, **click every link in the PDF** |
| 22:00 | report owner | submit on Gradescope; confirm the submission shows all pages and working links |
| 23:59 | | hard deadline |

## Risks

- **Rendered-only result.** Everything in the video is on one view with one seed per material on busy GPUs; the script says "FLIP 0.006" not "better than BC".
- **Someone cannot record.** Each segment is 10 to 15 s and can be recorded alone on a phone against a plain wall; the editor places it in the same slot.
- **Link permissions.** Unlisted YouTube works for staff; a Drive link must be Berkeley-visible. Test in a private window.
- **Time.** Cut slide 2's right half and the B crop strip first; the minimum acceptable set is: wipe clip, one chart, the plan.
