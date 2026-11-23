# Paper update: the renderer and Bistro work

`paper/` is the Overleaf submodule and needs an Overleaf git token to clone, so this file holds paste-ready text and the checklist instead
of an edit. Source of every number: `results/renderer/` (`neural-compare-v2/neural_compare.json`, `stage1_*.json`) and `docs/renderer-spec.md` §11.
`docs/paper-brief.md` is already updated to match (sections 2, 3, 4, 5, 6).

## What the renderer work gives the paper

1. **An evaluation protocol** (Method, one subsection). FLIP after shading on a rendered view, with the original and neural renders sharing camera, jitter and
   stochastic-tap seeds, so the difference is compression and decoding error only. This is what makes the Bistro numbers meaningful and it
   is cheap to describe.
2. **A scene result** (Results 5.7). Ten fitted Bistro materials rendered through the real CUDA decoder v2, FLIP and bytes against the shipped textures.
3. **Two findings that shape the claims**: Bistro's textures are already block-compressed (so the baseline is BC data, about 12 bits per texel), and the FP4-direct
   model pays a measurable quality price against the bilinear model.
4. **A decoder-fidelity check in a real renderer**: the CUDA render equals the PyTorch decode of the same fits to 2.5e-4.

It does **not** yet give the paper: a frame time, a BC baseline at equal memory, every-material coverage, a fly-through. Those stay `\todo{}`.

## Claims table (what the text may say)

| claim | number | file |
|---|---|---|
| Bistro loads headless at 1080p | interior 1,046,609 triangles, 71 materials | `docs/bistro-inventory.md` |
| Bistro's textures ship block-compressed | BC1/BC3 + BC5 + BC3, about 12 bits per full-res texel (16 with BC3 base colour) | `docs/bistro-inventory.md` |
| Coverage | 10 of 71 interior materials, 91.3 % of the entrance view's pixels | `neural_compare.json` |
| Quality | FLIP 0.0153 on neural pixels, 0.0140 whole frame; PSNR 51.5 / 52.0 dB; 32 frames, ambient 0.3, same seeds | `neural_compare.json` |
| Bytes | 28.4 MiB vs 60.0 MiB BC (top level) for the same ten materials, 80.0 MiB with mips: 2.1x and 2.8x | `neural_compare.json` |
| Decoder fidelity | CUDA v2 vs PyTorch decode of the same fits: max 2.5e-4 display value, FP4 vs FP16 layer-1 path 1.6e-4 | `neural_compare.json` |
| Model trade-off | concat4 + NVFP4 weights: FLIP 0.0153 vs 0.0106 (bilinear, PyTorch decode), worse in 10 of 10 materials; single seed per material | `neural-compare/`, `neural-compare-v2/` |
| v1 bias | latent-space tap stays 20.6 dB from the bilinear decode (already in the brief) | `results/decoder/` |

**Never write**: "better than BC at equal memory" (we compare bytes against decoded BC, not equal-memory quality); a Bistro frame time; "novel" for FP4-direct
before the literature check; the +44 % as a general figure (one scene view, one seed).

## A. Milestone report (`milestone.tex`)

The "Bistro integration not started" sentence is now wrong. Replace the progress-against-plan paragraph's Bistro clause with:

```latex
Bistro now loads headless at $1920{\times}1080$ with its original textures
(1.05\,M triangles, 71 materials), and ten fitted interior materials, covering
91\,\% of the entrance view's pixels, render through the CUDA FP4-direct
decoder. After shading, FLIP against the shipped textures is
$0.0153$ on those pixels at $28.4$\,MiB against $60.0$\,MiB of block-compressed
textures for the same materials. We also found that Bistro's textures are
already block-compressed, so our reference is decoded BC data, and that the
FP4-direct model costs quality against a bilinear one; recovering it is the main
open item.
```

Optional (only if the one-page budget allows): a small two-panel figure of `crop.jpg` (original, ours, FLIP map) with the caption below, replacing the second crop figure if space is tight.

```latex
\begin{figure}[t]
  \includegraphics[width=\linewidth]{figures/bistro_crop.jpg}
  \caption{Bistro interior, ceiling and lamp (crop): decoded BC original, ours
  through the CUDA decoder, and the FLIP error map (mean 0.008 in the crop).
  Ten of 71 materials are neural; 32-frame average with identical camera,
  jitter and tap seeds.}
\end{figure}
```

Also update the plan paragraph to the dates in `docs/paper-brief.md` §3 (11/23 to 11/27 all interior materials and the FP4 quality gap; 11/27 gate; 11/28 to 12/3 timing, BC baseline, fly-through; 12/1 to 12/6 paper), and replace "decoder v1 ... 1 ms on a busy GPU" with the decoder v2 sentence from the brief.
Figure to copy: `results/renderer/neural-compare-v2/crop.jpg` as `figures/bistro_crop.jpg`.

## B. Final paper

### B1. Introduction (teaser and one sentence)

Teaser: the wipe frame from `neural-compare-v2/wipe_poster.jpg` (regenerate at 1080p with all materials for the final). Add after the contributions list:

```latex
We evaluate on the Amazon Lumberyard Bistro scene rendered through our decoder,
comparing shaded frames against the shipped block-compressed textures.
```

### B2. Method, new subsection 4.4 "Rendering and evaluation setup"

```latex
\subsection{Rendering and evaluation setup}
We render Bistro with a deferred renderer: a CUDA rasterizer (nvdiffrast~\cite{laine2020nvdiffrast})
writes a G-buffer of texture coordinates, derivatives, material ids and tangent frames,
each pixel is then decoded and shaded with the same GGX model we train with,
and a temporal filter averages frames. Each pixel decodes one full-resolution
texel per frame, chosen at random with bilinear weights (stochastic texture
filtering~\cite{pharr2024stf}), while the camera jitters inside the pixel;
averaging the decoded colours is exact because colours are linear.
To compare textures and not noise, the original and neural renders share
camera, jitter and the per-pixel random seeds, so they differ only in the
stored textures and their decoding. We report FLIP~\cite{andersson2020flip}
after shading, over a 32-frame average, on the pixels that use neural textures
and on the whole frame. Bistro's textures ship block-compressed, so the
reference is the decoded BC data, and we compare bytes against the same
materials' BC files.
```

### B3. Results, new subsection 5.7 "Bistro"

```latex
\subsection{Bistro}
We fit ten interior materials (91\,\% of the entrance view's pixels) with
a concat4 model, NVFP4 latents and NVFP4 first-layer weights, as the
FP4-direct decoder requires, and render them through the CUDA decoder.
FLIP is $0.0153$ on the neural pixels and $0.0140$ over the frame
(Fig.~\ref{fig:bistro}); the ten materials take $28.4$\,MiB against $60.0$\,MiB
of top-level BC data ($80.0$\,MiB with mips). The CUDA render matches the
PyTorch decode of the same fits to $2.5{\times}10^{-4}$ in display value,
so the kernel adds no visible error. The largest errors are in the two plaster
materials (texel PSNR 32.0 and 33.4\,dB), which hold fine grain that one
2.84\,MiB latent grid cannot carry; smooth materials are close to exact.
Compared with the bilinear model at the same latent bytes, the FP4-direct model
is worse in all ten materials (FLIP $0.0153$ against $0.0106$ on the neural
pixels, single seed). We therefore present FP4-direct as a trade of some
quality for a 1.2$\times$ faster, unpack-free decode and treat closing the gap
as future work. \todo{all-material Bistro, idle-GPU frame time, BC baseline at equal memory}
```

Table (optional): per-material rows from `neural_compare.json` (`psnr_texel`, `flip_flat_plane`, `pixel_share`); the `docs/renderer-spec.md` §11 table has them.
Figure `fig:bistro`: `crop.jpg`, caption as in A with "through the CUDA decoder".

### B4. Limitations and lessons (add bullets)

- The 9-channel representation has no opacity, emissive or glass; cut-outs and lamps render with the original treatment, so coverage is reported per pixel.
- Bistro's reference is itself block-compressed, so "quality" means fidelity to decoded BC data.
- The ambient-occlusion channel of Bistro's specular maps is zero in all 201 maps (the README calls it occlusion); we treat it as no occlusion. A first version of our renderer used it and lit nothing. (One sentence; it is a lesson, not a result.)
- Renderer frame times were measured on shared GPUs and are not reported; the decoder timings in 5.5 are from an idle GPU.

### B5. Related work and credit

Add `laine2020nvdiffrast`, `andersson2020flip` and a Bistro dataset citation (BibTeX below); `pharr2024stf` should already be in the bib from related work. State the starting points in Method or the webpage (nvdiffrast, assimp, ORCA Bistro under CC BY 4.0, the FLIP evaluator).

```bibtex
@article{laine2020nvdiffrast,
  author  = {Laine, Samuli and Hellsten, Janne and Karras, Tero and Seol, Yeongho and Lehtinen, Jaakko and Aila, Timo},
  title   = {Modular Primitives for High-Performance Differentiable Rendering},
  journal = {ACM Transactions on Graphics},
  volume  = {39}, number = {6}, year = {2020}}
@article{andersson2020flip,
  author  = {Andersson, Pontus and Nilsson, Jim and Akenine-M{\"o}ller, Tomas and Oskarsson, Magnus and {\AA}str{\"o}m, Kalle and Fairchild, Mark D.},
  title   = {{FLIP}: A Difference Evaluator for Alternating Images},
  journal = {Proceedings of the ACM on Computer Graphics and Interactive Techniques},
  volume  = {3}, number = {2}, year = {2020}}
@misc{lumberyard2017bistro,
  author = {{Amazon Lumberyard}},
  title  = {Amazon Lumberyard Bistro, Open Research Content Archive ({ORCA})},
  year   = {2017}, howpublished = {\url{https://developer.nvidia.com/orca/amazon-lumberyard-bistro}}, note = {CC BY 4.0}}
```

Check the keys against the bib already in Overleaf before pasting (the project log says the Fujieda and Harada entry needs its spelling and venue fixed).

## Checklist to apply it

- [ ] Get Overleaf access: `git submodule update --init paper` needs the Overleaf git token. Or paste the blocks above into the Overleaf editor.
- [ ] Milestone: replace the Bistro clause (A), add the crop figure if space allows, update the plan paragraph, fill the three URLs.
- [ ] Copy `crop.jpg` into `figures/` (and `wipe_poster.jpg` later for the teaser).
- [ ] Final paper: B1 to B5, then delete the matching `\todo` items in the log: "concat4 quality, v2 timing, Bistro, BC baseline, final rounding, teaser".
- [ ] Re-run the Bistro comparison after all materials are fitted and replace the numbers here, in the brief and in the claims table.
- [ ] Update `.claude/project-log.md` (Bistro tick is done; add the v2-renderer entry) when the PR merges.
