# Sixteen Wide

Neural texture compression with block-scaled latents (MX / NVFP4) and
shading-aware quantization-aware training. CS184/284A final project; the
paper is in `paper/` (Overleaf submodule).

## Layout

- `python/sixteen/`: compressor (PyTorch)
  - `quant.py`: simulated ("fake") quantization, i.e. round in the forward pass, straight-through gradient, for fp16, int8/int4, MXINT8, MXFP8, MXFP6, MXFP4, NVFP4
  - `model.py`: latent grids + MLP decoder
  - `shading.py`: differentiable GGX shading (training loss and evaluation)
  - `train.py`: QAT training and evaluation of one run
  - `sweep.py`, `figures.py`: sweeps from `configs/`, plots and tables
  - `decoder.py`: packing + binding for the CUDA decoder (`cuda/ntc_decode.cu`)
- `cuda/`: Tensor Core decoder kernel; `cuda/probe/` checks block-scaled FP4 MMA (build target `sm_120a`)
- `python/tests/`: unit tests (decoder tests need a GPU and a trained NVFP4 checkpoint)
- `results/`: finished experiments, with `results/README.md` as the step-by-step summary
- `docs/paper-brief.md`: contributions, outline and figure list for the paper
- `docs/proposal-changes.md`: what changed since the proposal, and why
- `docs/renderer-spec.md`: plan for the Bistro renderer
- `docs/proposal/`: copy of the original proposal page, for context
- `configs/`: sweep definitions
- `tools/`: data download, latent profiling, rounding experiment, decoder benchmark
- `out/`: run outputs (not tracked)

## Quick start

Install [pixi](https://pixi.sh), then from the repository root:

```bash
pixi run fetch                                       # 3 Poly Haven materials into data/
pixi run test
CUDA_VISIBLE_DEVICES=0 pixi run sweep configs/milestone.yaml --shard 0/4   # ... shards 1-3 in parallel
pixi run figures out/milestone --out <plots dir>
```

Each run writes `out/<sweep>/<material>/<grid>/<loss>/<format>/metrics.json`
plus image crops; finished runs are skipped, so sweeps resume.
