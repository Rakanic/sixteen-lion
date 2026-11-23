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
- `python/sixteen/render/`: deferred renderer for Bistro (nvdiffrast G-buffer, GGX shading, TAA); plan in `docs/renderer-spec.md`
- `cuda/`: Tensor Core decoder kernel; `cuda/probe/` checks block-scaled FP4 MMA (build target `sm_120a`)
- `python/tests/`: unit tests (decoder tests need a GPU and a trained NVFP4 checkpoint)
- `results/`: finished experiments, with `results/README.md` as the step-by-step summary
- `docs/paper-brief.md`: contributions, outline and figure list for the paper
- `docs/proposal-changes.md`: what changed since the proposal, and why
- `docs/renderer-spec.md`: plan and status of the Bistro renderer; `docs/bistro-inventory.md`: what the Bistro assets contain
- `docs/proposal/`: copy of the original proposal page, for context
- `configs/`: sweep definitions
- `tools/`: data download, latent profiling, rounding experiment, decoder benchmark, Bistro inventory and renderer checks (`render_bistro.py`, `renderer_check.py`)
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

The renderer builds nvdiffrast from source on `pixi install`. That needs a visible GPU (the CUDA
architecture is detected) or `TORCH_CUDA_ARCH_LIST=12.0a pixi install` on a machine without one.

Bistro (for the renderer): download `Bistro_v5_2.zip` from <https://developer.nvidia.com/bistro> (CC BY 4.0), unzip into
`data/bistro-orca/`, then `pixi run python tools/render_bistro.py --scene BistroInterior --ambient 0.3`.

### Renderer: render Bistro with original or compressed textures

```bash
# 1. original textures (trilinear), one frame
pixi run python tools/render_bistro.py --scene BistroInterior --camera entrance --ambient 0.3 --out out/render/a.png

# 2. fit Bistro materials (the 10 most visible in the entrance view) for the decoder you want to test
pixi run python -m sixteen.sweep configs/bistro-v2.yaml      # concat4 + NVFP4 weights, what decoder v2 runs -> out/bistro-v2
pixi run python -m sixteen.sweep configs/bistro-pilot.yaml   # bilinear model, what decoder v1 runs        -> out/bistro-pilot

# 3. render through a decoder (fitted materials from our textures, the rest original); average >= 16 frames
pixi run python tools/render_bistro.py --source neural-v2 --fit out/bistro-v2 --frames 32 --ambient 0.3 --out out/render/b.png

# 4. compare against the original: FLIP, bytes against BC, crop, FLIP map; and the videos
pixi run python tools/bistro_compare.py --fit out/bistro-v2 --out results/renderer/neural-compare-v2
pixi run python tools/bistro_videos.py
```

`--source` is `original` | `original-stf` | `neural-ref` (PyTorch decode of the fits) | `neural-v1` | `neural-v2` | `neural-v2-fp16`.
Python entry points: `sixteen.render.renderer.Renderer` (`render`, `accumulate`), `sixteen.render.neural_ref.neural_source`, and the
decoders in `sixteen.decoder` (`CudaDecoder`, `CudaDecoderV2`). How the renderer and the decoders fit together:
`docs/renderer-spec.md` §3 and §11.
