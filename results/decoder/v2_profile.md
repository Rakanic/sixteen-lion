# Decoder v2 profile (idle RTX PRO 6000 Blackwell, castle_brick_07, concat4 + NVFP4 weights)

> **Timings superseded** (2026-11-22): canonical decode times are in `v2_baselines.json` (current kernel, one run, incl. two-part weights). The Nsight Compute analysis below still holds.

One 1920x1080 frame, one lookup per pixel. Event timing at normal clocks
(`v2_idle_time_tiles*.json`, best of 3 x 200 launches) and Nsight Compute 2025.2
(`--set full`, locked base clocks; report `out/profile_decoder/v2.ncu-rep`).

| texture tiling across the screen | FP16-unpack (ms) | FP4-direct (ms) | speedup |
|---|---|---|---|
| 1x | 0.243 | 0.243 | 1.00x |
| 4x | 0.252 | 0.247 | 1.02x |
| 16x | 0.256 | 0.250 | 1.02x |

| Nsight Compute (locked clocks) | FP16-unpack | FP4-direct |
|---|---|---|
| duration (µs) | 373–383 | 316–319 |
| SM throughput / memory throughput (%) | 57 / 51 | 49 / 59 |
| DRAM throughput (%) | 10 | 12 |
| L1 / L2 hit rate (%) | 47 / 88 | 34 / 91 |
| registers per thread; achieved occupancy (%) | 80; 40 | 72; 39 |
| cycles with no eligible warp (%) | 49 | 47 |
| warp stalls per issued instruction: math pipe throttle | 3.04 | 1.34 |
| ... long scoreboard (memory) | 1.20 | 2.29 |
| ... wait (fixed latency) | 2.00 | 1.67 |
| ... short scoreboard | 0.21 | 0.53 |

**Reading.** The FP16-unpack path is limited by its math pipes (unpacking plus 9
FP16 MMA K-tiles in layer 1). FP4-direct removes that (math-pipe stalls 3.0 →
1.3 per instruction) but then waits on the scattered latent fetches (memory
stalls 1.2 → 2.3): the bottleneck moves from arithmetic to fetch latency, so
end-to-end time drops only 2 % at normal clocks (16 % at locked base clocks).
Occupancy is register-limited (≈ 40 %), which leaves little to hide that latency.
Next: overlap the next tile's latent fetches with the current tile's MMAs
(software pipelining) and cut registers to raise occupancy.

## Launch configuration and best-vs-best timing (`v2_best_vs_best.json`)

Each path at its own best setting from a sweep of register caps (blocks per SM
the compiler must fit: 1–4) and grid sizes (4–16 blocks per SM); 5 interleaved
repeats of 200 launches, median [min–max]:

| texture tiling | FP16-unpack (ms) | FP4-direct (ms) | speedup |
|---|---|---|---|
| 1x | 0.268 [0.248–0.270] | 0.222 [0.206–0.224] | 1.21x |
| 4x | 0.274 [0.273–0.274] | 0.228 [0.228–0.229] | 1.20x |
| 16x | 0.284 [0.283–0.284] | 0.240 [0.240–0.240] | 1.18x |

Best settings: FP16-unpack at the default (no cap, 4 blocks/SM); FP4-direct with
registers capped for 4 blocks per SM. FP4-direct gains from more resident warps
because its bottleneck is fetch latency; FP16-unpack does not (it is
arithmetic-bound, and the cap makes it spill: 0.37 ms). The earlier 1.00–1.02x
compared both at the default setting.

## Baselines (`v2_baselines.json`, `tools/bench_baselines.py`; chart `speedup.png` / `speedup.pdf`)

Same work per frame (STF texel choice, 4 neighbours of both grids, positional
encoding, 3 layers), latents and layer-1 weights pre-quantized once; all
outputs agree within 0.003 of the fp32 reference.

| decoder | ms / 1080p frame | FP4-direct faster by |
|---|---|---|
| PyTorch eager fp32 (CUDA cores) | 19.3 | 84.9x |
| PyTorch eager fp32, TF32 Tensor Cores | 19.1 | 83.9x |
| PyTorch eager fp16 (Tensor Cores) | 15.8 | 69.4x |
| PyTorch compiled fp32 (`torch.compile`, max-autotune) | 5.83 | 25.7x |
| PyTorch compiled fp16 (Tensor Cores) | 4.22 | 18.6x |
| ours, fused CUDA, FP16-unpack | 0.275 | 1.21x |
| ours, fused CUDA, FP4-direct | 0.227 | 1x |

The gap to PyTorch is fusion (one pass, everything in registers), which is
standard practice (fully fused MLPs, tiny-cuda-nn); the FP4-specific gain is
the 1.21x over our own fused FP16 kernel.

## The FP4 path really uses FP4 Tensor Cores (SASS, `cuobjdump -sass`)

| kernel | Tensor Core instructions |
|---|---|
| FP4-direct | 16 x `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` (2 grids x 8 neuron groups) + 48 x `HMMA.16816.F32` (PE 8 + layers 2-3 40) |
| FP16-unpack | 112 x `HMMA.16816.F32` (layer 1: 64 + 8, layers 2-3: 40) |
