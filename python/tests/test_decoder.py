"""CUDA decoder vs PyTorch reference on the same STF taps (needs a GPU and a trained NVFP4 checkpoint)."""

import pathlib

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
CKPTS = sorted(ROOT.glob("out/*/*/m/*/nvfp4*/model.pt"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not CKPTS, reason="needs CUDA and an NVFP4 checkpoint")


@pytest.fixture(scope="module")
def dec():
    from sixteen.decoder import CudaDecoder

    return CudaDecoder.from_checkpoint(CKPTS[0])


def test_packing_matches_quantizer(dec):
    from sixteen.quant import nvfp4_quant

    for g, lat in zip(dec.grids, dec.model.latents):
        assert torch.equal(g["deq"], nvfp4_quant(lat.detach()))


def test_stf_taps_unbiased():
    from sixteen.decoder import stf_texels

    # A point 30 % of the way from texel 0's center to texel 1's: tap 1 should be picked ~30 % of the time.
    n = 200000
    uv = torch.tensor([[(0.5 + 0.3) / 8, 0.5 / 8]]).repeat(n, 1)
    seeds = torch.randint(0, 2**32, (n,), dtype=torch.int64)
    t = stf_texels(uv, 8, 8, seeds)
    assert abs((t == 1).float().mean().item() - 0.3) < 0.01
    assert set(t.unique().tolist()) <= {0, 1}


def test_kernel_matches_reference(dec):
    torch.manual_seed(0)
    uv = torch.rand(1 << 20, 2, device="cuda")
    for frame in (0, 7):
        out = dec(uv, frame)
        ref16 = dec.reference(uv, frame, half=True)
        ref32 = dec.reference(uv, frame)
        assert (out - ref16).abs().max().item() < 2e-2  # same taps, fp16 math in both
        assert (out - ref32).abs().mean().item() < 2e-3
