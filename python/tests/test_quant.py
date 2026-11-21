import torch

from sixteen.quant import E2M1, E2M3, E4M3, fake_quant, mx_quant, nvfp4_quant, quantize, round_minifloat


def grid(fmt):
    """Every non-negative value of a MiniFloat, enumerated from its bit fields."""
    vals = set()
    for e in range(2**fmt.ebits):
        for m in range(2**fmt.mbits):
            if e == 0:
                v = m / 2**fmt.mbits * 2.0**fmt.emin
            else:
                v = (1 + m / 2**fmt.mbits) * 2.0 ** (e - fmt.bias)
            if v <= fmt.max:
                vals.add(v)
    return torch.tensor(sorted(vals))


def test_e2m1_values():
    assert grid(E2M1).tolist() == [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def test_round_onto_grid():
    for fmt in (E2M1, E2M3, E4M3):
        g = grid(fmt)
        x = torch.linspace(-1.2 * fmt.max, 1.2 * fmt.max, 10001)
        q = round_minifloat(x, fmt)
        # Every output is representable and is a nearest grid point.
        assert torch.isin(q.abs(), g).all()
        d = (x.abs().clamp(max=fmt.max)[:, None] - g[None]).abs().min(dim=1).values
        assert torch.allclose((q.abs() - x.abs().clamp(max=fmt.max)).abs(), d, atol=1e-6)


def test_mx_block_scale_is_power_of_two():
    torch.manual_seed(0)
    x = torch.randn(64, 32) * torch.logspace(-3, 3, 64)[:, None]
    q = mx_quant(x, E2M1)
    # Within each block the ratio to the block's largest value stays on the E2M1 grid.
    for row_x, row_q in zip(x, q):
        s = 2.0 ** (torch.floor(torch.log2(row_x.abs().max())) - E2M1.emax)
        assert torch.isin((row_q / s).abs(), grid(E2M1)).all()


def test_error_ordering():
    torch.manual_seed(0)
    x = torch.randn(256, 16)
    err = {f: (quantize(x, f) - x).pow(2).mean().item() for f in ["mxint8", "mxfp8", "mxfp6", "mxfp4", "nvfp4"]}
    assert err["mxint8"] < err["mxfp6"] < err["mxfp4"]
    # E4M3 and E2M3 share 3 mantissa bits, so MXFP8 is not more accurate than
    # MXFP6 on well-scaled blocks; it only wins on wide in-block dynamic range.
    assert err["mxfp8"] < 2 * err["mxfp6"]
    assert err["nvfp4"] < err["mxfp4"]  # finer scale and smaller blocks


def test_nvfp4_shape_and_ste():
    x = torch.randn(5, 7, 16, requires_grad=True)
    assert nvfp4_quant(x.detach()).shape == x.shape
    fake_quant(x, "nvfp4").sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
