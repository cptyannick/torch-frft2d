import math

import pytest
import torch

from torch_frft2d import __version__, frft2d_multi
from torch_frft2d._frft2d import _cdft, _parity

DT = torch.complex128


def rel(u: torch.Tensor, v: torch.Tensor) -> float:
    return ((u - v).abs().norm() / v.abs().norm()).item()


def grid(n: int) -> torch.Tensor:
    return (torch.arange(n, dtype=torch.float64) - n // 2) / math.sqrt(n)


def hermite_gauss(n: int, x: torch.Tensor) -> torch.Tensor:
    """Hermite-Gaussian eigenfunction of the FrFT in this normalization (unnormalized)."""
    z = math.sqrt(2 * math.pi) * x
    h0, h1 = torch.ones_like(z), 2 * z
    if n == 0:
        h = h0
    else:
        for k in range(1, n):
            h0, h1 = h1, 2 * z * h1 - 2 * k * h0
        h = h1
    return h * torch.exp(-math.pi * x**2)


@pytest.fixture
def confined_field() -> torch.Tensor:
    """Random (2, 64, 64) field confined to the centered window in space and frequency."""
    torch.manual_seed(0)
    M = N = 64
    env = torch.exp(-math.pi * (grid(N)[None, :] ** 2 + grid(M)[:, None] ** 2) / 4)
    f = torch.randn(2, M, N, dtype=DT) * env
    f = _cdft(_cdft(f, -1), -2) * env
    return _cdft(_cdft(f, -1, True), -2, True)


def test_version():
    assert isinstance(__version__, str)


# 33 samples truncate the Gaussian tails a little more than the larger grids
@pytest.mark.parametrize(("M", "N", "tol"), [(96, 128, 1e-12), (95, 127, 1e-12), (64, 33, 1e-8)])
def test_hermite_gauss_eigenfunctions(M, N, tol):
    p, q = 2, 3
    f = (hermite_gauss(p, grid(M))[:, None] * hermite_gauss(q, grid(N))[None, :]).to(DT)[None]
    phis = torch.tensor(
        [[0.0, 0.1, 0.4, 0.8, math.pi / 2, 2.0, 2.9, 3.6, 5.0, -0.3, -2.0, math.pi]],
        dtype=torch.float64,
    )
    out = frft2d_multi(f, phis)
    assert out.shape == (1, phis.size(1), M, N)
    for i in range(phis.size(1)):
        expected = torch.exp(-1j * (p + q) * phis[0, i]) * f[0]
        assert rel(out[0, i], expected) < tol


def test_matches_torch_frft_separable(confined_field):
    frft = pytest.importorskip("torch_frft.frft_module").frft
    f = confined_field
    phis = torch.tensor([[0.3, 0.9, 1.4, 2.2], [2.7, 3.3, -0.6, -1.3]], dtype=torch.float64)
    out = frft2d_multi(f, phis)
    for bi in range(2):
        for ai in range(phis.size(1)):
            o = phis[bi, ai] * 2 / math.pi
            ref = frft(frft(f[bi], o, dim=-1), o, dim=-2)
            assert rel(out[bi, ai], ref) < 1e-4


def test_additivity(confined_field):
    a1 = torch.tensor([[0.7]], dtype=torch.float64)
    a2 = torch.tensor([[0.5]], dtype=torch.float64)
    f = confined_field[:1]
    two = frft2d_multi(frft2d_multi(f, a1)[:, 0], a2)[0, 0]
    one = frft2d_multi(f, a1 + a2)[0, 0]
    assert rel(two, one) < 1e-4


def test_integer_orders_exact(confined_field):
    f = confined_field
    phis = torch.tensor([[0.0, math.pi / 2, -math.pi / 2, math.pi]] * 2, dtype=torch.float64)
    out = frft2d_multi(f, phis)
    Ff = _cdft(_cdft(f, -1), -2)
    torch.testing.assert_close(out[:, 0], f)
    torch.testing.assert_close(out[:, 1], Ff)
    torch.testing.assert_close(out[:, 2], _cdft(_cdft(f, -1, True), -2, True))
    torch.testing.assert_close(out[:, 3], _parity(f, (-2, -1)))


def test_integer_orders_fast_path_close(confined_field):
    f = confined_field
    phis = torch.tensor([[0.0, math.pi / 2, -math.pi / 2, math.pi]] * 2, dtype=torch.float64)
    exact = frft2d_multi(f, phis)
    fast = frft2d_multi(f, phis, exact_integer_orders=False)
    for i in range(phis.size(1)):
        assert rel(fast[:, i], exact[:, i]) < 1e-3


def test_periodic_in_angle(confined_field):
    phis = torch.tensor([[0.4, 1.9], [2.6, -0.8]], dtype=torch.float64)
    a = frft2d_multi(confined_field, phis)
    b = frft2d_multi(confined_field, phis + 2 * math.pi)
    assert rel(b, a) < 1e-10


def test_unitary(confined_field):
    phis = torch.tensor([[0.3, 1.2, 2.5]] * 2, dtype=torch.float64)
    out = frft2d_multi(confined_field, phis)
    n_in = confined_field.abs().pow(2).sum((-2, -1))[:, None]
    n_out = out.abs().pow(2).sum((-2, -1))
    torch.testing.assert_close(n_out, n_in.expand_as(n_out), rtol=1e-4, atol=0)


def test_angle_chunk_matches(confined_field):
    phis = torch.rand(2, 7, dtype=torch.float64) * 2 * math.pi
    full = frft2d_multi(confined_field, phis)
    chunked = frft2d_multi(confined_field, phis, angle_chunk=3)
    torch.testing.assert_close(chunked, full)


@pytest.mark.parametrize(
    ("in_dtype", "out_dtype"),
    [
        (torch.float32, torch.complex64),
        (torch.float64, torch.complex128),
        (torch.complex64, torch.complex64),
        (torch.complex128, torch.complex128),
    ],
)
def test_dtypes(in_dtype, out_dtype):
    f = torch.randn(1, 16, 16).to(in_dtype)
    out = frft2d_multi(f, torch.tensor([[0.3, 1.0]]))
    assert out.dtype == out_dtype
    assert out.shape == (1, 2, 16, 16)


def test_real_input_matches_complex(confined_field):
    f = confined_field.real
    phis = torch.tensor([[0.6, 2.1]] * 2, dtype=torch.float64)
    torch.testing.assert_close(frft2d_multi(f, phis), frft2d_multi(f.to(DT), phis))


def test_angle_gradient_vs_finite_differences(confined_field):
    f = confined_field[:1]
    w = torch.rand(f.shape[-2:], dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    loss = lambda ang: (frft2d_multi(f, ang).abs() ** 2 * w).sum()
    ang = torch.tensor([[0.4, 1.1, 2.8]], dtype=torch.float64, requires_grad=True)
    loss(ang).backward()
    eps = 1e-6
    for i in range(ang.size(1)):
        e = torch.zeros_like(ang)
        e[0, i] = eps
        with torch.no_grad():
            fd = (loss(ang + e) - loss(ang - e)) / (2 * eps)
        assert abs(ang.grad[0, i].item() - fd.item()) / abs(fd.item()) < 1e-6


def test_field_gradcheck():
    torch.manual_seed(0)
    f = torch.randn(1, 6, 8, dtype=DT, requires_grad=True)
    ang = torch.tensor([[0.4, 2.0]], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: frft2d_multi(x, ang), (f,))


def test_angle_gradient_zero_at_integer_orders(confined_field):
    ang = torch.tensor([[0.0, math.pi / 2, 0.7]], dtype=torch.float64, requires_grad=True)
    frft2d_multi(confined_field[:1], ang).abs().pow(2).mul(torch.rand(64, 64)).sum().backward()
    assert ang.grad[0, 0] == 0 and ang.grad[0, 1] == 0
    assert ang.grad[0, 2] != 0


def test_no_grad_path_matches_grad_path(confined_field):
    phis = torch.tensor([[0.5, 1.7]] * 2, dtype=torch.float64)
    with torch.no_grad():
        a = frft2d_multi(confined_field, phis)
    b = frft2d_multi(confined_field.clone().requires_grad_(), phis)
    torch.testing.assert_close(b.detach(), a)


@pytest.mark.parametrize(
    ("field_shape", "angle_shape", "kwargs"),
    [
        ((4, 4), (1, 1), {}),
        ((1, 4, 4), (1,), {}),
        ((1, 4, 4), (2, 1), {}),
        ((1, 4, 4), (1, 1), {"angle_chunk": 0}),
    ],
)
def test_invalid_input(field_shape, angle_shape, kwargs):
    with pytest.raises(ValueError):
        frft2d_multi(torch.zeros(field_shape, dtype=DT), torch.zeros(angle_shape), **kwargs)
