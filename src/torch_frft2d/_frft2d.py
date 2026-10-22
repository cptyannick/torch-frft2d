"""
Batched, multi-angle 2D fractional Fourier transform (fast, O(MN log MN) per angle).

    out = frft2d_multi(field, angles)
        field  : complex (B, M, N)
        angles : real    (B, A)   rotation angle phi in RADIANS (order a = 2*phi/pi)
        out    : complex (B, A, M, N)

The same angle is applied to both image axes (F^a along x and along y). Grid and
normalization follow torch-frft's `frft` applied along dim -1 and then dim -2:
centered grid with coordinate (n - N//2)/sqrt(N), so a = 1 is the centered
unitary DFT. Like any fast FrFT, it assumes the field is (approximately)
confined to that window in BOTH space and frequency.

Algorithm (Ozaktas et al. 1996, chirp-multiply / chirp-convolve / chirp-multiply)
as two 1D stages, with everything angle-independent computed once per field:

  once per field (shared by all A angles):
    * centered unitary 2D DFT  Ff                    (for the order reduction)
    * 2x band-limited interpolation of f and Ff along x

  per (field, angle):
    * order reduction outside 0.5 <= |a| <= 1.5: start from Ff with order a-1
      -> a select, not an extra transform (torch-frft runs a full extra core)
    * x stage: shift + input chirp + zero padding fused into one write of the
      FFT buffer, FFT of length 4N (minimal; torch-frft pads to 16N), multiply by
      the analytic-kernel spectrum, fold the spectrum so the 2x output decimation
      becomes a half-length (2N) inverse FFT
    * the x output chirp and amplitude commute with the y stage -> deferred
    * y stage: 2x interpolation along y, same fused chirp convolution
    * one final elementwise multiply (both output chirps, amplitude)
    * exact results for integer orders 0, +-1, 2 (identity / DFT / IDFT / parity)

Differentiable w.r.t. `field` and `angles` (the angle gradient is zero exactly
at integer orders when exact_integer_orders=True).

References
    H. M. Ozaktas, O. Arikan, M. A. Kutay, G. Bozdagi, "Digital computation of
    the fractional Fourier transform", IEEE Trans. Signal Process. 44(9),
    2141-2150, 1996. doi:10.1109/78.536672
    E. Koc, T. Alikasifoglu, A. C. Aras, A. Koc, "Trainable Fractional Fourier
    Transform", IEEE Signal Process. Lett. 31, 751-755, 2024.
    doi:10.1109/LSP.2024.3372779  (torch-frft, the 1D reference implementation)
"""

from __future__ import annotations

import math

import torch
from torch.fft import fft, fftshift, ifft, ifftshift

__all__ = ["frft2d_multi"]


# ----------------------------------------------------------------------------
# helpers (all 1D operations act on the LAST dim; the y stage works on a transpose)
# ----------------------------------------------------------------------------
def _cdft(x: torch.Tensor, dim: int, inverse: bool = False) -> torch.Tensor:
    """Centered unitary DFT along `dim` (index n <-> coordinate n - N//2)."""
    f = ifft if inverse else fft
    return fftshift(f(ifftshift(x, dim=dim), dim=dim, norm="ortho"), dim=dim)


def _parity(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """x(n) -> x(-n) on the centered grid (periodic): F^2 = parity."""
    for d in dims:
        x = x.flip(d)
        if x.size(d) % 2 == 0:  # centered even grid is [-n/2, n/2): flip + roll by 1
            x = x.roll(1, dims=d)
    return x


def _interp2x_last(x: torch.Tensor) -> torch.Tensor:
    """
    Band-limited 2x interpolation along the last dim.
    In: centered samples (..., n). Out: (..., 2n) in *unshifted* order
    (index j <-> m = j for j < n, m = j - 2n otherwise; coordinate m / (2 sqrt n)).
    """
    n = x.size(-1)
    X = fft(ifftshift(x, dim=-1), dim=-1)
    Xp = X.new_empty(*X.shape[:-1], 2 * n)
    h = n // 2
    Xp[..., h + 1 : 2 * n - h].zero_()        # high-frequency gap; rest is overwritten
    if n % 2:
        Xp[..., : h + 1] = X[..., : h + 1]
        Xp[..., 2 * n - h :] = X[..., h + 1 :]
    else:
        Xp[..., :h] = X[..., :h]
        if h > 1:
            Xp[..., 2 * n - h + 1 :] = X[..., h + 1 :]
        nyq = 0.5 * X[..., h]          # split Nyquist bin: keeps real inputs real
        Xp[..., h] = nyq
        Xp[..., 2 * n - h] = nyq
    return ifft(Xp, dim=-1) * 2.0


def _axis_terms(n: int, b: torch.Tensor, ctype):
    """
    Angle-dependent 1D factors for one axis of length n, orders b (B, A) with
    0.5 <= |b| <= 1.5. All tiny compared to the data.
        cin   (B, A, 2n)  input chirp on the 2x grid, unshifted order
        K0,K1 (B, A, 2n)  halves of 0.5 * FFT_L(kernel), L = 4n (fold for decimation)
        cout  (B, A, n)   output chirp on the original centered grid
        amp   (B, A)      A_phi / (2 sqrt n)
    """
    L = 4 * n
    phi = b * (math.pi / 2)
    tan_half = torch.tan(phi / 2)[..., None]
    csc = (1.0 / torch.sin(phi))[..., None]
    sq = math.sqrt(n)

    j = torch.arange(2 * n, dtype=b.dtype, device=b.device)
    m = torch.where(j < n, j, j - 2 * n)                    # unshifted 2x grid
    ang = -math.pi * tan_half * (m / (2 * sq)) ** 2
    cin = torch.polar(torch.ones_like(ang), ang)

    # kernel c(d) = exp(i pi csc (d / (2 sqrt n))^2), |d| <= 2n-1, wrapped mod L
    d = torch.arange(L, dtype=b.dtype, device=b.device)
    d = torch.where(d >= L // 2, d - L, d)
    ang = math.pi * csc * (d / (2 * sq)) ** 2
    ker = torch.polar((d.abs() <= 2 * n - 1).to(b.dtype).expand_as(ang), ang)
    K = 0.5 * fft(ker.to(ctype), dim=-1)

    u = (torch.arange(n, dtype=b.dtype, device=b.device) - n // 2) / sq  # original centered grid
    ang = -math.pi * tan_half * u**2
    cout = torch.polar(torch.ones_like(ang), ang)

    s = torch.sin(phi)
    amp = torch.polar(1.0 / torch.sqrt(s.abs()), -(math.pi / 4) * torch.sign(s) + phi / 2)
    amp = amp / (2 * sq)
    return cin.to(ctype), K[..., : 2 * n], K[..., 2 * n :], cout.to(ctype), amp.to(ctype)


def _chirp_conv_last(y0, cin, K0, K1):
    """
    Core chirp-multiply + chirp-convolution along the last dim, output decimated.
        y0  (..., R, 2n)  2x-interpolated signal, unshifted order
        cin, K0, K1 broadcastable as (..., 1, 2n)
    Returns (..., R, n): the convolution on the original grid (no output chirp/amp).

    The shift to centered order, the chirp multiplication and the zero padding to
    L = 4n are a single write into the FFT buffer. Decimation by 2 is a spectral
    fold, so the inverse FFT has length 2n instead of 4n.
    """
    n = y0.size(-1) // 2
    s = n % 2                     # odd n: offset by 1 so even outputs land on even indices
    buf = y0.new_empty(*y0.shape[:-1], 4 * n)
    buf[..., :s].zero_()                      # only the padding needs zeroing
    buf[..., 2 * n + s :].zero_()
    if torch.is_grad_enabled() and (y0.requires_grad or cin.requires_grad):
        buf[..., n + s : 2 * n + s] = y0[..., :n] * cin[..., :n]  # autograd-safe path
        buf[..., s : n + s] = y0[..., n:] * cin[..., n:]
    else:  # fused: multiply straight into the padded FFT buffer
        torch.mul(y0[..., :n], cin[..., :n], out=buf[..., n + s : 2 * n + s])  # m >= 0
        torch.mul(y0[..., n:], cin[..., n:], out=buf[..., s : n + s])          # m <  0
    Y = fft(buf, dim=-1)
    Z = Y[..., : 2 * n] * K0
    Z.addcmul_(Y[..., 2 * n :], K1)
    return ifft(Z, dim=-1)[..., s : s + n]


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
def frft2d_multi(
    field: torch.Tensor,
    angles: torch.Tensor,
    *,
    exact_integer_orders: bool = True,
    angle_chunk: int | None = None,
) -> torch.Tensor:
    """
    Fast 2D fractional Fourier transform of a batch of fields at many angles.

    The same angle is applied along both image axes. Grid and normalization match
    ``torch_frft.frft_module.frft`` applied along dim -1 and then dim -2, i.e. the
    centered grid with coordinate ``(n - N//2) / sqrt(N)``; angle ``pi/2`` (order 1)
    is the centered unitary 2D DFT.

    Parameters
    ----------
    field : torch.Tensor
        Complex tensor of shape (B, M, N). Real inputs are promoted to complex
        (float64 -> complex128, everything else -> complex64).
    angles : torch.Tensor
        Real tensor of shape (B, A): rotation angles phi in radians, order
        ``a = 2 * phi / pi``. Any real value is allowed (period ``2 pi``).
    exact_integer_orders : bool, default True
        Return exact identity / DFT / inverse DFT / parity results for orders
        0, 1, -1, 2. The angle gradient is then zero at exactly those orders.
    angle_chunk : int, optional
        Process the angles in groups of this size to limit peak memory
        (about one complex (B, angle_chunk, M, 4N) buffer).

    Returns
    -------
    torch.Tensor
        Complex tensor of shape (B, A, M, N), differentiable w.r.t. both
        ``field`` and ``angles``.

    Notes
    -----
    Like any fast FrFT, the method assumes the field is (approximately) confined
    to the centered window in both space and frequency. Fields that fill the grid
    edge to edge lose accuracy near orders 0 and 2; zero-pad them first.
    """
    if field.dim() != 3:
        raise ValueError("field must be (B, M, N)")
    if angles.dim() != 2 or angles.size(0) != field.size(0):
        raise ValueError("angles must be (B, A) with the same B as field")
    if angle_chunk is not None and angle_chunk < 1:
        raise ValueError("angle_chunk must be a positive integer")
    if not field.is_complex():
        field = field.to(torch.complex128 if field.dtype == torch.float64 else torch.complex64)
    ctype = field.dtype
    rtype = torch.float64 if ctype == torch.complex128 else torch.float32

    # ---- angle-independent work (once per field, shared by all A angles) ----
    Ff = _cdft(_cdft(field, -1), -2)          # centered unitary 2D DFT (= order 1)
    S0 = _interp2x_last(field)                # f  interpolated along x   (B, M, 2N)
    S1 = _interp2x_last(Ff)                   # Ff interpolated along x   (B, M, 2N)

    # orders wrapped to (-2, 2]
    a = angles.to(device=field.device, dtype=rtype) * (2.0 / math.pi)
    a = torch.remainder(a + 2.0, 4.0) - 2.0
    a = torch.where(a == -2.0, a + 4.0, a)

    args = (field, Ff, S0, S1, ctype, exact_integer_orders)
    if angle_chunk is not None and angle_chunk < a.size(1):
        return torch.cat(
            [_frft2d_core(a[:, i : i + angle_chunk], *args)
             for i in range(0, a.size(1), angle_chunk)],
            dim=1,
        )
    return _frft2d_core(a, *args)


def _frft2d_core(a, field, Ff, S0, S1, ctype, exact_integer_orders):
    B, M, N = field.shape

    # order reduction: outside 0.5 <= |a| <= 1.5 start from Ff with order a - 1
    # (F^a = F^(a-1) F on both axes) -> a gather, not an extra transform
    use_F = (a.abs() < 0.5) | (a.abs() > 1.5)
    b = torch.where(use_F, torch.remainder(a + 1.0, 4.0) - 2.0, a)

    cin_x, K0x, K1x, cout_x, amp_x = _axis_terms(N, b, ctype)
    cin_y, K0y, K1y, cout_y, amp_y = _axis_terms(M, b, ctype)
    ex = lambda t: t[..., None, :]            # (B, A, L) -> (B, A, 1, L)

    # ---- x stage (last dim): (B, A, M, 2N) -> (B, A, M, N) ----
    src = torch.where(use_F[..., None, None], S1[:, None], S0[:, None])
    h = _chirp_conv_last(src, ex(cin_x), ex(K0x), ex(K1x))
    # x output chirp/amp commute with the y stage -> applied once at the very end

    # ---- y stage: work on the transpose so the FFTs are contiguous ----
    h = _interp2x_last(h.transpose(-1, -2))   # (B, A, N, 2M)
    h = _chirp_conv_last(h, ex(cin_y), ex(K0y), ex(K1y))  # (B, A, N, M)

    out = h.transpose(-1, -2) * (cout_y * (amp_x * amp_y)[..., None])[..., :, None]
    out = out * cout_x[..., None, :]

    if exact_integer_orders:
        is_int = a == torch.round(a)
        if bool(is_int.any()):
            ai = torch.round(a).long()[..., None, None]
            P = _parity(field, (-2, -1))       # F^2
            PF = _parity(Ff, (-2, -1))         # F^-1 = F^3
            exact = torch.where(ai == 0, field[:, None],
                    torch.where(ai == 1, Ff[:, None],
                    torch.where(ai == -1, PF[:, None], P[:, None])))
            out = torch.where(is_int[..., None, None], exact, out)
    return out
