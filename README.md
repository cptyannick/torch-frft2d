# torch-frft2d

[![Tests](https://github.com/cptyannick/torch-frft2d/actions/workflows/tests.yml/badge.svg)](https://github.com/cptyannick/torch-frft2d/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/cptyannick/torch-frft2d/graph/badge.svg)](https://codecov.io/gh/cptyannick/torch-frft2d)
[![PyPI](https://img.shields.io/pypi/v/torch-frft2d)](https://pypi.org/project/torch-frft2d)
[![Python](https://img.shields.io/pypi/pyversions/torch-frft2d)](https://pypi.org/project/torch-frft2d)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`torch-frft2d` is a fast, differentiable 2D fractional Fourier transform (FrFT) for PyTorch. It takes a batch of fields and many angles per field. The DFT and interpolations that don't depend on the angle are computed once per field and reused for every angle, so the per-angle cost is just the chirp convolutions.

```python
out = frft2d_multi(field, angles)
#   field  : complex (B, M, N)
#   angles : real    (B, A)    rotation angle φ in radians, order a = 2φ/π
#   out    : complex (B, A, M, N)
```

The same angle is applied along both image axes. The grid and normalization match [torch-frft](https://github.com/tunakasif/torch-frft)'s `frft` applied along `x` (dim −1) and then `y` (dim −2). The grid is centered, with coordinate `(n − N//2)/√N`, so `φ = π/2` (order 1) is the centered unitary 2D DFT.

## Installation

```sh
pip install torch-frft2d
# or
uv add torch-frft2d
```

The package requires Python ≥ 3.10 and PyTorch ≥ 2.1. It runs on whatever device the input tensor is on.

## Usage

```python
import math
import torch
from torch_frft2d import frft2d_multi

device = "cuda" if torch.cuda.is_available() else "cpu"

B, A, M, N = 4, 16, 256, 256
field = torch.randn(B, M, N, dtype=torch.complex64, device=device)
angles = torch.rand(B, A, device=device) * 2 * math.pi  # radians

out = frft2d_multi(field, angles)  # (B, A, M, N), complex64
```

### Gradients

Gradients flow to both the field and the angles, so angles can be learned:

```python
angles = torch.full((B, 1), 0.7, device=device, requires_grad=True)
loss = frft2d_multi(field, angles).abs().pow(2).mean()
loss.backward()
angles.grad  # (B, 1)
```

### Options

| keyword                | default | meaning                                                                                                                                                     |
| ---------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `exact_integer_orders` | `True`  | Return exact identity / DFT / inverse DFT / parity for orders 0, 1, −1, 2. The angle gradient is zero at exactly those orders. Set to `False` to use the fast path everywhere. |
| `angle_chunk`          | `None`  | Process the angles in groups of this size to limit peak memory.                                                                                            |

Real inputs are promoted to complex: `float64` becomes `complex128`, and everything else becomes `complex64`.

## Algorithm

The transform is the chirp-multiply, chirp-convolve, chirp-multiply method of Ozaktas et al. (1996), the same one torch-frft uses. Here it runs as two separable 1D stages, with everything that doesn't depend on the angle computed once per field.

**Once per field, shared by all `A` angles:**

- the centered unitary 2D DFT of the field;
- a band-limited 2× interpolation of the field and of its DFT along `x`.

Because both are already available, the order reduction (from `F^a` to `F^(a−1)·F` outside `0.5 ≤ |a| ≤ 1.5`) only picks which tensor to start from. It doesn't cost an extra transform.

**Per angle:**

- **x stage.** The shift, input-chirp multiply and zero padding happen in a single write into a length-4N FFT buffer. The FFT result is multiplied by the analytic kernel spectrum. The output's 2× decimation is folded into the spectrum, so the inverse FFT is only half length.
- **Deferred output chirp.** The `x` output chirp and amplitude commute with the `y` stage, so they're applied together with the `y` chirp in one final multiply.
- **y stage.** The same steps run on the transposed data, so the FFTs work on contiguous memory.
- **Integer orders.** Orders 0, ±1 and 2 are returned exactly when `exact_integer_orders=True`.

### Accuracy

The test suite in [`tests/`](tests/) checks the following:

| check                                                         | max relative error |
| ------------------------------------------------------------- | ------------------ |
| Hermite–Gaussian eigenfunctions (including odd sizes)          | ~1e-14             |
| torch-frft applied per axis                                   | ~6e-6              |
| additivity `F^a·F^b` vs `F^(a+b)`                             | ~6e-6              |
| angle gradient vs finite differences                          | ~5e-9              |

### Performance

On a 2-core CPU with `B=2, A=16`, 256×256 and `complex64`, the transform took about 480 ms. Looping torch-frft over the same inputs took about 2 s, so this is roughly 4× faster. It's still about 20× a plain `fft2` per image, which is the inherent cost of the chirp-convolution method. To reproduce (it picks CUDA automatically if available):

```sh
uv run python benchmarks/benchmark.py --B 2 --A 16 --size 256
```

### Caveats

- **Memory.** Peak memory is about one complex `(B, A, M, 4N)` buffer. Use `angle_chunk=` if that's too much.
- **Field coverage.** Like any fast FrFT, the method assumes the field fits inside the centered window in both space and frequency. Fields that fill the grid edge to edge lose accuracy near orders 0 and 2. Zero-pad such fields first.

## Development

The project is managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync                       # create .venv with dev dependencies
uv run pytest --cov           # tests + coverage
uv run ruff check . && uv run ruff format --check .
uv run python benchmarks/benchmark.py
```

CI runs on every push and pull request. It tests Python 3.10–3.13 on Linux, plus macOS and Windows, and one job uses the oldest supported dependencies (torch 2.1). Coverage is uploaded to Codecov.

### Releasing

1. Bump `version` in `pyproject.toml` (`uv version --bump minor`) and `CITATION.cff`, then commit.
2. Create a GitHub release with tag `vX.Y.Z`. The [publish workflow](.github/workflows/publish.yml) builds the package, checks that the tag matches the version, and uploads to PyPI via trusted publishing.

## References

The fast FrFT algorithm follows:

- H. M. Ozaktas, O. Arikan, M. A. Kutay, G. Bozdagı, [_Digital computation of the fractional Fourier transform_](https://ieeexplore.ieee.org/document/536672), IEEE Transactions on Signal Processing, 44(9):2141–2150, 1996. doi:[10.1109/78.536672](https://doi.org/10.1109/78.536672). The MATLAB reference implementation is [`fracF.m`](http://www.ee.bilkent.edu.tr/~haldun/fracF.m) on [Haldun M. Özaktaş's page](http://www.ee.bilkent.edu.tr/~haldun/wileybook.html).

The grid, normalization and validation target is the 1D PyTorch implementation [torch-frft](https://github.com/tunakasif/torch-frft), developed for:

- E. Koç, T. Alikaşifoğlu, A. C. Aras, A. Koç, [_Trainable Fractional Fourier Transform_](https://ieeexplore.ieee.org/document/10458263), IEEE Signal Processing Letters, 31:751–755, 2024. doi:[10.1109/LSP.2024.3372779](https://doi.org/10.1109/LSP.2024.3372779).

If you use this package, please cite those works:

```bibtex
@article{ozaktas1996digital,
  author  = {Ozaktas, Haldun M. and Arikan, Orhan and Kutay, M. Alper and Bozdagi, Gozde},
  title   = {Digital computation of the fractional {F}ourier transform},
  journal = {IEEE Transactions on Signal Processing},
  year    = {1996},
  volume  = {44},
  number  = {9},
  pages   = {2141--2150},
  doi     = {10.1109/78.536672}
}

@article{trainable-frft-2024,
  author  = {Koç, Emirhan and Alikaşifoğlu, Tuna and Aras, Arda Can and Koç, Aykut},
  title   = {Trainable Fractional {F}ourier Transform},
  journal = {IEEE Signal Processing Letters},
  year    = {2024},
  volume  = {31},
  pages   = {751--755},
  doi     = {10.1109/LSP.2024.3372779}
}
```

Citation metadata for this package itself is in [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE)
