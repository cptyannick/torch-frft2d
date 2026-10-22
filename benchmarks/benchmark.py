"""
Benchmark frft2d_multi against looped torch-frft and a plain fft2.

    uv run python benchmarks/benchmark.py [--B 2] [--A 16] [--size 256]

Uses CUDA automatically if available.
"""

import argparse
import math
import time

import torch
from torch_frft.frft_module import frft

from torch_frft2d import frft2d_multi


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--A", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)

    def timeit(fn, reps=args.reps):
        fn()
        sync()
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            sync()
            ts.append(time.perf_counter() - t)
        return sorted(ts)[len(ts) // 2]

    B, A, M, N = args.B, args.A, args.size, args.size
    f32 = torch.randn(B, M, N, dtype=torch.complex64, device=dev)
    ang = torch.rand(B, A, device=dev) * 2 * math.pi
    n_ref = min(A, 4)
    with torch.no_grad():
        t_ours = timeit(lambda: frft2d_multi(f32, ang))

        def ref():  # torch-frft takes a scalar order only -> loop; extrapolate to A
            for bi in range(B):
                for ai in range(n_ref):
                    o = ang[bi, ai] * 2 / math.pi
                    frft(frft(f32[bi], o, dim=-1), o, dim=-2)

        t_ref = timeit(ref, reps=1) * A / n_ref
        t_fft = timeit(lambda: torch.fft.fft2(f32[:, None].expand(B, A, M, N)))

    where = dev.type if dev.type == "cuda" else f"cpu, {torch.get_num_threads()} threads"
    print(f"benchmark (B, A, M, N) = {(B, A, M, N)}, complex64, {where}")
    print(f"  frft2d_multi           : {t_ours * 1e3:8.1f} ms")
    print(f"  torch-frft, looped     : {t_ref * 1e3:8.1f} ms  (extrapolated from {n_ref} angles)")
    print(f"  plain fft2 on B*A imgs : {t_fft * 1e3:8.1f} ms  (lower bound for reference)")


if __name__ == "__main__":
    main()
