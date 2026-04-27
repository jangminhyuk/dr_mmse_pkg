"""Latency benchmark for solve_dr_mmse_tac on the deployment platform.

Measures end-to-end wall-clock per call across realistic problem sizes:

  * (nx, ny) = (9, 1)   — small ESKF + scalar range / TDoA measurement
  * (nx, ny) = (15, 3)  — INS error-state + 3D position (PVA-only INS)
  * (nx, ny) = (21, 3)  — full INS error-state (PVA + biases + scale) + 3D position

Reports cold and warm-start latency separately. Run on the actual
deployment hardware for an honest number — laptop numbers are not
representative of an embedded SoC.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from dr_mmse import solve_dr_mmse_tac  # noqa: E402


def make_spd(n, scale=1.0, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) * scale
    return A @ A.T + 0.1 * np.eye(n)


def bench_one(nx, ny, n_iter=200, warm=True):
    rng = np.random.default_rng(nx * 100 + ny)
    APA = make_spd(nx, seed=nx)
    Sigma_w = make_spd(nx, 0.2, seed=nx + 50)
    R = make_spd(ny, 0.3, seed=ny + 80)
    H = rng.standard_normal((ny, nx))

    # Warm-up to amortize libstdc++ + libgomp first-call overhead.
    res = solve_dr_mmse_tac(APA, H, Sigma_w, R, 0.3, 0.05)
    last = res

    times_us = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        if warm:
            res = solve_dr_mmse_tac(
                APA, H, Sigma_w, R, 0.3, 0.05,
                warm_Sigma_w=last.wc_Sigma_w,
                warm_Sigma_v=last.wc_Sigma_v,
            )
        else:
            res = solve_dr_mmse_tac(APA, H, Sigma_w, R, 0.3, 0.05)
        t1 = time.perf_counter_ns()
        times_us.append((t1 - t0) / 1000.0)
        last = res

    arr = np.array(times_us)
    return {
        'nx': nx, 'ny': ny, 'warm': warm,
        'mean_us': float(arr.mean()),
        'p50_us':  float(np.median(arr)),
        'p99_us':  float(np.percentile(arr, 99)),
        'iter_avg': float(res.iterations),
    }


def main():
    print(f"{'config':<14} {'warm':<6} {'mean(us)':>10} {'p50(us)':>10} {'p99(us)':>10} {'fw_iter':>8}")
    print('-' * 64)
    for nx, ny in [(9, 1), (15, 3), (21, 3)]:
        for warm in [False, True]:
            r = bench_one(nx, ny, warm=warm)
            print(f"({r['nx']:>2},{r['ny']:>2}){'':<8} "
                  f"{str(r['warm']):<6} "
                  f"{r['mean_us']:>10.1f} "
                  f"{r['p50_us']:>10.1f} "
                  f"{r['p99_us']:>10.1f} "
                  f"{r['iter_avg']:>8.1f}")


if __name__ == '__main__':
    main()
