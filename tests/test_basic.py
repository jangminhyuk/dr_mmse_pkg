"""Smoke tests for the DR-MMSE package.

This file covers the contract that anyone porting to a new platform
should run before trusting the build:

  * theta = 0 returns the standard Kalman posterior in 0 iterations.
  * Outputs are PSD.
  * wc_Sigma_v >= R_hat in the BW sense (adversary inflates measurement noise).
  * Warm-start does not increase iteration count.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from dr_mmse import (  # noqa: E402
    solve_dr_mmse_tac, posterior_cov, DRWrapper,
)


def make_spd(n, scale=1.0, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) * scale
    return A @ A.T + 0.1 * np.eye(n)


def test_theta_zero_parity():
    """theta=0 must reproduce the standard Kalman posterior exactly."""
    for nx, ny in [(2, 1), (5, 2), (9, 1), (15, 3)]:
        APA = make_spd(nx, seed=nx)
        Sigma_w = make_spd(nx, scale=0.3, seed=nx + 100)
        R = make_spd(ny, scale=0.5, seed=ny + 200)
        H = np.random.default_rng(nx * 7 + ny).standard_normal((ny, nx))

        res = solve_dr_mmse_tac(APA, H, Sigma_w, R, 0.0, 0.0)
        assert res.iterations == 0, f"iterations={res.iterations} for theta=0"

        P_pri = APA + Sigma_w
        P_post_ref = posterior_cov(P_pri, R, H)
        diff = float(np.max(np.abs(res.wc_Xpost - P_post_ref)))
        assert diff < 1e-12, f"parity mismatch {diff:.2e} for nx={nx}, ny={ny}"
    print("[PASS] theta=0 parity")


def test_psd_outputs():
    nx, ny = 9, 2
    rng = np.random.default_rng(0)
    for trial in range(10):
        APA = make_spd(nx, seed=trial)
        Sigma_w = make_spd(nx, 0.2, seed=trial + 50)
        R = make_spd(ny, 0.3, seed=trial + 80)
        H = rng.standard_normal((ny, nx))
        theta_w = float(rng.uniform(0.01, 0.5))
        theta_v = float(rng.uniform(0.01, 0.5))

        res = solve_dr_mmse_tac(APA, H, Sigma_w, R, theta_w, theta_v)
        assert res.success
        for name, M in [('wc_Sigma_w', res.wc_Sigma_w),
                        ('wc_Sigma_v', res.wc_Sigma_v),
                        ('wc_Xprior',  res.wc_Xprior),
                        ('wc_Xpost',   res.wc_Xpost)]:
            eig_min = float(np.linalg.eigvalsh(M).min())
            assert eig_min > -1e-8, f"{name} not PSD: eig_min={eig_min:.2e}"
    print("[PASS] PSD outputs")


def test_adversary_inflates_meas_noise():
    """trace(wc_Sigma_v) should be >= trace(R_hat) (BW ball is around R_hat)."""
    nx, ny = 9, 1
    APA = make_spd(nx, seed=10)
    Sigma_w = make_spd(nx, 0.2, seed=11)
    R = np.array([[0.05]])
    H = np.random.default_rng(12).standard_normal((ny, nx))

    res = solve_dr_mmse_tac(APA, H, Sigma_w, R, 0.0, 0.5)
    assert res.wc_Sigma_v[0, 0] >= R[0, 0] - 1e-10, \
        f"adversary did not inflate R: {res.wc_Sigma_v[0,0]} vs {R[0,0]}"
    print("[PASS] adversary inflates measurement noise")


def test_warm_start_helps():
    nx, ny = 9, 1
    APA = make_spd(nx, seed=20)
    Sigma_w = make_spd(nx, 0.2, seed=21)
    R = np.array([[0.05]])
    H = np.random.default_rng(22).standard_normal((ny, nx))

    cold = solve_dr_mmse_tac(APA, H, Sigma_w, R, 0.1, 0.1)
    warm = solve_dr_mmse_tac(APA, H, Sigma_w, R, 0.1, 0.1,
                              warm_Sigma_w=cold.wc_Sigma_w,
                              warm_Sigma_v=cold.wc_Sigma_v)
    assert warm.iterations <= cold.iterations
    diff = float(np.max(np.abs(warm.wc_Xpost - cold.wc_Xpost)))
    assert diff < 1e-6, f"warm/cold disagree: {diff:.2e}"
    print(f"[PASS] warm-start ({warm.iterations} iters vs cold {cold.iterations})")


def test_dr_wrapper_apa_tracking():
    """End-to-end: DRWrapper APA tracking matches direct solve_dr_mmse_tac."""
    nx, ny = 6, 1
    rng = np.random.default_rng(30)

    # Two propagation steps then a measurement.
    Phi1 = np.eye(nx) + 0.01 * rng.standard_normal((nx, nx))
    Phi2 = np.eye(nx) + 0.01 * rng.standard_normal((nx, nx))
    Q1 = make_spd(nx, 0.05, seed=31)
    Q2 = make_spd(nx, 0.05, seed=32)
    P0 = 0.1 * np.eye(nx)

    P1 = Phi1 @ P0 @ Phi1.T + Q1
    P2 = Phi2 @ P1 @ Phi2.T + Q2
    Phi_accum = Phi2 @ Phi1
    APA_ref = Phi_accum @ P0 @ Phi_accum.T
    Sigma_w_ref = P2 - APA_ref

    H = rng.standard_normal((ny, nx))
    R = np.array([[0.05]])
    z = np.array([0.3])

    # Direct call
    direct = solve_dr_mmse_tac(APA_ref, H, Sigma_w_ref, R, 0.2, 0.05)

    # Via wrapper
    dr = DRWrapper(nx=nx, theta_w=0.2, theta_v=0.05)
    dr.reset(P0)
    dr.predict(Phi1)
    dr.predict(Phi2)
    x_pri = np.zeros(nx)
    h_x = H @ x_pri
    _, P_post_w, res_w = dr.update(x_pri, P2, z, h_x, H, R)

    diff = float(np.max(np.abs(P_post_w - direct.wc_Xpost)))
    assert diff < 1e-8, f"wrapper vs direct: {diff:.2e}"
    print(f"[PASS] DRWrapper APA tracking (diff={diff:.2e})")


if __name__ == '__main__':
    test_theta_zero_parity()
    test_psd_outputs()
    test_adversary_inflates_meas_noise()
    test_warm_start_helps()
    test_dr_wrapper_apa_tracking()
    print("\nAll smoke tests passed.")
