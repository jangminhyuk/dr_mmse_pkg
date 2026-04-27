"""Minimal linear-Gaussian example: 2-state random walk with scalar measurement.

Compares three updates side-by-side at a single time step:

  1. Vanilla Kalman update (theta_w = theta_v = 0).
  2. DR update with theta_w > 0 (process-noise robust).
  3. DR update with theta_w = theta_v > 0 (joint robust).

The point of this example is to show how to call solve_dr_mmse_tac
directly on raw (APA, C, Sigma_w_hat, R, theta_w, theta_v) — no APA
tracking helper, no filter object. This is the smallest possible
integration surface for porting the DR step to your own filter.

Run from the repo root after building the C++ module::

    python examples/01_minimal_linear.py
"""
import os
import sys

import numpy as np

# Make the python/ package importable from the source tree without install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from dr_mmse import solve_dr_mmse_tac  # noqa: E402


def kalman_update(P_pri, H, R):
    """Standard Kalman posterior covariance (no DR)."""
    S = H @ P_pri @ H.T + R
    K = np.linalg.solve(S, H @ P_pri).T
    P_post = P_pri - K @ H @ P_pri
    return 0.5 * (P_post + P_post.T)


def main():
    rng = np.random.default_rng(0)
    nx, ny = 2, 1

    # Set up a single timestep. In a real filter, APA and Sigma_w come
    # from your propagation; here we synthesize plausible values.
    Phi = np.array([[1.0, 0.1], [0.0, 1.0]])  # constant-velocity step
    P_post_prev = 0.5 * np.eye(nx)             # last post-update cov
    Sigma_w_hat = np.diag([0.05, 0.10])        # nominal process noise

    APA = Phi @ P_post_prev @ Phi.T
    P_pri = APA + Sigma_w_hat                  # prior the filter would have

    H = np.array([[1.0, 0.0]])                 # measure position only
    R = np.array([[0.20]])                     # nominal meas variance

    print(f"Prior P_pri trace = {np.trace(P_pri):.4f}")

    # --- 1) Vanilla Kalman (parity check) -----------------------------
    res0 = solve_dr_mmse_tac(APA, H, Sigma_w_hat, R, theta_w=0.0, theta_v=0.0)
    P_post_kf_solver = res0.wc_Xpost
    P_post_kf_ref    = kalman_update(P_pri, H, R)
    err = np.max(np.abs(P_post_kf_solver - P_post_kf_ref))
    print(f"\n[1] Vanilla Kalman (theta=0):")
    print(f"    iterations          = {res0.iterations}  (must be 0)")
    print(f"    P_post trace        = {np.trace(P_post_kf_solver):.4f}")
    print(f"    parity vs analytic  = {err:.2e}  (must be < 1e-12)")

    # --- 2) DR over process noise only --------------------------------
    res1 = solve_dr_mmse_tac(
        APA, H, Sigma_w_hat, R,
        theta_w=0.5, theta_v=0.0, solver='fw_exact',
    )
    print(f"\n[2] DR theta_w=0.5, theta_v=0:")
    print(f"    iterations          = {res1.iterations}")
    print(f"    wc_Sigma_w trace    = {np.trace(res1.wc_Sigma_w):.4f}  "
          f"(nominal {np.trace(Sigma_w_hat):.4f})")
    print(f"    wc_Sigma_v          = {res1.wc_Sigma_v[0,0]:.4f}     "
          f"(nominal {R[0,0]:.4f}, stays nominal)")
    print(f"    P_post trace        = {np.trace(res1.wc_Xpost):.4f}  "
          f"(>= vanilla {np.trace(P_post_kf_solver):.4f})")

    # --- 3) Joint DR --------------------------------------------------
    res2 = solve_dr_mmse_tac(
        APA, H, Sigma_w_hat, R,
        theta_w=0.5, theta_v=0.1, solver='fw_exact',
    )
    print(f"\n[3] DR theta_w=0.5, theta_v=0.1:")
    print(f"    iterations          = {res2.iterations}")
    print(f"    wc_Sigma_w trace    = {np.trace(res2.wc_Sigma_w):.4f}")
    print(f"    wc_Sigma_v          = {res2.wc_Sigma_v[0,0]:.4f}")
    print(f"    P_post trace        = {np.trace(res2.wc_Xpost):.4f}")

    # Apply the worst-case Kalman gain to a synthetic measurement
    z_true = np.array([1.5])
    P_pri_dr = res2.wc_Xprior
    R_dr = res2.wc_Sigma_v
    S_dr = H @ P_pri_dr @ H.T + R_dr
    K_dr = np.linalg.solve(S_dr, H @ P_pri_dr).T
    x_pri = np.array([1.0, 0.0])
    h_x = H @ x_pri
    x_post = x_pri + K_dr @ (z_true - h_x)
    print(f"\n[3] Posterior state under DR gain:")
    print(f"    x_pri  = {x_pri}")
    print(f"    z      = {z_true}, innov = {z_true - h_x}")
    print(f"    K (DR) = {K_dr.flatten()}")
    print(f"    x_post = {x_post}")


if __name__ == '__main__':
    main()
