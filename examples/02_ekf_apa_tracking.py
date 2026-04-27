"""EKF integration via the DRWrapper helper (APA tracking).

Demonstrates the typical embedded-robotics pattern:

  * 9-state error-state filter (position 3 + velocity 3 + attitude 3).
  * High-rate IMU propagation between sparse range-like measurements.
  * The wrapper accumulates Phi_k across propagation steps, then runs
    the DR-MMSE update at each measurement.

This file is meant to be copy-pasted into your own filter and adapted.
The propagate / measurement-jacobian functions here are stubs — replace
them with your real dynamics.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from dr_mmse import DRWrapper  # noqa: E402


# --------- Stub filter dynamics (replace with your real EKF) -----------

def imu_step_jacobian(dt: float) -> np.ndarray:
    """Phi_k for a constant-acceleration error-state on (p, v, theta)."""
    nx = 9
    Phi = np.eye(nx)
    Phi[0:3, 3:6] = dt * np.eye(3)        # dp/dv
    Phi[3:6, 6:9] = dt * np.eye(3)        # dv/dtheta (small-angle coupling)
    return Phi


def imu_process_noise(dt: float) -> np.ndarray:
    """Discrete-time process noise. Replace with your filter's Q_d."""
    nx = 9
    Q_c = np.diag([0.0]*3 + [0.01]*3 + [1e-4]*3)  # accel + gyro spectra
    return Q_c * dt  # crude Euler discretization


def measurement_model(x: np.ndarray) -> tuple:
    """Stub measurement: observe (x, y) position with additive noise."""
    H = np.zeros((2, 9))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    h_x = H @ x
    R = np.diag([0.04, 0.04])  # 20cm std per axis
    return h_x, H, R


# --------- Filter loop with DR ----------------------------------------

def run_demo():
    nx = 9
    rng = np.random.default_rng(1)

    # Initial state and covariance
    x = np.zeros(nx)
    P = 0.01 * np.eye(nx)

    # The DR wrapper. theta_w / theta_v are dataset-tuned hyperparameters;
    # see README §"Tuning theta" for guidance.
    dr = DRWrapper(nx=nx, theta_w=0.3, theta_v=0.05)
    dr.reset(P)

    dt_imu = 0.005   # 200 Hz IMU
    measurement_period = 1.0  # 1 Hz measurement

    n_steps = int(10.0 / dt_imu)
    next_meas_t = measurement_period
    t = 0.0

    n_meas = 0
    iterations_total = 0
    for k in range(n_steps):
        # 1) Predict step (your filter)
        Phi = imu_step_jacobian(dt_imu)
        Q_d = imu_process_noise(dt_imu)
        x = Phi @ x   # nominal mean propagation (no IMU input here, stub)
        P = Phi @ P @ Phi.T + Q_d
        P = 0.5 * (P + P.T)

        # 2) Tell the wrapper what Phi the filter just applied.
        dr.predict(Phi)

        t += dt_imu

        # 3) Measurement update?
        if t >= next_meas_t - 1e-9:
            h_x, H, R = measurement_model(x)
            # Synthetic measurement: ground-truth position + noise.
            z = h_x + rng.standard_normal(2) * np.sqrt(np.diag(R))

            x, P, res = dr.update(
                x_pri=x, P_pri=P, z=z, h_x=h_x, H=H, R_hat=R,
            )

            n_meas += 1
            iterations_total += res.iterations
            next_meas_t += measurement_period

            print(f"  t={t:.2f}s  iters={res.iterations}  "
                  f"trP={np.trace(P):.4f}  "
                  f"|x_pos|={np.linalg.norm(x[0:3]):.4f}")

    print(f"\nDone. {n_meas} measurements processed, "
          f"{iterations_total / max(n_meas, 1):.2f} avg FW iters.")


if __name__ == '__main__':
    run_demo()
