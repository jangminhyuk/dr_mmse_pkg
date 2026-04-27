"""Thin Python wrapper around solve_dr_mmse_tac.

The C++ solver is filter-agnostic but it requires inputs in a specific
shape: APA = Phi_accum @ P_post_prev @ Phi_accum^T (deterministic prior
propagation since the last measurement) and Sigma_w (accumulated process
noise since the last measurement). Most existing Kalman filters do not
expose those quantities directly — they only give you the prior P_pri
already collapsed.

This module provides two helpers:

  * :func:`dr_kalman_update` — one-shot stateless DR update. Use when
    you already have ``(P_pri, APA, Sigma_w)`` from your own filter
    bookkeeping.
  * :class:`DRWrapper` — accumulates APA across multiple propagation
    steps, then triggers a DR update on each measurement. Use when your
    filter only exposes a per-step state-transition matrix Phi_k and you
    want the wrapper to track APA for you.

Neither is required — you can call ``solve_dr_mmse_tac`` directly from
your filter code if APA and Sigma_w are easy to extract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from dr_mmse_cpp import solve_dr_mmse_tac, DRMMSEResult


def dr_kalman_update(
    x_pri: np.ndarray,
    P_pri: np.ndarray,
    APA: np.ndarray,
    Sigma_w_hat: np.ndarray,
    z: np.ndarray,
    h_x: np.ndarray,
    H: np.ndarray,
    R_hat: np.ndarray,
    theta_w: float,
    theta_v: float,
    *,
    solver: str = 'fw_exact',
    warm_Sigma_w: Optional[np.ndarray] = None,
    warm_Sigma_v: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, DRMMSEResult]:
    """One-shot DR-MMSE measurement update.

    Computes the worst-case posterior within Bures-Wasserstein balls of
    radius ``(theta_w, theta_v)`` around the nominal noise, then applies
    the Kalman gain implied by the worst-case noise.

    Parameters
    ----------
    x_pri : (nx,)        Prior state mean.
    P_pri : (nx, nx)     Prior covariance (= APA + Sigma_w_hat).
    APA   : (nx, nx)     Phi_accum @ P_post_prev @ Phi_accum^T.
    Sigma_w_hat : (nx, nx)
                         Accumulated nominal process noise since the last
                         measurement (= P_pri - APA).
    z, h_x, H : ((ny,), (ny,), (ny, nx))
                         Measurement, predicted measurement, and Jacobian.
    R_hat : (ny, ny)     Nominal measurement covariance.
    theta_w, theta_v : floats >= 0
                         Wasserstein-ball radii. (0, 0) reduces to the
                         standard Kalman update.

    Returns
    -------
    x_post : (nx,)       Posterior state mean.
    P_post : (nx, nx)    Worst-case posterior covariance.
    result : DRMMSEResult
                         Solver diagnostics, including ``wc_Sigma_w``
                         and ``wc_Sigma_v`` for warm-starting.
    """
    res = solve_dr_mmse_tac(
        APA=np.asarray(APA, dtype=np.float64),
        C=np.asarray(H, dtype=np.float64),
        Sigma_w_hat=np.asarray(Sigma_w_hat, dtype=np.float64),
        Sigma_v_hat=np.asarray(R_hat, dtype=np.float64),
        theta_w=float(theta_w),
        theta_v=float(theta_v),
        solver=solver,
        warm_Sigma_w=warm_Sigma_w,
        warm_Sigma_v=warm_Sigma_v,
    )

    # Worst-case prior. Note this is APA + wc_Sigma_w, not the original
    # P_pri — the adversary inflates the noise.
    P_pri_dr = res.wc_Xprior
    R_dr = res.wc_Sigma_v

    # Standard Kalman update with the worst-case (P_pri, R)
    S = H @ P_pri_dr @ H.T + R_dr
    K = np.linalg.solve(S, H @ P_pri_dr).T  # nx x ny
    innov = np.asarray(z, dtype=np.float64) - np.asarray(h_x, dtype=np.float64)
    x_post = np.asarray(x_pri, dtype=np.float64) + K @ innov
    P_post = res.wc_Xpost

    return x_post, P_post, res


@dataclass
class DRWrapper:
    """APA-tracking wrapper: feed each propagation step's Phi_k, then call
    update() at every measurement.

    Usage::

        dr = DRWrapper(nx=9, theta_w=0.5, theta_v=0.1)
        for k in range(N):
            # 1) Your filter does its predict step.
            #    Tell us the per-step Phi_k it used.
            dr.predict(Phi_k)

            if measurement_at_step_k:
                # 2) Your filter exposes (x_pri, P_pri, h_x, H, R_hat).
                #    DRWrapper extracts Sigma_w as P_pri - APA, then
                #    calls solve_dr_mmse_tac and applies the gain.
                x_post, P_post, _ = dr.update(
                    x_pri, P_pri, z, h_x, H, R_hat,
                )
                # Push the post-update covariance back into your filter.

    APA tracking starts from ``P_post_prev = P0`` at construction time
    and is reset to the post-update covariance after every ``update()``.
    """
    nx: int
    theta_w: float
    theta_v: float
    solver: str = 'fw_exact'

    _Phi_accum: np.ndarray = field(init=False)
    _P_post_last: Optional[np.ndarray] = field(init=False, default=None)
    _warm_Sigma_w: Optional[np.ndarray] = field(init=False, default=None)
    _warm_Sigma_v: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self):
        self._Phi_accum = np.eye(self.nx)

    def reset(self, P_post_init: np.ndarray) -> None:
        """Reset APA tracking to a fresh post-update covariance.

        Call once at filter init with your initial covariance, and any
        time you re-initialize the filter (e.g. on a GT reset)."""
        self._Phi_accum = np.eye(self.nx)
        self._P_post_last = np.asarray(P_post_init, dtype=np.float64).copy()
        self._warm_Sigma_w = None
        self._warm_Sigma_v = None

    def predict(self, Phi: np.ndarray) -> None:
        """Accumulate the per-step state-transition matrix.

        Pass the same Phi your filter applied during its predict step.
        For an EKF, this is the linearized state Jacobian (e.g. the
        F-matrix in error-state ESKF parlance)."""
        Phi = np.asarray(Phi, dtype=np.float64)
        self._Phi_accum = Phi @ self._Phi_accum

    def update(
        self,
        x_pri: np.ndarray,
        P_pri: np.ndarray,
        z: np.ndarray,
        h_x: np.ndarray,
        H: np.ndarray,
        R_hat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, DRMMSEResult]:
        """Apply the DR measurement update.

        Returns ``(x_post, P_post, solver_result)``. The wrapper resets
        APA tracking to ``P_post`` so the next ``predict()`` accumulation
        starts from the post-update state.
        """
        if self._P_post_last is None:
            raise RuntimeError(
                "DRWrapper.reset(P0) must be called before the first update()."
            )

        APA = self._Phi_accum @ self._P_post_last @ self._Phi_accum.T
        APA = 0.5 * (APA + APA.T)
        Sigma_w_hat = np.asarray(P_pri, dtype=np.float64) - APA
        # Numerical safety: enforce PSD by symmetrizing + tiny floor.
        Sigma_w_hat = 0.5 * (Sigma_w_hat + Sigma_w_hat.T)
        eig_min = np.linalg.eigvalsh(Sigma_w_hat).min()
        if eig_min < 1e-12:
            Sigma_w_hat = Sigma_w_hat + (1e-12 - eig_min) * np.eye(self.nx)

        x_post, P_post, res = dr_kalman_update(
            x_pri=x_pri, P_pri=P_pri,
            APA=APA, Sigma_w_hat=Sigma_w_hat,
            z=z, h_x=h_x, H=H, R_hat=R_hat,
            theta_w=self.theta_w, theta_v=self.theta_v,
            solver=self.solver,
            warm_Sigma_w=self._warm_Sigma_w,
            warm_Sigma_v=self._warm_Sigma_v,
        )

        # Cache for warm-start on the next attempt.
        self._warm_Sigma_w = res.wc_Sigma_w
        self._warm_Sigma_v = res.wc_Sigma_v

        # Reset APA tracking to start accumulating from the new post-update.
        self._Phi_accum = np.eye(self.nx)
        self._P_post_last = P_post.copy()

        return x_post, P_post, res
