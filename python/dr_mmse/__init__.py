"""DR-MMSE TAC: Distributionally Robust per-step Kalman update.

This package provides:

  * The compiled C++ solver as ``dr_mmse_cpp`` (built via CMake — see
    README.md for build instructions).
  * A thin Python wrapper :class:`DRWrapper` that handles APA tracking
    and warm-starting around a generic predict/update Kalman filter.

The C++ solver itself is filter-agnostic — it consumes ``(APA, C, Sigma_w_hat,
Sigma_v_hat, theta_w, theta_v)`` and returns the worst-case posterior
within Bures-Wasserstein balls. You can call it directly without the
wrapper; the wrapper is only convenience for the common case where you
already have a working KF and want to swap in DR at the measurement
update.

Quick start::

    from dr_mmse import solve_dr_mmse_tac

    # APA = Phi_accum @ P_post_prev @ Phi_accum.T
    # Sigma_w accumulated process noise over the inter-measurement interval
    # (extract from your filter's prior P_pri = APA + Sigma_w)
    res = solve_dr_mmse_tac(APA, C, Sigma_w, R, theta_w=0.5, theta_v=0.1)
    P_post_dr = res.wc_Xpost
    # Compute the worst-case Kalman gain from the worst-case noise:
    # K = P_pri^DR C^T (C P_pri^DR C^T + R^DR)^{-1}
    # See examples/01_minimal_linear.py for the full update.

theta=0 short-circuit returns the standard Kalman posterior in zero
iterations — use that as a parity check against your existing filter.
"""
from .wrapper import DRWrapper, dr_kalman_update  # noqa: F401

# Re-export the C++ symbols at top level for convenience.
try:
    from dr_mmse_cpp import (  # noqa: F401
        solve_dr_mmse_tac,
        solve_dr_mmse_tac_factored,
        posterior_cov,
        kalman_stats,
        oracle_bisection_fast,
        oracle_exact_fast,
        DRMMSEResult,
        DRMMSEFactoredResult,
    )
except ImportError as e:
    raise ImportError(
        "dr_mmse_cpp not found. Build the C++ extension first:\n"
        "    cd <package-root>\n"
        "    ./build.sh\n"
        "or, manually:\n"
        "    cmake -S . -B build -DCMAKE_PREFIX_PATH=$(python -m pybind11 --cmakedir)\n"
        "    cmake --build build -j\n"
        "See README.md for build prerequisites."
    ) from e

__all__ = [
    'DRWrapper', 'dr_kalman_update',
    'solve_dr_mmse_tac', 'solve_dr_mmse_tac_factored',
    'posterior_cov', 'kalman_stats',
    'oracle_bisection_fast', 'oracle_exact_fast',
    'DRMMSEResult', 'DRMMSEFactoredResult',
]
