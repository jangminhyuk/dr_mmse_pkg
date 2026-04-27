#pragma once

#include "types.h"

namespace dr_mmse {

// Bisection oracle (Algorithm 2): approximate maximizer of <D, Sigma> s.t. BW(Sigma, Sigma_hat) <= rho
MatXd oracle_bisection_fast(
    const MatXd& Sigma_hat, double rho,
    const MatXd& Sigma_ref, const MatXd& D, double delta,
    int bisect_max_iters = 30, double bisect_tol = 1e-6);

// Exact oracle: exact maximizer of <D, Sigma> s.t. BW(Sigma, Sigma_hat) <= rho
MatXd oracle_exact_fast(
    const MatXd& Sigma_hat, double rho, const MatXd& D,
    double tol = 1e-8, int max_iters = 60);

} // namespace dr_mmse
