#include "dr_mmse/fw_oracle.h"

namespace dr_mmse {

MatXd oracle_bisection_fast(
    const MatXd& Sigma_hat_in, double rho,
    const MatXd& Sigma_ref_in, const MatXd& D_in, double delta,
    int bisect_max_iters, double bisect_tol)
{
    // Symmetrize inputs
    MatXd Sigma_hat = sym(Sigma_hat_in);
    MatXd Sigma_ref = sym(Sigma_ref_in);
    MatXd D = sym(D_in);
    int d = D.rows();

    if (rho <= 0) return Sigma_hat;

    // Eigendecompose D (ascending order)
    Eigen::SelfAdjointEigenSolver<MatXd> eig(D);
    VecXd lam = eig.eigenvalues();    // ascending
    MatXd Q = eig.eigenvectors();
    double lam_max = lam(d - 1);

    if (lam_max <= 0) return Sigma_hat;

    // Compute B = Q^T Sigma_hat Q and its diagonal
    MatXd B = Q.transpose() * Sigma_hat * Q;
    VecXd b = B.diagonal();

    // Precompute <Sigma_ref, D>
    double inner_ref = inner(Sigma_ref, D);

    // Bracket gamma
    VecXd v1 = Q.col(d - 1);
    double v1Sigv1 = v1.dot(Sigma_hat * v1);
    double trSig = Sigma_hat.trace();

    double gamma_lb = lam_max * (1.0 + std::sqrt(std::max(v1Sigv1, 0.0)) / rho);
    double gamma_ub = lam_max * (1.0 + std::sqrt(std::max(trSig, 0.0)) / rho);
    gamma_lb = std::max(gamma_lb, lam_max + 1e-12);

    double gamma_star = gamma_ub;

    for (int iter = 0; iter < bisect_max_iters; ++iter) {
        double gamma_mid = 0.5 * (gamma_lb + gamma_ub);

        // O(d) computation
        VecXd denom = (gamma_mid * VecXd::Ones(d) - lam).cwiseMax(1e-14);
        VecXd a = lam.array() / denom.array();
        VecXd ratio = gamma_mid * VecXd::Ones(d).array() / denom.array();

        double sum1 = (b.array() * a.array() * a.array()).sum();
        double sum2 = (b.array() * a.array()).sum();
        double sum3 = (lam.array() * b.array() * ratio.array() * ratio.array()).sum();

        double dphi = rho * rho - sum1;
        double phi = gamma_mid * (rho * rho + sum2) - inner_ref;
        double inner_stopping = sum3 - inner_ref;

        // Early exit
        if (dphi > 0 && phi > 0 && inner_stopping >= delta * phi) {
            gamma_star = gamma_mid;
            break;
        }

        // Bracket update
        if (dphi > 0) {
            gamma_ub = gamma_mid;
        } else {
            gamma_lb = gamma_mid;
        }

        if (std::abs(gamma_ub - gamma_lb) <= bisect_tol) {
            gamma_star = gamma_mid;
            break;
        }

        // If last iteration, use upper bound
        if (iter == bisect_max_iters - 1) {
            gamma_star = gamma_ub;
        }
    }

    // Reconstruct L(gamma_star) once
    VecXd denom = (gamma_star * VecXd::Ones(d) - lam).cwiseMax(1e-14);
    VecXd inv_vec = denom.cwiseInverse();

    // M = diag(inv) B diag(inv) efficiently
    MatXd M = (inv_vec.asDiagonal() * B) * inv_vec.asDiagonal();
    MatXd L = (gamma_star * gamma_star) * (Q * M * Q.transpose());

    return sym(L);
}

MatXd oracle_exact_fast(
    const MatXd& Sigma_hat_in, double rho, const MatXd& D_in,
    double tol, int max_iters)
{
    MatXd D = sym(D_in);
    MatXd Sigma_hat = sym(Sigma_hat_in);
    int d = D.rows();

    if (rho <= 0) return Sigma_hat;
    if (D.norm() < 1e-15) return Sigma_hat;

    // Eigendecompose D
    Eigen::SelfAdjointEigenSolver<MatXd> eig(D);
    VecXd eigvals = eig.eigenvalues();   // ascending
    MatXd Q = eig.eigenvectors();
    double lambda1 = eigvals(d - 1);

    if (lambda1 <= 0) return Sigma_hat;

    // B = Q^T Sigma_hat Q
    MatXd B_mat = Q.transpose() * Sigma_hat * Q;
    VecXd b = B_mat.diagonal();

    // Bracket gamma
    VecXd v1 = Q.col(d - 1);
    double v1Sigv1 = v1.dot(Sigma_hat * v1);
    double trSig = Sigma_hat.trace();

    double gamma_lo = lambda1 * (1.0 + std::sqrt(std::max(v1Sigv1, 0.0)) / rho);
    double gamma_hi = lambda1 * (1.0 + std::sqrt(std::max(trSig, 0.0)) / rho);
    gamma_lo = std::max(gamma_lo, lambda1 + 1e-12);

    // Bisection: find gamma s.t. h(gamma) = 0
    // h(gamma) = rho^2 - sum_i b[i] * (eigvals[i] / (gamma - eigvals[i]))^2
    for (int iter = 0; iter < max_iters; ++iter) {
        double gamma = 0.5 * (gamma_lo + gamma_hi);

        VecXd denom = (gamma * VecXd::Ones(d) - eigvals).cwiseMax(1e-14);
        VecXd ratio = eigvals.array() / denom.array();
        double h_val = rho * rho - (b.array() * ratio.array() * ratio.array()).sum();

        if (h_val < 0) {
            gamma_lo = gamma;
        } else {
            gamma_hi = gamma;
        }

        if (std::abs(gamma_hi - gamma_lo) <= tol) break;
    }

    double gamma_star = 0.5 * (gamma_lo + gamma_hi);

    // Reconstruct L
    VecXd denom = (gamma_star * VecXd::Ones(d) - eigvals).cwiseMax(1e-14);
    VecXd inv_vec = denom.cwiseInverse();

    MatXd M = (inv_vec.asDiagonal() * B_mat) * inv_vec.asDiagonal();
    MatXd L = (gamma_star * gamma_star) * (Q * M * Q.transpose());

    return sym(L);
}

} // namespace dr_mmse
