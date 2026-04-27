#include "dr_mmse/kalman_utils.h"

namespace dr_mmse {

MatXd posterior_cov(const MatXd& X_pred, const MatXd& Sigma_v, const MatXd& C_t) {
    // X_post = X_pred - X_pred C^T (C X_pred C^T + Sigma_v)^{-1} C X_pred
    MatXd C_X = C_t * X_pred;        // ny x nx
    MatXd S = C_X * C_t.transpose() + Sigma_v;  // ny x ny

    // K = solve(S, C_X)^T  => K = X_pred C^T S^{-1}
    // solve S * Z = C_X => Z = S^{-1} C_X, then K = Z^T
    Eigen::LDLT<MatXd> ldlt(S);
    if (ldlt.info() != Eigen::Success) {
        // Regularize
        MatXd S_reg = S + 1e-12 * MatXd::Identity(S.rows(), S.cols());
        ldlt.compute(S_reg);
    }
    MatXd K = ldlt.solve(C_X).transpose();  // nx x ny

    MatXd X_post = X_pred - K * C_X;
    return sym(X_post);
}

KalmanStats kalman_stats(const MatXd& X_pred, const MatXd& Sigma_v, const MatXd& C_t) {
    int nx = X_pred.rows();
    int ny = Sigma_v.rows();

    MatXd C_X = C_t * X_pred;        // ny x nx
    MatXd S = C_X * C_t.transpose() + Sigma_v;  // ny x ny

    Eigen::LDLT<MatXd> ldlt(S);
    if (ldlt.info() != Eigen::Success) {
        MatXd S_reg = S + 1e-12 * MatXd::Identity(ny, ny);
        ldlt.compute(S_reg);
    }

    // K = solve(S, C_X)^T
    MatXd K = ldlt.solve(C_X).transpose();  // nx x ny

    // X_post = X_pred - K * C_X
    MatXd X_post = X_pred - K * C_X;
    double trace_post = X_post.trace();

    // Gradients
    // U = S^{-1} C_t  (ny x nx)
    MatXd U = ldlt.solve(C_t);
    // M = C_t^T U  (nx x nx)
    MatXd M = C_t.transpose() * U;

    // A = I - X_pred * M
    MatXd I_nx = MatXd::Identity(nx, nx);
    MatXd A = I_nx - X_pred * M;
    // D_x = A^T A
    MatXd D_x = A.transpose() * A;

    // D_v = S^{-1} (C X_pred^2 C^T) S^{-1}
    MatXd X2 = X_pred * X_pred;
    MatXd T = C_t * X2 * C_t.transpose();  // ny x ny
    MatXd W1 = ldlt.solve(T);              // ny x ny
    MatXd D_v = ldlt.solve(W1.transpose()).transpose();  // ny x ny

    // Check for numerically zero matrices
    if (D_x.norm() < 1e-15) {
        D_x = 1e-12 * I_nx;
    }
    if (D_v.norm() < 1e-15) {
        D_v = 1e-12 * MatXd::Identity(ny, ny);
    }

    return {trace_post, D_x, D_v};
}

double fw_objective_f_min(const MatXd& X_pred, const MatXd& Sigma_v, const MatXd& C_t) {
    MatXd X_post = posterior_cov(X_pred, Sigma_v, C_t);
    return -X_post.trace();
}

} // namespace dr_mmse
