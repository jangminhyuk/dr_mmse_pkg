#include "dr_mmse/dr_mmse_tac.h"
#include "dr_mmse/kalman_utils.h"
#include "dr_mmse/fw_oracle.h"

namespace dr_mmse {

DRMMSEResult solve_dr_mmse_tac(
    const MatXd& APA,
    const MatXd& C,
    const MatXd& Sigma_w_hat,
    const MatXd& Sigma_v_hat,
    double theta_w, double theta_v,
    const std::string& solver,
    const MatXd& warm_Sigma_w,
    const MatXd& warm_Sigma_v,
    double fw_beta_minus1, double fw_tau,
    double fw_zeta, double fw_delta,
    int fw_max_iters, double fw_gap_tol,
    int fw_bisect_max_iters, double fw_bisect_tol,
    int fw_exact_iters, double fw_exact_gap_tol,
    double oracle_exact_tol, int oracle_exact_max_iters)
{
    int nx = APA.rows();
    int ny = C.rows();
    MatXd I_nx = MatXd::Identity(nx, nx);
    MatXd I_ny = MatXd::Identity(ny, ny);

    // Short-circuit: theta=0 => standard Kalman (no DR)
    if (theta_w <= 0 && theta_v <= 0) {
        MatXd Xprior = sym(APA + Sigma_w_hat);
        MatXd Xpost = posterior_cov(Xprior, Sigma_v_hat, C);
        return {Sigma_v_hat, Sigma_w_hat, Xprior, Xpost, true, 0};
    }

    // Determine iteration count and gap tolerance based on solver type
    bool use_exact = (solver == "fw_exact");
    int max_iters = use_exact ? fw_exact_iters : fw_max_iters;
    double gap_tol = use_exact ? fw_exact_gap_tol : fw_gap_tol;

    MatXd Sigma_w_nom = Sigma_w_hat;
    MatXd Sigma_v_nom = Sigma_v_hat;
    MatXd Sigma_w, Sigma_v;

    // --- Safe warm start for Sigma_w ---
    if (theta_w > 0 && warm_Sigma_w.rows() > 0) {
        MatXd Sigma_w_warm = sym(warm_Sigma_w);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_hat_w(Sigma_w_hat);
        double lambda_min_w = eig_hat_w.eigenvalues()(0);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_warm_w(Sigma_w_warm);
        double eig_min_w = eig_warm_w.eigenvalues()(0);
        if (eig_min_w < lambda_min_w) {
            Sigma_w_warm = sym(Sigma_w_warm + (lambda_min_w - eig_min_w) * I_nx);
        }
        double f_nom_w = fw_objective_f_min(APA + Sigma_w_nom, Sigma_v_nom, C);
        double f_warm_w = fw_objective_f_min(APA + Sigma_w_warm, Sigma_v_nom, C);
        Sigma_w = (f_warm_w < f_nom_w) ? Sigma_w_warm : Sigma_w_nom;
    } else {
        Sigma_w = Sigma_w_nom;
    }

    // --- Safe warm start for Sigma_v ---
    if (theta_v > 0 && warm_Sigma_v.rows() > 0) {
        MatXd Sigma_v_warm = sym(warm_Sigma_v);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_hat_v(Sigma_v_hat);
        double lambda_min_v = eig_hat_v.eigenvalues()(0);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_warm_v(Sigma_v_warm);
        double eig_min_v = eig_warm_v.eigenvalues()(0);
        if (eig_min_v < lambda_min_v) {
            Sigma_v_warm = sym(Sigma_v_warm + (lambda_min_v - eig_min_v) * I_ny);
        }
        double f_nom_v = fw_objective_f_min(APA + Sigma_w, Sigma_v_nom, C);
        double f_warm_v = fw_objective_f_min(APA + Sigma_w, Sigma_v_warm, C);
        Sigma_v = (f_warm_v < f_nom_v) ? Sigma_v_warm : Sigma_v_nom;
    } else {
        Sigma_v = Sigma_v_nom;
    }

    double beta_prev = fw_beta_minus1;

    // Initial objective and gradients
    auto stats = kalman_stats(APA + Sigma_w, Sigma_v, C);
    double f_curr = -stats.trace_post;
    MatXd D_w = stats.D_x;  // Chain rule: dX_pred/dSigma_w = I
    MatXd D_v = stats.D_v;

    int iterations = 0;
    for (int t = 0; t < max_iters; ++t) {
        iterations = t + 1;

        // Oracle calls
        MatXd L_w, L_v;
        if (use_exact) {
            L_w = (theta_w > 0) ?
                oracle_exact_fast(Sigma_w_hat, theta_w, D_w,
                                  oracle_exact_tol, oracle_exact_max_iters) :
                Sigma_w_nom;
            L_v = (theta_v > 0) ?
                oracle_exact_fast(Sigma_v_hat, theta_v, D_v,
                                  oracle_exact_tol, oracle_exact_max_iters) :
                Sigma_v_nom;
        } else {
            L_w = (theta_w > 0) ?
                oracle_bisection_fast(Sigma_w_hat, theta_w, Sigma_w, D_w, fw_delta,
                                      fw_bisect_max_iters, fw_bisect_tol) :
                Sigma_w_nom;
            L_v = (theta_v > 0) ?
                oracle_bisection_fast(Sigma_v_hat, theta_v, Sigma_v, D_v, fw_delta,
                                      fw_bisect_max_iters, fw_bisect_tol) :
                Sigma_v_nom;
        }

        MatXd d_w = L_w - Sigma_w;
        MatXd d_v = L_v - Sigma_v;

        double g = inner(d_w, D_w) + inner(d_v, D_v);
        if (g <= gap_tol) break;

        double beta_t = beta_prev / fw_zeta;
        double d_norm2 = fro_norm2(d_w) + fro_norm2(d_v);
        if (d_norm2 <= 0 || g <= 0) break;

        double eta = std::min(1.0, g / (beta_t * d_norm2));

        double f_next = 0;
        for (int bt = 0; bt < 20; ++bt) {
            auto stats_next = kalman_stats(APA + Sigma_w + eta * d_w,
                                           Sigma_v + eta * d_v, C);
            f_next = -stats_next.trace_post;
            double armijo_rhs = f_curr - eta * g + 0.5 * eta * eta * beta_t * d_norm2;

            if (f_next <= armijo_rhs) {
                D_w = stats_next.D_x;
                D_v = stats_next.D_v;
                break;
            }
            beta_t = fw_tau * beta_t;
            eta = std::min(1.0, g / (beta_t * d_norm2));
        }

        Sigma_w = Sigma_w + eta * d_w;
        Sigma_v = Sigma_v + eta * d_v;
        f_curr = f_next;
        beta_prev = beta_t;
    }

    MatXd wc_Sigma_w = sym(Sigma_w);
    MatXd wc_Sigma_v = sym(Sigma_v);

    // Ensure wc_Sigma_v is PD (eigenvalue floor)
    {
        Eigen::SelfAdjointEigenSolver<MatXd> eig(wc_Sigma_v);
        double eig_min = eig.eigenvalues()(0);
        if (eig_min < 1e-14) {
            wc_Sigma_v = sym(wc_Sigma_v + (1e-14 - eig_min) * I_ny);
        }
    }

    MatXd wc_Xprior = sym(APA + wc_Sigma_w);
    MatXd wc_Xpost = posterior_cov(wc_Xprior, wc_Sigma_v, C);

    return {wc_Sigma_v, wc_Sigma_w, wc_Xprior, wc_Xpost, true, iterations};
}

// --- Factored solver helpers ---

// forward_map: sum_k G_k @ Q_w @ G_k^T  (nw x nw -> nx x nx)
static MatXd forward_map(const std::vector<MatXd>& G_list, const MatXd& Q_w, int nx) {
    MatXd result = MatXd::Zero(nx, nx);
    for (const auto& G : G_list) {
        result.noalias() += G * Q_w * G.transpose();
    }
    return result;
}

// adjoint_map: sum_k G_k^T @ D_x @ G_k  (nx x nx -> nw x nw)
static MatXd adjoint_map(const std::vector<MatXd>& G_list, const MatXd& D_x, int nw) {
    MatXd result = MatXd::Zero(nw, nw);
    for (const auto& G : G_list) {
        result.noalias() += G.transpose() * D_x * G;
    }
    return result;
}

// fw objective for factored warm-start comparison
static double fw_objective_f_min_factored(
    const MatXd& APA, const std::vector<MatXd>& G_list,
    const MatXd& Q_w, const MatXd& Sigma_v, const MatXd& C)
{
    int nx = APA.rows();
    MatXd Sigma_w = forward_map(G_list, Q_w, nx);
    MatXd X_pred = APA + Sigma_w;
    return fw_objective_f_min(X_pred, Sigma_v, C);
}

DRMMSEFactoredResult solve_dr_mmse_tac_factored(
    const MatXd& APA,
    const MatXd& C,
    const std::vector<MatXd>& G_list,
    const MatXd& Q_hat,
    const MatXd& Sigma_v_hat,
    double theta_w, double theta_v,
    const std::string& solver,
    const MatXd& warm_Q_w,
    const MatXd& warm_Sigma_v,
    double fw_beta_minus1, double fw_tau,
    double fw_zeta, double fw_delta,
    int fw_max_iters, double fw_gap_tol,
    int fw_bisect_max_iters, double fw_bisect_tol,
    int fw_exact_iters, double fw_exact_gap_tol,
    double oracle_exact_tol, int oracle_exact_max_iters)
{
    int nx = APA.rows();
    int ny = C.rows();
    int nw = Q_hat.rows();
    MatXd I_ny = MatXd::Identity(ny, ny);
    MatXd I_nw = MatXd::Identity(nw, nw);

    // Guard: if G_list is empty, treat as theta_w = 0
    bool effective_theta_w_zero = G_list.empty() || (theta_w <= 0);

    // Short-circuit: theta=0 => standard Kalman (no DR)
    if (effective_theta_w_zero && theta_v <= 0) {
        MatXd Sigma_w_nom = forward_map(G_list, Q_hat, nx);
        MatXd Xprior = sym(APA + Sigma_w_nom);
        MatXd Xpost = posterior_cov(Xprior, Sigma_v_hat, C);
        return {Sigma_v_hat, Q_hat, Sigma_w_nom, Xprior, Xpost, true, 0};
    }

    // Determine iteration count and gap tolerance based on solver type
    bool use_exact = (solver == "fw_exact");
    int max_iters = use_exact ? fw_exact_iters : fw_max_iters;
    double gap_tol_val = use_exact ? fw_exact_gap_tol : fw_gap_tol;

    MatXd Q_w_nom = Q_hat;
    MatXd Sigma_v_nom = Sigma_v_hat;
    MatXd Q_w, Sigma_v;

    // --- Safe warm start for Q_w ---
    if (!effective_theta_w_zero && warm_Q_w.rows() > 0) {
        MatXd Q_w_warm = sym(warm_Q_w);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_hat_q(Q_hat);
        double lambda_min_q = eig_hat_q.eigenvalues()(0);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_warm_q(Q_w_warm);
        double eig_min_q = eig_warm_q.eigenvalues()(0);
        if (eig_min_q < lambda_min_q) {
            Q_w_warm = sym(Q_w_warm + (lambda_min_q - eig_min_q) * I_nw);
        }
        double f_nom_q = fw_objective_f_min_factored(APA, G_list, Q_w_nom, Sigma_v_nom, C);
        double f_warm_q = fw_objective_f_min_factored(APA, G_list, Q_w_warm, Sigma_v_nom, C);
        Q_w = (f_warm_q < f_nom_q) ? Q_w_warm : Q_w_nom;
    } else {
        Q_w = Q_w_nom;
    }

    // --- Safe warm start for Sigma_v ---
    if (theta_v > 0 && warm_Sigma_v.rows() > 0) {
        MatXd Sigma_v_warm = sym(warm_Sigma_v);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_hat_v(Sigma_v_hat);
        double lambda_min_v = eig_hat_v.eigenvalues()(0);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_warm_v(Sigma_v_warm);
        double eig_min_v = eig_warm_v.eigenvalues()(0);
        if (eig_min_v < lambda_min_v) {
            Sigma_v_warm = sym(Sigma_v_warm + (lambda_min_v - eig_min_v) * I_ny);
        }
        double f_nom_v = fw_objective_f_min_factored(APA, G_list, Q_w, Sigma_v_nom, C);
        double f_warm_v = fw_objective_f_min_factored(APA, G_list, Q_w, Sigma_v_warm, C);
        Sigma_v = (f_warm_v < f_nom_v) ? Sigma_v_warm : Sigma_v_nom;
    } else {
        Sigma_v = Sigma_v_nom;
    }

    double beta_prev = fw_beta_minus1;

    // Current Sigma_w from Q_w
    MatXd Sigma_w_curr = forward_map(G_list, Q_w, nx);

    // Initial objective and gradients
    auto stats = kalman_stats(APA + Sigma_w_curr, Sigma_v, C);
    double f_curr = -stats.trace_post;
    // Chain rule: D_q = adjoint_map(G_list, D_x)
    MatXd D_q = effective_theta_w_zero ? MatXd::Zero(nw, nw) : adjoint_map(G_list, stats.D_x, nw);
    MatXd D_v = stats.D_v;

    int iterations = 0;
    for (int t = 0; t < max_iters; ++t) {
        iterations = t + 1;

        // Oracle calls (on nw x nw matrices — always well-conditioned)
        MatXd L_q, L_v;
        if (use_exact) {
            L_q = (!effective_theta_w_zero) ?
                oracle_exact_fast(Q_hat, theta_w, D_q,
                                  oracle_exact_tol, oracle_exact_max_iters) :
                Q_w_nom;
            L_v = (theta_v > 0) ?
                oracle_exact_fast(Sigma_v_hat, theta_v, D_v,
                                  oracle_exact_tol, oracle_exact_max_iters) :
                Sigma_v_nom;
        } else {
            L_q = (!effective_theta_w_zero) ?
                oracle_bisection_fast(Q_hat, theta_w, Q_w, D_q, fw_delta,
                                      fw_bisect_max_iters, fw_bisect_tol) :
                Q_w_nom;
            L_v = (theta_v > 0) ?
                oracle_bisection_fast(Sigma_v_hat, theta_v, Sigma_v, D_v, fw_delta,
                                      fw_bisect_max_iters, fw_bisect_tol) :
                Sigma_v_nom;
        }

        MatXd d_q = L_q - Q_w;
        MatXd d_v = L_v - Sigma_v;

        double g = inner(d_q, D_q) + inner(d_v, D_v);
        if (g <= gap_tol_val) break;

        double beta_t = beta_prev / fw_zeta;
        double d_norm2 = fro_norm2(d_q) + fro_norm2(d_v);
        if (d_norm2 <= 0 || g <= 0) break;

        double eta = std::min(1.0, g / (beta_t * d_norm2));

        double f_next = 0;
        for (int bt = 0; bt < 20; ++bt) {
            MatXd Q_w_next = Q_w + eta * d_q;
            MatXd Sigma_w_next = forward_map(G_list, Q_w_next, nx);
            auto stats_next = kalman_stats(APA + Sigma_w_next,
                                           Sigma_v + eta * d_v, C);
            f_next = -stats_next.trace_post;
            double armijo_rhs = f_curr - eta * g + 0.5 * eta * eta * beta_t * d_norm2;

            if (f_next <= armijo_rhs) {
                D_q = effective_theta_w_zero ? MatXd::Zero(nw, nw) : adjoint_map(G_list, stats_next.D_x, nw);
                D_v = stats_next.D_v;
                break;
            }
            beta_t = fw_tau * beta_t;
            eta = std::min(1.0, g / (beta_t * d_norm2));
        }

        Q_w = Q_w + eta * d_q;
        Sigma_v = Sigma_v + eta * d_v;
        Sigma_w_curr = forward_map(G_list, Q_w, nx);
        f_curr = f_next;
        beta_prev = beta_t;
    }

    MatXd wc_Q_w = sym(Q_w);
    MatXd wc_Sigma_v = sym(Sigma_v);
    MatXd wc_Sigma_w = forward_map(G_list, wc_Q_w, nx);

    // Ensure wc_Sigma_v is PD (eigenvalue floor)
    {
        Eigen::SelfAdjointEigenSolver<MatXd> eig(wc_Sigma_v);
        double eig_min = eig.eigenvalues()(0);
        if (eig_min < 1e-14) {
            wc_Sigma_v = sym(wc_Sigma_v + (1e-14 - eig_min) * I_ny);
        }
    }

    MatXd wc_Xprior = sym(APA + wc_Sigma_w);
    MatXd wc_Xpost = posterior_cov(wc_Xprior, wc_Sigma_v, C);

    return {wc_Sigma_v, wc_Q_w, wc_Sigma_w, wc_Xprior, wc_Xpost, true, iterations};
}

} // namespace dr_mmse
