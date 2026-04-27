#pragma once

#include "types.h"
#include <string>
#include <vector>

namespace dr_mmse {

struct DRMMSEResult {
    MatXd wc_Sigma_v;   // ny x ny
    MatXd wc_Sigma_w;   // nx x nx
    MatXd wc_Xprior;    // nx x nx (= APA + wc_Sigma_w)
    MatXd wc_Xpost;     // nx x nx
    bool success;
    int iterations;
};

struct DRMMSEFactoredResult {
    MatXd wc_Sigma_v;   // ny x ny
    MatXd wc_Q_w;       // nw x nw (decision variable in reduced space)
    MatXd wc_Sigma_w;   // nx x nx (= forward_map(G_list, wc_Q_w))
    MatXd wc_Xprior;    // nx x nx (= APA + wc_Sigma_w)
    MatXd wc_Xpost;     // nx x nx
    bool success;
    int iterations;
};

// Standalone DR-MMSE TAC solver.
// Optimizes over (Sigma_w, Sigma_v) within Bures-Wasserstein balls
// around (Sigma_w_hat, Sigma_v_hat) to maximize trace(posterior covariance).
//
// When theta_w <= 0 and theta_v <= 0, returns standard Kalman update (no DR).
DRMMSEResult solve_dr_mmse_tac(
    const MatXd& APA,            // Phi @ P_post @ Phi^T  (nx x nx)
    const MatXd& C,              // measurement Jacobian   (ny x nx)
    const MatXd& Sigma_w_hat,    // nominal process noise  (nx x nx, PSD)
    const MatXd& Sigma_v_hat,    // nominal meas noise     (ny x ny, PD)
    double theta_w, double theta_v,
    const std::string& solver = "fw_exact",
    const MatXd& warm_Sigma_w = MatXd(),  // empty = no warm start
    const MatXd& warm_Sigma_v = MatXd(),
    // FW parameters
    double fw_beta_minus1 = 1.0, double fw_tau = 2.0,
    double fw_zeta = 2.0, double fw_delta = 0.05,
    int fw_max_iters = 50, double fw_gap_tol = 1e-4,
    int fw_bisect_max_iters = 30, double fw_bisect_tol = 1e-6,
    int fw_exact_iters = 8, double fw_exact_gap_tol = 1e-6,
    double oracle_exact_tol = 1e-8, int oracle_exact_max_iters = 60);

// Factored DR-MMSE TAC solver.
// Optimizes over Q_w (nw x nw, full-rank) instead of Sigma_w (nx x nx, rank-deficient).
// Process noise is factored as sum_k G_k Q_w G_k^T where G_k are nx x nw matrices.
// The Wasserstein ball acts on the nw x nw Q_w space.
DRMMSEFactoredResult solve_dr_mmse_tac_factored(
    const MatXd& APA,                        // Phi @ P_post @ Phi^T  (nx x nx)
    const MatXd& C,                          // measurement Jacobian   (ny x nx)
    const std::vector<MatXd>& G_list,        // list of G_k matrices   (each nx x nw)
    const MatXd& Q_hat,                      // nominal noise in reduced space (nw x nw, PD)
    const MatXd& Sigma_v_hat,                // nominal meas noise     (ny x ny, PD)
    double theta_w, double theta_v,
    const std::string& solver = "fw_exact",
    const MatXd& warm_Q_w = MatXd(),         // empty = no warm start
    const MatXd& warm_Sigma_v = MatXd(),
    // FW parameters
    double fw_beta_minus1 = 1.0, double fw_tau = 2.0,
    double fw_zeta = 2.0, double fw_delta = 0.05,
    int fw_max_iters = 50, double fw_gap_tol = 1e-4,
    int fw_bisect_max_iters = 30, double fw_bisect_tol = 1e-6,
    int fw_exact_iters = 8, double fw_exact_gap_tol = 1e-6,
    double oracle_exact_tol = 1e-8, int oracle_exact_max_iters = 60);

} // namespace dr_mmse
