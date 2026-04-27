#pragma once

#include "types.h"

namespace dr_mmse {

struct KalmanStats {
    double trace_post;
    MatXd D_x;
    MatXd D_v;
};

// Compute posterior covariance: X_post = X_pred - X_pred C^T (C X_pred C^T + Sigma_v)^{-1} C X_pred
MatXd posterior_cov(const MatXd& X_pred, const MatXd& Sigma_v, const MatXd& C_t);

// Combined objective + gradients in one linear solve
KalmanStats kalman_stats(const MatXd& X_pred, const MatXd& Sigma_v, const MatXd& C_t);

// Objective: f_min = -trace(X_post) (FW minimizes this)
double fw_objective_f_min(const MatXd& X_pred, const MatXd& Sigma_v, const MatXd& C_t);

} // namespace dr_mmse
