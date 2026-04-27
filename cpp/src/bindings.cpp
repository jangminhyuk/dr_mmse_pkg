// pybind11 bindings for the standalone DR-MMSE TAC solver.
//
// Exposes:
//   - solve_dr_mmse_tac           — primary worst-case Kalman update
//   - solve_dr_mmse_tac_factored  — same, parameterized over the reduced
//                                   noise space Q_w (when Sigma_w = sum_k
//                                   G_k Q_w G_k^T is rank-deficient)
//   - posterior_cov, kalman_stats — standalone Kalman utilities
//   - oracle_bisection_fast,
//     oracle_exact_fast           — Bures-Wasserstein-ball linear oracle
//
// No DR-EKF / DR-CDC / dynamics classes are exposed — this package is
// strictly the per-step minimax solver. Wire it into your own filter via
// the API documented in README.md.

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "dr_mmse/types.h"
#include "dr_mmse/kalman_utils.h"
#include "dr_mmse/fw_oracle.h"
#include "dr_mmse/dr_mmse_tac.h"

namespace py = pybind11;
using namespace dr_mmse;

PYBIND11_MODULE(dr_mmse_cpp, m) {
    m.doc() = "DR-MMSE TAC solver (Frank-Wolfe over Bures-Wasserstein balls). "
              "Pure C++/Eigen, no CUDA. Build once per target architecture.";

    // ---------------- Result structs ----------------
    py::class_<DRMMSEResult>(m, "DRMMSEResult",
            "Worst-case Kalman update (unfactored).")
        .def_readonly("wc_Sigma_v", &DRMMSEResult::wc_Sigma_v,
                      "Worst-case measurement noise (ny x ny).")
        .def_readonly("wc_Sigma_w", &DRMMSEResult::wc_Sigma_w,
                      "Worst-case process noise (nx x nx).")
        .def_readonly("wc_Xprior",  &DRMMSEResult::wc_Xprior,
                      "Worst-case prior covariance = APA + wc_Sigma_w.")
        .def_readonly("wc_Xpost",   &DRMMSEResult::wc_Xpost,
                      "Posterior covariance under the worst-case noise.")
        .def_readonly("success",    &DRMMSEResult::success)
        .def_readonly("iterations", &DRMMSEResult::iterations);

    py::class_<DRMMSEFactoredResult>(m, "DRMMSEFactoredResult",
            "Worst-case Kalman update (factored Sigma_w = sum_k G_k Q_w G_k^T).")
        .def_readonly("wc_Sigma_v", &DRMMSEFactoredResult::wc_Sigma_v)
        .def_readonly("wc_Q_w",     &DRMMSEFactoredResult::wc_Q_w,
                      "Worst-case noise in the reduced (nw x nw) space.")
        .def_readonly("wc_Sigma_w", &DRMMSEFactoredResult::wc_Sigma_w,
                      "Lifted worst-case process noise (nx x nx).")
        .def_readonly("wc_Xprior",  &DRMMSEFactoredResult::wc_Xprior)
        .def_readonly("wc_Xpost",   &DRMMSEFactoredResult::wc_Xpost)
        .def_readonly("success",    &DRMMSEFactoredResult::success)
        .def_readonly("iterations", &DRMMSEFactoredResult::iterations);

    // ---------------- Standalone solvers ----------------
    m.def("solve_dr_mmse_tac",
          [](const MatXd& APA, const MatXd& C,
             const MatXd& Sigma_w_hat, const MatXd& Sigma_v_hat,
             double theta_w, double theta_v,
             const std::string& solver,
             const py::object& warm_Sigma_w_obj,
             const py::object& warm_Sigma_v_obj,
             double fw_beta_minus1, double fw_tau,
             double fw_zeta, double fw_delta,
             int fw_max_iters, double fw_gap_tol,
             int fw_bisect_max_iters, double fw_bisect_tol,
             int fw_exact_iters, double fw_exact_gap_tol,
             double oracle_exact_tol, int oracle_exact_max_iters) {
              MatXd warm_w = warm_Sigma_w_obj.is_none() ? MatXd(0, 0) :
                  warm_Sigma_w_obj.cast<MatXd>();
              MatXd warm_v = warm_Sigma_v_obj.is_none() ? MatXd(0, 0) :
                  warm_Sigma_v_obj.cast<MatXd>();
              return solve_dr_mmse_tac(
                  APA, C, Sigma_w_hat, Sigma_v_hat,
                  theta_w, theta_v, solver, warm_w, warm_v,
                  fw_beta_minus1, fw_tau, fw_zeta, fw_delta,
                  fw_max_iters, fw_gap_tol,
                  fw_bisect_max_iters, fw_bisect_tol,
                  fw_exact_iters, fw_exact_gap_tol,
                  oracle_exact_tol, oracle_exact_max_iters);
          },
          py::arg("APA"),
          py::arg("C"),
          py::arg("Sigma_w_hat"),
          py::arg("Sigma_v_hat"),
          py::arg("theta_w"),
          py::arg("theta_v"),
          py::arg("solver") = "fw_exact",
          py::arg("warm_Sigma_w") = py::none(),
          py::arg("warm_Sigma_v") = py::none(),
          py::arg("fw_beta_minus1") = 1.0,
          py::arg("fw_tau") = 2.0,
          py::arg("fw_zeta") = 2.0,
          py::arg("fw_delta") = 0.05,
          py::arg("fw_max_iters") = 50,
          py::arg("fw_gap_tol") = 1e-4,
          py::arg("fw_bisect_max_iters") = 30,
          py::arg("fw_bisect_tol") = 1e-6,
          py::arg("fw_exact_iters") = 8,
          py::arg("fw_exact_gap_tol") = 1e-6,
          py::arg("oracle_exact_tol") = 1e-8,
          py::arg("oracle_exact_max_iters") = 60,
          R"pbdoc(
Solve the per-step DR-MMSE TAC minimax. Returns the worst-case noise
covariances within Bures-Wasserstein balls of radius (theta_w, theta_v)
around the nominal (Sigma_w_hat, Sigma_v_hat), and the corresponding
posterior covariance.

The TAC ('test-and-adapt covariance') decomposition assumes the prior is
expressed as
    P_prior = APA + Sigma_w
where APA = Phi_accum @ P_post_prev @ Phi_accum^T accumulates the
deterministic dynamics from the last measurement, and Sigma_w accumulates
the process-noise contributions over the same interval.

When theta_w <= 0 AND theta_v <= 0, this is a no-op: the solver returns
the standard Kalman posterior in zero iterations. Use that for parity
checks against your existing filter.

Returns DRMMSEResult with attributes:
    wc_Sigma_w, wc_Sigma_v, wc_Xprior, wc_Xpost, success, iterations.
)pbdoc");

    m.def("solve_dr_mmse_tac_factored",
          [](const MatXd& APA, const MatXd& C,
             const std::vector<MatXd>& G_list,
             const MatXd& Q_hat, const MatXd& Sigma_v_hat,
             double theta_w, double theta_v,
             const std::string& solver,
             const py::object& warm_Q_w_obj,
             const py::object& warm_Sigma_v_obj,
             double fw_beta_minus1, double fw_tau,
             double fw_zeta, double fw_delta,
             int fw_max_iters, double fw_gap_tol,
             int fw_bisect_max_iters, double fw_bisect_tol,
             int fw_exact_iters, double fw_exact_gap_tol,
             double oracle_exact_tol, int oracle_exact_max_iters) {
              MatXd warm_q = warm_Q_w_obj.is_none() ? MatXd(0, 0) :
                  warm_Q_w_obj.cast<MatXd>();
              MatXd warm_v = warm_Sigma_v_obj.is_none() ? MatXd(0, 0) :
                  warm_Sigma_v_obj.cast<MatXd>();
              return solve_dr_mmse_tac_factored(
                  APA, C, G_list, Q_hat, Sigma_v_hat,
                  theta_w, theta_v, solver, warm_q, warm_v,
                  fw_beta_minus1, fw_tau, fw_zeta, fw_delta,
                  fw_max_iters, fw_gap_tol,
                  fw_bisect_max_iters, fw_bisect_tol,
                  fw_exact_iters, fw_exact_gap_tol,
                  oracle_exact_tol, oracle_exact_max_iters);
          },
          py::arg("APA"),
          py::arg("C"),
          py::arg("G_list"),
          py::arg("Q_hat"),
          py::arg("Sigma_v_hat"),
          py::arg("theta_w"),
          py::arg("theta_v"),
          py::arg("solver") = "fw_exact",
          py::arg("warm_Q_w") = py::none(),
          py::arg("warm_Sigma_v") = py::none(),
          py::arg("fw_beta_minus1") = 1.0,
          py::arg("fw_tau") = 2.0,
          py::arg("fw_zeta") = 2.0,
          py::arg("fw_delta") = 0.05,
          py::arg("fw_max_iters") = 50,
          py::arg("fw_gap_tol") = 1e-4,
          py::arg("fw_bisect_max_iters") = 30,
          py::arg("fw_bisect_tol") = 1e-6,
          py::arg("fw_exact_iters") = 8,
          py::arg("fw_exact_gap_tol") = 1e-6,
          py::arg("oracle_exact_tol") = 1e-8,
          py::arg("oracle_exact_max_iters") = 60,
          R"pbdoc(
Factored variant for filters whose process noise lives in a smaller
subspace (e.g. a 21-state INS with a 6D acc+gyro noise input). The
adversary optimizes over Q_w in the nw-dimensional space; the lifted
Sigma_w = sum_k G_k Q_w G_k^T is then used in the prior. The Bures-
Wasserstein constraint acts on the well-conditioned nw x nw matrix.

Pass an empty G_list to disable process-noise robustification (this
silently fixes Q_w = Q_hat regardless of theta_w).
)pbdoc");

    // ---------------- Utility functions ----------------
    m.def("posterior_cov", &posterior_cov,
          py::arg("X_pred"), py::arg("Sigma_v"), py::arg("C"),
          "Standard Kalman posterior covariance: "
          "X_pred - X_pred C^T (C X_pred C^T + Sigma_v)^{-1} C X_pred.");

    m.def("kalman_stats",
          [](const MatXd& X_pred, const MatXd& Sigma_v, const MatXd& C_t) {
              auto r = kalman_stats(X_pred, Sigma_v, C_t);
              return py::make_tuple(r.trace_post, r.D_x, r.D_v);
          },
          py::arg("X_pred"), py::arg("Sigma_v"), py::arg("C"),
          "Returns (trace_post, dF/dX_pred, dF/dSigma_v) for F = -trace(X_post). "
          "Used internally by the FW solver; exposed for testing.");

    m.def("oracle_bisection_fast", &oracle_bisection_fast,
          py::arg("Sigma_hat"), py::arg("rho"),
          py::arg("Sigma_ref"), py::arg("D"), py::arg("delta"),
          py::arg("bisect_max_iters") = 30, py::arg("bisect_tol") = 1e-6,
          "Approximate maximizer of <D, Sigma> over BW(Sigma, Sigma_hat) <= rho.");

    m.def("oracle_exact_fast", &oracle_exact_fast,
          py::arg("Sigma_hat"), py::arg("rho"), py::arg("D"),
          py::arg("tol") = 1e-8, py::arg("max_iters") = 60,
          "Exact maximizer of <D, Sigma> over BW(Sigma, Sigma_hat) <= rho.");
}
