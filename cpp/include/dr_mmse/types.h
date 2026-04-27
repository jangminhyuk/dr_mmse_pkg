#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace dr_mmse {

using MatXd = Eigen::MatrixXd;
using VecXd = Eigen::VectorXd;

// Symmetrize a matrix: 0.5*(A + A^T)
inline MatXd sym(const MatXd& A) {
    return 0.5 * (A + A.transpose());
}

// Frobenius inner product: trace(A^T * B)
inline double inner(const MatXd& A, const MatXd& B) {
    return (A.array() * B.array()).sum();
}

// Frobenius norm squared: ||A||_F^2
inline double fro_norm2(const MatXd& A) {
    return A.squaredNorm();
}

} // namespace dr_mmse
