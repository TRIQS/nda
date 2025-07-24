// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a class that can solve multiple linear least squares problems for a given matrix \f$ \mathbf{A} \f$.
 */

#pragma once

#include "./gesvd.hpp"
#include "../algorithms.hpp"
#include "../arithmetic.hpp"
#include "../basic_array.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../layout/policies.hpp"
#include "../layout_transforms.hpp"
#include "../linalg.hpp"
#include "../mapped_functions.hpp"
#include "../matrix_functions.hpp"
#include "nda/traits.hpp"

#include <itertools/itertools.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <optional>
#include <utility>
#include <vector>

namespace nda::lapack {

  /**
   * @addtogroup linalg_lapack
   * @{
   */

  /**
   * @brief Worker class for solving linear least squares problems.
   *
   * @details Solving a linear least squares problem means finding the minimum norm solution \f$ \mathbf{x} \f$ of a
   * linear system of equations, i.e.
   * \f[
   *   \min_x | \mathbf{b} - \mathbf{A x} |_2 \; ,
   * \f]
   * where \f$ \mathbf{A} \f$ is a given matrix and \f$ \mathbf{b} \f$ is a given vector (although it can also be a
   * matrix, in this case one searches for a solution matrix \f$ \mathbf{X} \f$).
   *
   * Let \f$ \mathbf{A} \in \mathbb{C}^{M \times N} \f$ with rank \f$ \rho \leq \min(M, N) \f$. Its singular value
   * decomposition (SVD) is given by
   * \f[
   *   \mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^H =
   *   \begin{bmatrix} \mathbf{U}_R & \mathbf{U}_N \end{bmatrix}
   *   \begin{bmatrix} \mathbf{S} & 0 \\ 0 & 0 \end{bmatrix}
   *   \begin{bmatrix} \mathbf{V}_R^H \\ \mathbf{V}_N^H \end{bmatrix} \; ,
   * \f]
   * where \f$ \mathbf{U}_R \in \mathbb{C}^{M \times \rho} \f$, \f$ \mathbf{U}_N \in \mathbb{C}^{M \times (M - \rho)}
   * \f$, \f$ \mathbf{S} \in \mathbb{R}^{\rho \times \rho} \f$, \f$ \mathbf{V}_R \in \mathbb{C}^{N \times \rho} \f$,
   * and \f$ \mathbf{V}_N \in \mathbb{C}^{N \times (N - \rho)} \f$.
   *
   * The least squares solution can then be written as
   * \f[
   *   \mathbf{x} = \mathbf{A}^{+} \mathbf{b} = \mathbf{V} \mathbf{\Sigma}^{+} \mathbf{U}^H \mathbf{b} =
   *   \mathbf{V}_R \mathbf{S}^{-1} \mathbf{U}_R^H \mathbf{b} \; ,
   * \f]
   * where \f$ \mathbf{M}^{+} \f$ denotes the Moore-Penrose pseudo-inverse of the matrix \f$ \mathbf{M} \f$, and the
   * residual error as
   * \f[
   *   \epsilon = \left\| \mathbf{A} \mathbf{x} - \mathbf{b} \right\|_2 = \left\| \mathbf{U}_N^H \mathbf{b} \right\|_2
   *   \; .
   * \f]
   *
   * See <a href="https://math.stackexchange.com/questions/772039/how-does-the-svd-solve-the-least-squares-problem">this
   * question</a> on math.stackexchange for more information.
   *
   * @tparam T Value type of the given problem.
   */
  template <typename T>
  class gelss_worker {
    public:
    /**
     * @brief Get the number of variables of the given problem, i.e. the size of the vector \f$ \mathbf{x} \f$.
     * @return Number of columns of the matrix \f$ \mathbf{A} \f$ .
     */
    int n_var() const { return N_; }

    /**
     * @brief Get the singular values, i.e. the diagonal elements of the matrix \f$ \mathbf{S} \f$.
     * @return 1-dimensional nda::array containing the singular values.
     */
    [[nodiscard]] array<double, 1> const &S_vec() const { return s_; }

    /**
     * @brief Construct a new worker object for a given matrix \f$ \mathbf{A} \f$ .
     *
     * @details It performs the SVD decomposition of the given matrix \f$ \mathbf{A} \f$ and stores its pseudo inverse
     * \f$ \mathbf{A}^{+} \f$, its singular values \f$ \mathbf{s} = \mathrm{diag}(\mathbf{S}) \f$ and the matrix \f$
     * \mathbf{U}_N^H \f$. The latter is used to calculate the error of the least squares problem.
     *
     * @param A %Matrix \f$ \mathbf{A} \f$ used in the least squares problem.
     */
    gelss_worker(matrix_const_view<T> A) : M_(A.extent(0)), N_(A.extent(1)), s_(std::min(M_, N_)) {
      if (N_ > M_) NDA_RUNTIME_ERROR << "Error in nda::lapack::gelss_worker: Matrix A cannot have more columns than rows";

      // initialize matrices
      matrix<T, F_layout> A_work{A};
      matrix<T, F_layout> U(M_, M_);
      matrix<T, F_layout> V_H(N_, N_);

      // calculate the SVD: A = U * \Sigma * V^H
      gesvd(A_work, s_, U, V_H);

      // calculate the pseudo inverse A^{+} = V * \Sigma^{+} * U^H
      matrix<get_fp_t<T>, F_layout> S_plus(N_, M_);
      S_plus = 0.;
      for (long i : range(s_.size())) S_plus(i, i) = 1.0 / s_(i);
      A_plus_ = dagger(V_H) * S_plus * dagger(U);

      // set U_N^H
      if (N_ < M_) U_N_H_ = dagger(U)(range(N_, M_), range(M_));
    }

    /**
     * @brief Solve the least squares problem for a given right hand side matrix \f$ \mathbf{B} \f$.
     *
     * @details It calculates and returns the least squares solution \f$ \mathbf{X} = \mathbf{A}^{+} \mathbf{B} \f$ and
     * the error
     * \f[
     *   \epsilon = \max\left\{ \frac{ \left\| \mathbf{U}_N^H \mathbf{b} \right\|_2 }{ \sqrt{N} } : \mathbf{b}
     *   \text{ is a column vector in } \mathbf{B} \right\} \; .
     * \f]
     *
     * @param B Right hand side matrix.
     * @return A `std::pair<matrix<T>, double>` containing the solution matrix \f$ \mathbf{X} \f$ and the error \f$
     * \epsilon \f$.
     */
    auto operator()(matrix_const_view<T> B, std::optional<long> /* inner_matrix_dim */ = {}) const {
      using std::sqrt;
      double err = 0.0;
      if (M_ != N_) {
        std::vector<double> err_vec;
        for (long i : range(B.shape()[1])) err_vec.push_back(frobenius_norm(U_N_H_ * B(range::all, range(i, i + 1))) / sqrt(B.shape()[0]));
        err = *std::ranges::max_element(err_vec);
      }
      return std::pair<matrix<T>, double>{A_plus_ * B, err};
    }

    /**
     * @brief Solve the least squares problem for a given right hand side vector \f$ \mathbf{b} \f$.
     *
     * @details It calculates and returns the least squares solution \f$ \mathbf{x} = \mathbf{A}^{+} \mathbf{b} \f$ and
     * the error \f$ \epsilon = \frac{ \left\| \mathbf{U}_N^H \mathbf{b} \right\|_2 }{ \sqrt{N} } \f$.
     *
     * @param b Right hand side vector.
     * @return A `std::pair<vector<T>, double>` containing the solution vector \f$ \mathbf{x} \f$ and the error \f$
     * \epsilon \f$.
     */
    auto operator()(vector_const_view<T> b, std::optional<long> /*inner_matrix_dim*/ = {}) const {
      using std::sqrt;
      double err = 0.0;
      if (M_ != N_) { err = nda::linalg::norm(U_N_H_ * b) / sqrt(b.size()); }
      return std::pair<vector<T>, double>{A_plus_ * b, err};
    }

    private:
    // Number of rows (M) and columns (N) of the Matrix A.
    long M_, N_;

    // Pseudo inverse of A, i.e. A^{+} = V * \Sigma^{+} * U^H.
    matrix<T> A_plus_;

    // U_N^H defining the error of the least squares problem.
    matrix<T> U_N_H_;

    // Array containing the singular values.
    array<get_fp_t<T>, 1> s_;
  };

  /**
   * @brief Specialized worker class for solving linear least squares problems while enforcing a certain hermitian
   * symmetry.
   *
   * @details This is a more specialized version of nda::lapack::gelss_worker that is tailored to perform tail fitting
   * for scalar- or matrix-valued functions defined on a Matsubara frequency mesh satisfying the symmetry \f$ \left[
   * f(-i\omega_n) \right]_{ij} = \left[ f(i\omega_n) \right]_{ji}^* \f$.
   *
   * The least squares problem to be solved is given by
   * \f[
   *   \mathbf{A}_E \mathbf{X} = \begin{bmatrix} \mathbf{A} \\ \mathbf{A}^* \end{bmatrix} \mathbf{X} =
   *   \begin{bmatrix} \mathbf{B} \\ \tilde{\mathbf{B}} \end{bmatrix} = \mathbf{B}_E \; ,
   * \f]
   * where the original problem \f$ \mathbf{A} \mathbf{X} = \mathbf{B} \f$ has been extended by \f$ \mathbf{A}^*
   * \mathbf{X} = \tilde{\mathbf{B}} \f$ to account for the symmetry given above.
   *
   * The calculated solution \f$ \mathbf{X} \f$ is explictly symmetrized at the end to ensure that the symmetry is
   * satisfied.
   *
   * See `triqs::mesh::tail_fitter` for more information.
   */
  class gelss_worker_hermitian {
    public:
    /**
     * @brief Get the number of variables of the given problem.
     * @return Number of columns of the matrix \f$ \mathbf{A} \f$.
     */
    int n_var() const { return lss_herm_.n_var(); }

    /**
     * @brief Get the singular values of the original matrix \f$ A \f$.
     * @return 1-dimensional nda::array containing the singular values.
     */
    [[nodiscard]] array<double, 1> const &S_vec() const { return lss_.S_vec(); }

    /**
     * @brief Construct a new worker object for a given matrix \f$ \mathbf{A} \f$.
     * @param A %Matrix \f$ \mathbf{A} \f$ used in the least squares problem.
     */
    gelss_worker_hermitian(matrix_const_view<std::complex<double>> A) : lss_(A), lss_herm_(vstack(A, conj(A))) {}

    /**
     * @brief Solve the least squares problem for a given right hand side matrix \f$ \mathbf{B} \f$.
     *
     * @details The inner matrix dimension \f$ d \f$ specifies the shape of the matrix that \f$ f(i\omega_n) \f$
     * returns. It is used to make sure that the symmetry is satisfied. For scalar-valued functions, \f$ d = 1 \f$.
     *
     * @param B Right hand side matrix.
     * @param inner_matrix_dim Inner matrix dimension \f$ d \f$.
     * @return A `std::pair<matrix<std::complex<double>>, double>` containing the solution matrix \f$ \mathbf{X} \f$ and the error
     * \f$ \epsilon \f$.
     */
    auto operator()(matrix_const_view<std::complex<double>> B, std::optional<long> inner_matrix_dim = {}) const {
      if (not inner_matrix_dim.has_value())
        NDA_RUNTIME_ERROR << "Error in nda::lapack::gelss_worker_hermitian: Inner matrix dimension required for hermitian least square fitting";
      long d = *inner_matrix_dim;

      // take the inner 'adjoint' of a matrix C:
      // * reshape C -> C': (M, N) -> (M, N', d, d)
      // * for each m and n: C'(m, n, :, :) -> C'(m, n, :, :)^/dagger
      // * reshape C' -> C: (M, N', d, d) -> (M, N)
      auto inner_adjoint = [d](auto &C) {
        NDA_ASSERT2(C.shape()[1] % (d * d) == 0, "Error in nda::lapack::gelss_worker_hermitian: Data shape incompatible with inner matrix dimension");
        auto shape = C.shape();

        // get extent N' of second dimension
        long N = shape[1] / (d * d);

        // reshape, transpose and take the complex conjugate
        array<std::complex<double>, 4> arr_dag =
           conj(permuted_indices_view<encode(std::array{0, 1, 3, 2})>(reshape(C, std::array{shape[0], N, d, d})));

        // return the result in a new matrix
        return matrix<std::complex<double>>{reshape(std::move(arr_dag), shape)};
      };

      // solve the extended system vstack(A, A*) * X = vstack(B, B_dag)
      auto B_dag    = inner_adjoint(B);
      auto [x, err] = lss_herm_(vstack(B, B_dag));

      // resymmetrize the results to cure small hermiticity violations
      return std::pair<matrix<std::complex<double>>, double>{0.5 * (x + inner_adjoint(x)), err};
    }

    private:
    // Worker for the original least squares problem.
    gelss_worker<std::complex<double>> lss_;

    // Worker for the extended least squares problem.
    gelss_worker<std::complex<double>> lss_herm_;
  };

  /** @} */

} // namespace nda::lapack
