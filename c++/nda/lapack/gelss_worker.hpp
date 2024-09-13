// Copyright (c) 2020-2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn, Jason Kaye, Olivier Parcollet, Nils Wentzell

#pragma once

#include "./gesvd.hpp"
#include "../algorithms.hpp"
#include "../basic_array.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../layout/policies.hpp"
#include "../layout_transforms.hpp"
#include "../linalg.hpp"
#include "../mapped_functions.hpp"
#include "../matrix_functions.hpp"

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
   * @brief Worker class for solving linear least square problems.
   *
   * @details Solving a linear least squares problem means finding the minimum norm solution \f$ \mathbf{x} \f$ of a
   * linear system of equations, i.e.
   * \f[
   *   \min_x | \mathbf{b} - \mathbf{A x} |_2 \; ,
   * \f]
   * where \f$ \mathbf{A} \f$ is a given matrix and \f$ \mathbf{b} \f$ is a given vector (although it can also be a
   * matrix, in this case one gets a solution matrix\f$ \mathbf{X} \f$).
   *
   * See https://math.stackexchange.com/questions/772039/how-does-the-svd-solve-the-least-squares-problem for the
   * notation used in this file.
   *
   * @tparam T Value type of the given problem.
   */
  template <typename T>
  class gelss_worker {
    // Number of rows (M) and columns (N) of the Matrix A.
    long M, N;

    // FIXME Do we need to store it ? only use n_var
    // Matrix to be decomposed by SVD.
    matrix<T> A;

    // (Pseudo) Inverse of A, i.e. V * Diag(S_vec)^{-1} * UH, for the least square problem.
    matrix<T> V_x_InvS_x_UH;

    // Part of UH fixing the error of the least square problem.
    matrix<T> UH_NULL;

    // Array containing the singular values.
    array<double, 1> s_vec;

    public:
    /**
     * @brief Get the number of variables of the given problem.
     * @return Number of columns of the matrix \f$ \mathbf{A} \f$ .
     */
    int n_var() const { return A.extent(1); }

    /**
     * @brief Get the singular value array.
     * @return 1-dimensional array containing the singular values.
     */
    [[nodiscard]] array<double, 1> const &S_vec() const { return s_vec; }

    /**
     * @brief Construct a new worker object for a given matrix \f$ \mathbf{A} \f$ .
     *
     * @details It performs the SVD decomposition of the given matrix \f$ \mathbf{A} \f$  and calculates the (pseudo)
     * inverse of \f$ \mathbf{A} \f$. Furthermore, it sets the null space term which determines the error of the least
     * square problem.
     *
     * @param A_ Matrix to be decomposed by SVD.
     */
    gelss_worker(matrix<T> A_) : M(A_.extent(0)), N(A_.extent(1)), A(std::move(A_)), s_vec(std::min(M, N)) {
      if (N > M) NDA_RUNTIME_ERROR << "Error in nda::lapack::gelss_worker: Matrix A cannot have more columns than rows";

      // initialize matrices
      matrix<T, F_layout> A_FL{A};
      matrix<T, F_layout> U(M, M);
      matrix<T, F_layout> VH(N, N);

      // calculate the SVD: A = U * Diag(S_vec) * VH
      gesvd(A_FL, s_vec, U, VH);

      // calculate the matrix V * Diag(S_vec)^{-1} * UH for the least square procedure
      matrix<double, F_layout> S_inv(N, M);
      S_inv = 0.;
      for (long i : range(std::min(M, N))) S_inv(i, i) = 1.0 / s_vec(i);
      V_x_InvS_x_UH = dagger(VH) * S_inv * dagger(U);

      // read off UH_Null for defining the error of the least square procedure
      if (N < M) UH_NULL = dagger(U)(range(N, M), range(M));
    }

    /**
     * @brief Solve the least-square problem for a given right hand side matrix \f$ \mathbf{B} \f$.
     *
     * @param B Right hand side matrix.
     * @return A std::pair containing the solution matrix \f$ \mathbf{X} \f$ and the error of the least square problem.
     */
    std::pair<matrix<T>, double> operator()(matrix_const_view<T> B, std::optional<long> /* inner_matrix_dim */ = {}) const {
      using std::sqrt;
      double err = 0.0;
      if (M != N) {
        std::vector<double> err_vec;
        for (long i : range(B.shape()[1])) err_vec.push_back(frobenius_norm(UH_NULL * B(range::all, range(i, i + 1))) / sqrt(B.shape()[0]));
        err = *std::max_element(err_vec.begin(), err_vec.end());
      }
      return std::make_pair(V_x_InvS_x_UH * B, err);
    }

    /**
     * @brief Solve the least-square problem for a given right hand side vector \f$ \mathbf{b} \f$.
     *
     * @param b Right hand side vector.
     * @return A std::pair containing the solution vector \f$ \mathbf{x} \f$ and the error of the least square problem.
     */
    std::pair<vector<T>, double> operator()(vector_const_view<T> b, std::optional<long> /*inner_matrix_dim*/ = {}) const {
      using std::sqrt;
      double err = 0.0;
      if (M != N) { err = norm(UH_NULL * b) / sqrt(b.size()); }
      return std::make_pair(V_x_InvS_x_UH * b, err);
    }
  };

  /**
   * @brief Worker class for solving linear least square problems for hermitian tail-fitting.
   * @details Restrict the resulting vector of moment matrices to one of hermitian matrices.
   *
   * See also nda::lapack::gelss_worker.
   */
  struct gelss_worker_hermitian {
    private:
    // Complex double type.
    using dcomplex = std::complex<double>;

    // Matrix to be decomposed by SVD.
    matrix<dcomplex> A;

    // Solver for the associated real-valued least-squares problem.
    gelss_worker<dcomplex> _lss;

    // Solver for the associated real-valued least-squares problem imposing hermiticity.
    gelss_worker<dcomplex> _lss_matrix;

    public:
    /**
     * @brief Get the number of variables of the given problem.
     * @return Number of columns of the matrix \f$ \mathbf{A} \f$.
     */
    int n_var() const { return static_cast<int>(A.extent(1)); }

    /**
     * @brief Get the singular value array.
     * @return 1-dimensional array containing the singular values.
     */
    array<double, 1> const &S_vec() const { return _lss.S_vec(); }

    /**
     * @brief Construct a new worker object for a given matrix \f$ \mathbf{A} \f$.
     * @param A_ Matrix to be decomposed by SVD.
     */
    gelss_worker_hermitian(matrix<dcomplex> A_) : A(std::move(A_)), _lss(A), _lss_matrix(vstack(A, conj(A))) {}

    /**
     * @brief Solve the least-square problem for a given right hand side matrix \f$ \mathbf{B} \f$.
     * @param B Right hand side matrix.
     * @param inner_matrix_dim Inner matrix dimension for hermitian least square fitting.
     * @return A std::pair containing the solution matrix \f$ \mathbf{X} \f$ and the error of the least square problem.
     */
    std::pair<matrix<dcomplex>, double> operator()(matrix_const_view<dcomplex> B, std::optional<long> inner_matrix_dim = {}) const {
      if (not inner_matrix_dim.has_value())
        NDA_RUNTIME_ERROR << "Error in nda::lapack::gelss_worker_hermitian: Inner matrix dimension required for hermitian least square fitting";
      long d = *inner_matrix_dim;

      // Construction of an inner 'adjoint' matrix by performing the following steps
      // * reshape B from (M, M1) to (M, N, d, d)
      // * for each M and N take the adjoint matrix (d, d)
      // * reshape to (M, M)
      auto inner_adjoint = [d](auto &M) {
        auto idx_map = M.indexmap();
        auto l       = idx_map.lengths();
        //auto s       = idx_map.strides();

        NDA_ASSERT2(l[1] % (d * d) == 0, "Error in nda::lapack::gelss_worker_hermitian: Data shape incompatible with given dimension");
        long N = l[1] / (d * d);

        // We reshape the Matrix into a dim=4 array and swap the two innermost indices

        // FIXME OLD CODE  SURPRESS AFTER PORTING
        // FIXME We would like to write: transpose(reshape(idx_map, {l[0], N, d, d}), {0, 1, 3, 2})
        // auto idx_map_inner_transpose = array_view<dcomplex, 4>::layout_t{{l[0], N, d, d}, {s[0], d * d * s[1], s[1], d * s[1]}};
        // Deep copy
        //array<dcomplex, 4> arr_dag = conj(array_const_view<dcomplex, 4>{idx_map_inner_transpose, M.storage()});
        //return matrix<dcomplex>{matrix<dcomplex>::layout_t{l, s}, std::move(arr_dag).storage()};

        // FIXME C++20 remove encode
        auto arr_dag = array<dcomplex, 4>{conj(permuted_indices_view<encode(std::array{0, 1, 3, 2})>(reshape(M, std::array{l[0], N, d, d})))};

        return matrix<dcomplex>{reshape(std::move(arr_dag), l)}; // move into a matrix
      };

      // Solve the enlarged system vstack(A, A*) * x = vstack(B, B_dag)
      matrix<dcomplex> B_dag = inner_adjoint(B);
      auto B_stack           = vstack(B, B_dag);
      auto [x, err]          = _lss_matrix(B_stack);

      // Resymmetrize results to cure small hermiticity violations
      return {matrix<dcomplex>{0.5 * (x + inner_adjoint(x))}, err};
    }
  };

  /** @} */

} // namespace nda::lapack
