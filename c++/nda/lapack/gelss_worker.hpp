// Copyright (c) 2020-2021 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once

#include <optional>

#include "./gesvd.hpp"

namespace nda::lapack {

  template <typename T>
  class gelss_worker {
    // cf. Notation in https://math.stackexchange.com/questions/772039/how-does-the-svd-solve-the-least-squares-problem

    // Number of rows (M) and columns (N) of the Matrix A
    long M, N;

    // The matrix to be decomposed by SVD
    // FIXME Do we need to store it ? only use n_var
    matrix<T> A;

    // The (pseudo) inverse of A, i.e. V * Diag(S_vec)^{-1} * UT, for the least square procedure
    matrix<T> V_x_InvS_x_UT;

    // The part of UT fixing the error of the LLS
    matrix<T> UT_NULL;

    // Vector containing the singular values
    array<double, 1> s_vec;

    public:
    int n_var() const { return A.extent(1); }

    /// ???
    // FIXME Looks it is not used
    //matrix<T> const &A_mat() const { return A; }

    /// ???
    array<double, 1> const &S_vec() const { return s_vec; }

    /// ???
    gelss_worker(matrix<T> _A) : M(_A.extent(0)), N(_A.extent(1)), A(std::move(_A)), s_vec(std::min(M, N)) {

      if (N > M) NDA_RUNTIME_ERROR << "ERROR: Matrix A for linear least square procedure cannot have more columns than rows";

      matrix<T, F_layout> A_FL{A};
      matrix<T, F_layout> U(M, M);
      matrix<T, F_layout> VT(N, N);

      // Calculate the SVD A = U * Diag(S_vec) * VT
      gesvd(A_FL, s_vec, U, VT);

      // Calculate the matrix V * Diag(S_vec)^{-1} * UT for the least square procedure
      matrix<double, F_layout> S_inv(N, M);
      S_inv = 0.;
      for (int i : range(std::min(M, N))) S_inv(i, i) = 1.0 / s_vec(i);
      V_x_InvS_x_UT = dagger(VT) * S_inv * dagger(U);

      // Read off U_Null for defining the error of the least square procedure
      if (N < M) UT_NULL = dagger(U)(range(N, M), range(M));
    }

    /// Solve the least-square problem that minimizes || A * x - B ||_2 given A and B
    std::pair<matrix<T>, double> operator()(matrix_const_view<T> B, std::optional<long> /*inner_matrix_dim*/ = {}) const {
      using std::sqrt;
      double err = 0.0;
      if (M != N) {
        std::vector<double> err_vec;
        for (int i : range(B.shape()[1])) err_vec.push_back(frobenius_norm(UT_NULL * B(range(), range(i, i + 1))) / sqrt(B.shape()[0]));
        err = *std::max_element(err_vec.begin(), err_vec.end());
      }
      return std::make_pair(V_x_InvS_x_UT * B, err);
    }
  };

  // Least square solver version specific for hermitian tail-fitting.
  // Restrict the resulting vector of moment matrices to one of hermitian matrices
  struct gelss_worker_hermitian {

    using dcomplex = std::complex<double>;

    private:
    // The matrix to be decomposed by SVD
    matrix<dcomplex> A;

    // Solver for the associated real-valued least-squares problem
    gelss_worker<dcomplex> _lss;

    // Solver for the associated real-valued least-squares problem imposing hermiticity
    gelss_worker<dcomplex> _lss_matrix;

    public:
    int n_var() const { return A.extent(1); }

    //matrix<dcomplex> const &A_mat() const { return A; }
    array<double, 1> const &S_vec() const { return _lss.S_vec(); }

    gelss_worker_hermitian(matrix<dcomplex> _A) : A(std::move(_A)), _lss(A), _lss_matrix(vstack(A, conj(A))) {}

    // Solve the least-square problem that minimizes || A * x - B ||_2 given A and B with a real-valued vector x
    std::pair<matrix<dcomplex>, double> operator()(matrix_const_view<dcomplex> B, std::optional<long> inner_matrix_dim = {}) const {

      if (not inner_matrix_dim.has_value()) NDA_RUNTIME_ERROR << "Inner matrix dimension required for hermitian least square fitting\n";
      long d = *inner_matrix_dim;

      // Construction of an inner 'adjoint' matrix by performing the following steps
      // * reshape B from (M, M1) to (M, N, d, d)
      // * for each M and N take the adjoint matrix (d, d)
      // * reshape to (M, M)
      auto inner_adjoint = [d](auto &M) {
        auto idx_map = M.indexmap();
        auto l       = idx_map.lengths();
        //auto s       = idx_map.strides();

        NDA_ASSERT2(l[1] % (d * d) == 0, "ERROR in hermitian least square fitting: Data shape incompatible with given dimension");
        long N = l[1] / (d * d);

        // We reshape the Matrix into a dim=4 array and swap the two innermost indices

        // FIXME OLD CODE  SUPRRESS AFTER PORTING
        // FIXME We would like to write: tranpose(reshape(idx_map, {l[0], N, d, d}), {0, 1, 3, 2})
        // auto idx_map_inner_transpose = array_view<dcomplex, 4>::layout_t{{l[0], N, d, d}, {s[0], d * d * s[1], s[1], d * s[1]}};
        // Deep copy
        //array<dcomplex, 4> arr_dag = conj(array_const_view<dcomplex, 4>{idx_map_inner_transpose, M.storage()});
        //return matrix<dcomplex>{matrix<dcomplex>::layout_t{l, s}, std::move(arr_dag).storage()};

        // FIXME C++20 remove encode
        array<dcomplex, 4> arr_dag = conj(permuted_indices_view<encode(std::array{0, 1, 3, 2})>(reshaped_view(M, std::array{l[0], N, d, d})));

        return matrix<dcomplex>{reshape(std::move(arr_dag), l)}; // move into a matrix
      };

      // Solve the enlarged system vstack(A, A*) * x = vstack(B, B_dag)
      matrix<dcomplex> B_dag = inner_adjoint(B);
      auto B_stack           = vstack(B, B_dag);
      auto [x, err]          = _lss_matrix(B_stack);

      // Resymmetrize results to cure small hermiticity violations
      return {0.5 * (x + inner_adjoint(x)), err};
    }
  };

} // namespace nda::lapack
