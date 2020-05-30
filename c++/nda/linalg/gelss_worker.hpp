/*******************************************************************************
 *
 * Copyright (C) 2012-2017 by O. Parcollet
 * Copyright (C) 2018 by Simons Foundation
 *   author : O. Parcollet, P. Dumitrescu, N. Wentzell
 *
 ******************************************************************************/
#pragma once

#include <nda/blas/gelss.hpp>
#include <nda/blas/gesvd.hpp>

namespace nda { 

  template <typename T> class gelss_worker {
    // cf. Notation in https://math.stackexchange.com/questions/772039/how-does-the-svd-solve-the-least-squares-problem
    
    // Number of rows (M) and columns (N) of the Matrix A
    long M, N;

    // The matrix to be decomposed by SVD
    matrix<T> A;

    // The (pseudo) inverse of A, i.e. V * Diag(S_vec)^{-1} * UT, for the least square procedure
    matrix<T> V_x_InvS_x_UT;

    // The part of UT fixing the error of the LLS
    matrix<T> UT_NULL;

    // Vector containing the singular values
    vector<double> s_vec;

    public:
    int n_var() const { return A.extent(1); }
    
    /// ???
    matrix<T> const &A_mat() const { return A; }
    
    /// ???
    vector<double> const &S_vec() const { return s_vec; }

    /// ???
    gelss_worker(matrix_const_view<T> _A) : M(first_dim(_A)), N(second_dim(_A)), A(_A), s_vec(std::min(M, N)) {

      if (N > M) TRIQS_RUNTIME_ERROR << "ERROR: Matrix A for linear least square procedure cannot have more columns than rows";

      matrix<T> A_FL{_A, FORTRAN_LAYOUT};
      matrix<T> U{M, M, FORTRAN_LAYOUT};
      matrix<T> VT{N, N, FORTRAN_LAYOUT};

      // Calculate the SVD A = U * Diag(S_vec) * VT
      gesvd(A_FL, s_vec, U, VT);

      // Calculate the matrix V * Diag(S_vec)^{-1} * UT for the least square procedure
      matrix<double> S_inv{N, M, FORTRAN_LAYOUT};
      S_inv() = 0.;
      for (int i : range(std::min(M, N))) S_inv(i, i) = 1.0 / s_vec(i);
      V_x_InvS_x_UT = dagger(VT) * S_inv * dagger(U);

      // Read off U_Null for defining the error of the least square procedure
      if (N < M) UT_NULL = dagger(U)(range(N, M), range(M));
    }

    /// Solve the least-square problem that minimizes || A * x - B ||_2 given A and B
    std::pair<matrix<T>, double> operator()(matrix_const_view<T> B, std::optional<long> inner_matrix_dim = {} /*unused*/) const {
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
    int n_var() const { return second_dim(A); }
    matrix<dcomplex> const &A_mat() const { return A; }
    vector<double> const &S_vec() const { return _lss.S_vec(); }

    gelss_worker_hermitian(matrix_const_view<dcomplex> _A) : A(_A), _lss(A), _lss_matrix(vstack(A, conj(A))) {}

    // Solve the least-square problem that minimizes || A * x - B ||_2 given A and B with a real-valued vector x
    std::pair<matrix<dcomplex>, double> operator()(matrix_const_view<dcomplex> B, std::optional<long> inner_matrix_dim = {}) const {

      if (not inner_matrix_dim.has_value()) TRIQS_RUNTIME_ERROR << "Inner matrix dimension required for hermitian least square fitting\n";
      unsigned d = *inner_matrix_dim;

      // Construction of an inner 'adjoint' matrix by performing the following steps
      // * reshape B from (M, M1) to (M, N, d, d)
      // * for each M and N take the adjoint matrix (d, d)
      // * reshape to (M, M)
      auto inner_adjoint = [&d](matrix_view<dcomplex> M) {
        auto idx_map = M.indexmap();
        auto l       = idx_map.lengths();
        auto s       = idx_map.strides();

        TRIQS_ASSERT2(l[1] % (d * d) == 0, "ERROR in hermitian least square fitting: Data shape incompatible with given dimension");
        long N = l[1] / (d * d);

        // We reshape the Matrix into a dim=4 array and swap the two innermost indices
	// FIXME We would like to write: tranpose(reshape(idx_map, {l[0], N, d, d}), {0, 1, 3, 2})
	auto idx_map_inner_transpose = array_view<dcomplex, 4>::indexmap_type{
	   {l[0], N, d, d}, {s[0], d * d * s[1], s[1], d * s[1]}, static_cast<ptrdiff_t>(idx_map.start_shift())};

        // Deep copy
        array<dcomplex, 4> arr_dag = conj(array_view<dcomplex, 4>{idx_map_inner_transpose, M.storage()});
        return matrix<dcomplex>{idx_map, std::move(arr_dag).storage()};
      };

      // Solve the enlarged system vstack(A, A*) * x = vstack(B, B_dag)
      matrix<dcomplex> B_dag = inner_adjoint(B);
      auto [x, err]          = _lss_matrix(vstack(B, B_dag));

      // Resymmetrize results to cure small hermiticity violations
      return {0.5 * (x + inner_adjoint(x)), err};
    }
  };

} // namespace triqs::arrays::lapack

