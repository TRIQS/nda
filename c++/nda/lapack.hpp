#pragma once

#include <complex>
#include "lapack/f77/cxx_interface.hpp"
#include "./blas/tools.hpp"

namespace nda::lapack {

  using blas::get_ld;
  using blas::get_n_cols;
  using blas::get_n_rows;
  using blas::is_blas_lapack_v;

  // RULES :
  // - Only basic interface to lapack
  // - no copy, no cache, just EXPECTS

  // FIXME NILS : port here a minimum interface to gelss
  // as before, minimal, no qcache, just a call to lapack
  // ALSO : add a test
  // Move back the gelss_cache to TRIQS and separate the two.

  // REMOVED FROM TRIQS arrays
  // stev + its test : it is a worker, we just want here a simple interface to lapack
  // Used by Igor ?

  /**
   * LU decomposition of a matrix_view
   * The matrix m is modified during the operation.
   * The matrix is interpreted as FORTRAN ordered, and then LU decomposed.
   * NB : for some operation, like det, inversion, it is fine to be transposed, 
   *      for some it may not be ... 
   *
   * @tparam T Element type
   * @tparam L Layout
   * @param m  matrix to be LU decomposed. It is destroyed by the operation
   * @param ipiv  Gauss Pivot, cf lapack doc
   *
   */
  template <typename T, typename L>
  [[nodiscard]] int getrf(matrix_view<T, L> m, array<int, 1> &ipiv) {
    static_assert(is_blas_lapack_v<T>, "Matrices must have the same element type and it must be double, complex ...");
    auto dm = std::min(m.extent(0), m.extent(1));
    if (ipiv.size() < dm) ipiv.resize(dm);
    int info = 0;
    f77::getrf(get_n_rows(m), get_n_cols(m), m.data_start(), get_ld(m), ipiv.data_start(), info);
    return info;
  }

  /**
   * Computes the inverse of an LU-factored general matrix.
   * The matrix m is modified during the operation.
   * The matrix is interpreted as FORTRAN orderedf getrf.
   *
   * @tparam T Element type
   * @tparam L Layout
   * @param m  matrix to be LU decomposed. It is destroyed by the operation
   * @param ipiv  Gauss Pivot, cf lapack doc
   *
   */
  template <typename T, typename L>
  [[nodiscard]] int getri(matrix_view<T, L> m, array<int, 1> &ipiv) {
    static_assert(is_blas_lapack_v<T>, "Matrices must have the same element type and it must be double, complex ...");

    auto dm = std::min(m.extent(0), m.extent(1));
    EXPECTS(ipiv.size() >= dm);

    int info = 0;
    T work1[2];

    // first call to get the optimal lwork
    f77::getri(get_n_rows(m), m.data_start(), get_ld(m), ipiv.data_start(), work1, -1, info);
    int lwork;
    if constexpr (is_complex_v<T>)
      lwork = std::round(std::real(work1[0])) + 1;
    else
      lwork = std::round(work1[0]) + 1;

    array<T, 1> work(lwork);

    f77::getri(get_n_rows(m), m.data_start(), get_ld(m), ipiv.data_start(), work.data_start(), lwork, info);
    return info;
  }

  /**
   * Computes the solution to the system of linear equations with a tridiagonal coefficient matrix A and multiple right-hand sides.
   *
   * The routine solves for X the system of linear equations A*X = B, where A is an n-by-n tridiagonal matrix.
   * The columns of matrix B are individual right-hand sides, and the columns of X are the corresponding solutions. 
   *
   * @tparam T Element type
   * @param dl
   * @param d
   * @param du
   * @param b 
   */
  template <typename T>
  [[nodiscard]] int gtsv(array_view<T, 1> dl, array_view<T, 1> d, array_view<T, 1> du, matrix_view<T, F_layout> b) {

    static_assert(is_blas_lapack_v<T>, "Must be double or double complex");

    int N = first_dim(d);
    EXPECT(dl.extent(0) == d.extent(0) - 1); // "gtsv : dimension mismatch between sub-diagonal and diagonal vectors "
    EXPECT(du.extent(0) == d.extent(0) - 1); //"gtsv : dimension mismatch between super-diagonal and diagonal vectors "
    EXPECT(b.extent(0) == d.extent(0));      // "gtsv : dimension mismatch between diagonal vector and RHS matrix, "

    int info = 0;
    f77::gtsv(N, second_dim(b), dl.data_start(), d.data_start(), du.data_start(), b.data_start(), N, info);
    return info;
  }

} // namespace nda::lapack
