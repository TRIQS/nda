#pragma once

#include <complex>
#include "lapack/interface/lapack_cxx_interface.hpp"
#include "blas/tools.hpp"

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
   * LU decomposition of a matrix
   *
   * The matrix m is modified during the operation.
   * The matrix is interpreted as FORTRAN ordered, and then LU decomposed.
   * NB : for some operation, like det, inversion, it is fine to be transposed, 
   *      for some it may not be ... 
   *
   * @tparam M matrix, matrix_view, array, array_view of rank 2. M can be a temporary view
   * @param m  matrix to be LU decomposed. It is destroyed by the operation
   * @param ipiv  Gauss Pivot, cf lapack doc
   *
   */
  template <typename M>
  [[nodiscard]] int getrf(M &&m, array<int, 1> &ipiv) {
    using M_t = std::decay_t<M>;
    static_assert(is_regular_or_view_v<M_t>, "getrf: M must be a matrix, matrix_view, array or array_view of rank 2");
    static_assert(M_t::rank == 2, "M must be of rank 2");
    static_assert(is_blas_lapack_v<typename M_t::value_type>, "Matrices must have the same element type and it must be double, complex ...");

    auto dm = std::min(m.extent(0), m.extent(1));
    if (ipiv.size() < dm) ipiv.resize(dm);

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    ipiv = 0;
#endif
#endif

    int info = 0;
    f77::getrf(get_n_rows(m), get_n_cols(m), m.data_start(), get_ld(m), ipiv.data_start(), info);
    return info;
  }

  /**
   * Computes the inverse of an LU-factored general matrix.
   * The matrix m is modified during the operation.
   * The matrix is interpreted as FORTRAN orderedf getrf.
   *
   * @tparam M matrix, matrix_view, array, array_view of rank 2. M can be a temporary view
   * @param m  matrix to be LU decomposed. It is destroyed by the operation
   * @param ipiv  Gauss Pivot, cf lapack doc
   *
   */
  template <typename M>
  [[nodiscard]] int getri(M &&m, array<int, 1> &ipiv) {
    using M_t = std::decay_t<M>;
    static_assert(is_regular_or_view_v<M_t>, "getrf: M must be a matrix, matrix_view, array or array_view of rank 2");
    static_assert(M_t::rank == 2, "M must be of rank 2");
    static_assert(is_blas_lapack_v<typename M_t::value_type>, "Matrices must have the same element type and it must be double, complex ...");

    auto dm = std::min(m.extent(0), m.extent(1));
    EXPECTS(ipiv.size() >= dm);

    using T  = typename M_t::value_type;
    int info = 0;
    std::array<T, 2> work1{0, 0}; // always init for MSAN and clang-tidy ...

    // first call to get the optimal lwork
    f77::getri(get_n_rows(m), m.data_start(), get_ld(m), ipiv.data_start(), work1.data(), -1, info);
    int lwork;
    if constexpr (is_complex_v<T>)
      lwork = std::round(std::real(work1[0])) + 1;
    else
      lwork = std::round(work1[0]) + 1;

    array<T, 1> work(lwork);

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    work = 0;
#endif
#endif

    // second call to do the job
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
  template <typename V1, typename V2, typename V3, typename M>
  [[nodiscard]] int gtsv(V1 &dl, V2 &d, V3 &du, M &b) {

    static_assert(is_regular_or_view_v<V1> and (V1::rank == 1), "gtsv: V1 must be an array/view of rank 1");
    static_assert(is_regular_or_view_v<V2> and (V2::rank == 1), "gtsv: V2 must be an array/view of rank 1");
    static_assert(is_regular_or_view_v<V3> and (V3::rank == 1), "gtsv: V3 must be an array/view of rank 1");
    //   static_assert(is_regular_or_view_v<M> and (M::rank == 2), "gtsv: M must be an matrix/array/view of rank 2");
    static_assert(is_regular_or_view_v<M>, "gtsv: M must be an matrix/array/view of rank  1 or 2");
    static_assert(is_blas_lapack_v<typename M::value_type>, "Matrices must have the same element type and it must be double, complex ...");
    static_assert(blas::have_same_element_type_and_it_is_blas_type_v<V1, V2, V3, M>,
                  "All arguments must have the same element type and it must be double, complex ...");

    int N = d.extent(0);
    EXPECTS(dl.extent(0) == d.extent(0) - 1); // "gtsv : dimension mismatch between sub-diagonal and diagonal vectors "
    EXPECTS(du.extent(0) == d.extent(0) - 1); // "gtsv : dimension mismatch between super-diagonal and diagonal vectors "
    EXPECTS(b.extent(0) == d.extent(0));      // "gtsv : dimension mismatch between diagonal vector and RHS matrix, "

    int info = 0;
    f77::gtsv(N, b.extent(1), dl.data_start(), d.data_start(), du.data_start(), b.data_start(), N, info);
    return info;
  }

} // namespace nda::lapack
