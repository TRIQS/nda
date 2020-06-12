#pragma once

namespace nda::lapack {

  ///
  ///  $$ A = U S {}^t V$$
  /// A is destroyed during the computation
  template <CONCEPT(MatrixView) A, CONCEPT(MatrixView) U, CONCEPT(MatrixView) V>

  REQUIRES(have_same_value_type_v<A, U, V> and is_blas_lapack_v<typename A::value_type>)

  int gesvd1(A &a, array_view<double, 1> c, U &u, V &v) {

    static_assert(A::layout_t::is_stride_order_Fortran(), "C order not implemented");
    static_assert(U::layout_t::is_stride_order_Fortran(), "C order not implemented");
    static_assert(V::layout_t::is_stride_order_Fortran(), "C order not implemented");

    int info = 0;

    using T = typename A::value_type;
    static_assert(IsDoubleOrComplex<T>, "Not implemented");

    if constexpr (std::is_same_v<T, double>) {

      // first call to get the optimal lwork
      T work1[1];
      lapack::f77::gesvd('A', 'A', get_n_rows(a), get_n_cols(a), a.data_start(), get_ld(a), c.data_start(), u.data_start(), get_ld(u), v.data_start(),
                         get_ld(v), work1, -1, info);

      int lwork = std::round(work1[0]) + 1;
      array<T, 1> work(lwork);

      lapack::f77::gesvd('A', 'A', get_n_rows(a), get_n_cols(a), a.data_start(), get_ld(a), c.data_start(), u.data_start(), get_ld(u), v.data_start(),
                         get_ld(v), work.data_start(), lwork, info);

    } else {

      auto rwork = array<double, 1>(5 * std::min(a.extent(0), a.extent(1)));

      // first call to get the optimal lwork
      T work1[1];
      lapack::f77::gesvd('A', 'A', get_n_rows(a), get_n_cols(a), a.data_start(), get_ld(a), c.data_start(), u.data_start(), get_ld(u), v.data_start(),
                         get_ld(v), work1, -1, rwork.data_start(), info);

      int lwork = std::round(std::real(work1[0])) + 1;
      array<T, 1> work(lwork);

      lapack::f77::gesvd('A', 'A', get_n_rows(a), get_n_cols(a), a.data_start(), get_ld(a), c.data_start(), u.data_start(), get_ld(u), v.data_start(),
                         get_ld(v), work.data_start(), lwork, rwork.data_start(), info);
    }

    if (info) NDA_RUNTIME_ERROR << "Error in gesvd : info = " << info;
    return info;
  }

  inline int gesvd(matrix_view<double, F_layout> a, array_view<double, 1> c, matrix_view<double, F_layout> u, matrix_view<double, F_layout> v) {
    return gesvd1(a, c, u, v);
  }

  inline int gesvd(matrix_view<dcomplex, F_layout> a, array_view<double, 1> c, matrix_view<dcomplex, F_layout> u, matrix_view<dcomplex, F_layout> v) {
    return gesvd1(a, c, u, v);
  }

} // namespace nda::lapack
