#pragma once

#include <complex>
#include "tools.hpp"
#include "blas_interface/cxx_interface.hpp"

namespace nda::blas {

  // make the generic version for non lapack types or more complex types
  // largely suboptimal
  template <typename A, typename B, typename Out>
  void gemm_generic(typename A::value_type alpha, A const &a, B const &b, typename A::value_type beta, Out &c) {

    EXPECTS(a.extent(1) == b.extent(0));
    EXPECTS(a.extent(0) == c.extent(0));
    EXPECTS(b.extent(1) == c.extent(1));

    c() = 0;
    for (int i = 0; i < a.extent(0); ++i)
      for (int j = 0; j < b.extent(1); ++j) {
        typename A::value_type acc = 0;
        for (int k = 0; k < a.extent(1); ++k) acc += alpha * a(i, k) * b(k, j);
        c(i, j) = acc + beta * c(i, j);
      }
  }

  /**
   * Calls gemm on a matrix, matrix_view, array, array_view of rank 2
   * to compute c <- alpha a*b + beta * c
   *
   * @tparam A matrix, matrix_view, array, array_view of rank 2
   * @tparam B matrix, matrix_view, array, array_view of rank 2
   * @tparam C matrix, matrix_view, array, array_view of rank 2
   * @param alpha
   * @param a 
   * @param b
   * @param beta
   * @param c The result. Can be a temporary view. 
   *         
   * @StaticPrecondition : A, B, C have the same value_type and it is complex<double> or double         
   * @Precondition : 
   *       * c has the correct dimension given a, b. 
   *         gemm does not resize the object, 
   */
  template <typename A, typename B, typename C>
  void gemm(typename A::value_type alpha, A const &a, B const &b, typename A::value_type beta, C &&c) {

    using C_t = std::decay_t<C>;
    static_assert(is_regular_or_view_v<C_t>, "gemm: Out must be a matrix, matrix_view, array or array_view of rank 2");
    static_assert(A::rank == 2, "A must be of rank 2");
    static_assert(B::rank == 2, "B must be of rank 2");
    static_assert(C_t::rank == 2, "C must be of rank 2");
    static_assert(have_same_element_type_and_it_is_blas_type_v<A, B, C_t>,
                  "Matrices must have the same element type and it must be double, complex ...");

    EXPECTS(a.extent(1) == b.extent(0));
    EXPECTS(a.extent(0) == c.extent(0));
    EXPECTS(b.extent(1) == c.extent(1));

    // Must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(c.indexmap().min_stride() == 1);

    // We need to see if C is in Fortran order or C order
    if constexpr (C_t::is_stride_order_C()) {
      // C order. We compute the transpose of the product in this case
      // since BLAS is in Fortran order
      char trans_a = get_trans(b, true);
      char trans_b = get_trans(a, true);
      int m        = (trans_a == 'N' ? get_n_rows(b) : get_n_cols(b));
      int n        = (trans_b == 'N' ? get_n_cols(a) : get_n_rows(a));
      int k        = (trans_a == 'N' ? get_n_cols(b) : get_n_rows(b));
      f77::gemm(trans_a, trans_b, m, n, k, alpha, b.data_start(), get_ld(b), a.data_start(), get_ld(a), beta, c.data_start(), get_ld(c));
    } else {
      // C is in fortran or, we compute the product.
      char trans_a = get_trans(a, false);
      char trans_b = get_trans(b, false);
      int m        = (trans_a == 'N' ? get_n_rows(a) : get_n_cols(a));
      int n        = (trans_b == 'N' ? get_n_cols(b) : get_n_rows(b));
      int k        = (trans_a == 'N' ? get_n_cols(a) : get_n_rows(a));
      f77::gemm(trans_a, trans_b, m, n, k, alpha, a.data_start(), get_ld(a), b.data_start(), get_ld(b), beta, c.data_start(), get_ld(c));
    }
  }

} // namespace nda::blas
