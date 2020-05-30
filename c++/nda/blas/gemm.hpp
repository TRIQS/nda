#pragma once
#include <complex>
#include "tools.hpp"
#include "interface/cxx_interface.hpp"

namespace nda::blas {

  /**
   * Compute c <- alpha a*b + beta * c using BLAS dgemm or zgemm 
   * 
   * using a generic version for non lapack types or more complex types
   * largely suboptimal. For testing, or in case of value_type being a complex type.
   * SHOULD not be used with double and dcomplex
   *
   * \private : DO NOT DOCUMENT, testing only ??
   */
  template <CONCEPT(MatrixView) A, CONCEPT(MatrixView) B, CONCEPT(MatrixView) Out>

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
   * Compute c <- alpha a*b + beta * c using BLAS dgemm or zgemm 
   *
   * @param c Out parameter. Can be a temporary view (hence the &&).
   *         
   * @Precondition : 
   *       * c has the correct dimension given a, b. 
   *         gemm does not resize the object, 
   */
  template <CONCEPT(MatrixView) A, CONCEPT(MatrixView) B, CONCEPT(MatrixView) C>

  REQUIRES(have_same_value_type_v<A, B, C> and is_blas_lapack_v<typename A::value_type>)

  void gemm(typename A::value_type alpha, A const &a, B const &b, typename A::value_type beta, C &&c) {

    using C_t = std::decay_t<C>;

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
