#pragma once

#include <complex>
#include "f77/cxx_interface.hpp"
#include "tools.hpp"
#include "qcache.hpp"

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
   * Calls gemm on a matrix or view
   * to compute c <- alpha a*b + beta * c
   * @param alpha
   * @param a
   * @param b
   * @param beta
   * @param c  Note that c can be a temporary (view)
   */
  template <typename A, typename B, typename Out>
  void gemm(typename A::value_type alpha, A const &a, B const &b, typename A::value_type beta, Out &&c) {

    using Out_t = std::decay_t<Out>;
    static_assert(is_regular_or_view_v<Out_t>, "gemm: Out must be a matrix or matrix_view");
    static_assert(have_same_element_type_and_it_is_blas_type_v<A, B, Out_t>,
                  "Matrices must have the same element type and it must be double, complex ...");

    EXPECTS(a.extent(1) == b.extent(0));
    EXPECTS(a.extent(0) == c.extent(0));
    EXPECTS(b.extent(1) == c.extent(1));

    // Must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(c.indexmap().min_stride() == 1);

    // For MSAN : we avoid to recompile the lapack/blas with MSAN which can be tricky and set to 0 here.
    // of course, in production code, we do NOT waste time to do this.
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    c = 0;
#endif
#endif

    auto idx = c.indexmap();
    if constexpr (idx.is_stride_order_C()) {
      // we compute the product of the transpose in this case
      // since BLAS is in Fortran order
      char trans_a = get_trans(b, true);
      char trans_b = get_trans(a, true);
      int m        = (trans_a == 'N' ? get_n_rows(b) : get_n_cols(b));
      int n        = (trans_b == 'N' ? get_n_cols(a) : get_n_rows(a));
      int k        = (trans_a == 'N' ? get_n_cols(b) : get_n_rows(b));
      f77::gemm(trans_a, trans_b, m, n, k, alpha, b.data_start(), get_ld(b), a.data_start(), get_ld(a), beta, c.data_start(), get_ld(c));
    } else {
      auto Ca      = qcache(a);
      auto Cb      = qcache(b);
      char trans_a = get_trans(a, false);
      char trans_b = get_trans(b, false);
      int m        = (trans_a == 'N' ? get_n_rows(a) : get_n_cols(a));
      int n        = (trans_b == 'N' ? get_n_cols(b) : get_n_rows(b));
      int k        = (trans_a == 'N' ? get_n_cols(a) : get_n_rows(a));
      f77::gemm(trans_a, trans_b, m, n, k, alpha, a.data_start(), get_ld(a), b.data_start(), get_ld(b), beta, c.data_start(), get_ld(c));
    }
  }

} // namespace nda::blas
