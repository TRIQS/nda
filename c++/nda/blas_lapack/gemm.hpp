#pragma once

#include <complex>
#include "f77/cxx_interface.hpp"
#include "tools.hpp"
#include "qcache.hpp"

namespace nda::blas {

  namespace generic {
    // make the generic version for non lapack types or more complex types
    // largely suboptimal
    template <typename A, typename B, typename Out>
    void gemm(typename A::value_type alpha, A const &a, B const &b, typename A::value_type beta, Out &c) {
      if (second_dim(a) != first_dim(b)) TRIQS_RUNTIME_ERROR << "gemm generic : dimension mismatch " << get_shape(a) << get_shape(b);
      resize_or_check_if_view(c, make_shape(first_dim(a), second_dim(b)));
      c() = 0;
      for (int i = 0; i < first_dim(a); ++i)
        for (int k = 0; k < second_dim(a); ++k)
          for (int j = 0; j < second_dim(b); ++j) c(i, j) += a(i, k) * b(k, j);
    }
  } // namespace generic

  /**
   * Calls gemm on a matrix or view
   * @param alpha
   * @param a
   * @param b
   * @param beta
   * @param out  Note that out can be a temporary (view)
   */
  template <typename A, typename B, typename Out>
  void gemm(typename A::value_type alpha, A const &a, B const &b, typename A::value_type beta, Out &&c) {

    static_assert(is_regular_or_view_v<Out>, "gemm: Out must be a matrix or matrix_view");
    static_assert(have_same_element_type_and_it_is_blas_type_v<A, B, Out>,
                  "Matrices must have the same element type and it must be double, complex ...");

    if (second_dim(a) != first_dim(b)) NDA_RUNTIME_ERROR << "Dimension mismatch in gemm : a : " << second_dim(a) << " while b : " << first_dim(b);

    // first resize if necessary and possible
    resize_or_check_if_view(c, make_shape(first_dim(a), second_dim(b)));

    // now we use qcache instead of the matrix to make a copy if necessary ...
    // not optimal : if stride == 1, N ---> use LDA parameters
    // change the condition in the qcache construction....
    auto Cc = reflexive_qcache(c);

    if constexpr (Cc().indexmap().is_stride_order_C()) {
      // then tC = tB tA !
      auto Cb = qcache(a); // note the inversion  a <-> b
      auto Ca = qcache(b); // note the inversion  a <-> b
      char trans_a = get_trans(Ca(), true);
      char trans_b = get_trans(Cb(), true);
      int m        = (trans_a == 'N' ? get_n_rows(Ca()) : get_n_cols(Ca()));
      int n        = (trans_b == 'N' ? get_n_cols(Cb()) : get_n_rows(Cb()));
      int k        = (trans_a == 'N' ? get_n_cols(Ca()) : get_n_rows(Ca()));
      f77::gemm(trans_a, trans_b, m, n, k, alpha, Ca().data_start(), get_ld(Ca()), Cb().data_start(), get_ld(Cb()), beta, Cc().data_start(),
                get_ld(Cc()));
    } else {
      auto Ca = qcache(a);
      auto Cb = qcache(b);
      char trans_a = get_trans(Ca(), false);
      char trans_b = get_trans(Cb(), false);
      int m        = (trans_a == 'N' ? get_n_rows(Ca()) : get_n_cols(Ca()));
      int n        = (trans_b == 'N' ? get_n_cols(Cb()) : get_n_rows(Cb()));
      int k        = (trans_a == 'N' ? get_n_cols(Ca()) : get_n_rows(Ca()));
      f77::gemm(trans_a, trans_b, m, n, k, alpha, Ca().data_start(), get_ld(Ca()), Cb().data_start(), get_ld(Cb()), beta, Cc().data_start(),
                get_ld(Cc()));
    }
  }

} // namespace nda::blas
