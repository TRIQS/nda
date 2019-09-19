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
      if (a.extent(1) != b.extent(0)) NDA_RUNTIME_ERROR << "gemm generic : dimension mismatch " << a.extent(1) << b.extent(0);
      resize_or_check_if_view(c, make_shape(a.extent(0), b.extent(1)));
      c() = 0;
      for (int i = 0; i < a.extent(0); ++i)
        for (int k = 0; k < a.extent(1); ++k)
          for (int j = 0; j < b.extent(1); ++j) c(i, j) = alpha * a(i, k) * b(k, j) + beta * c(i, j);
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

    using Out_t = std::decay_t<Out>;
    static_assert(is_regular_or_view_v<Out_t>, "gemm: Out must be a matrix or matrix_view");
    static_assert(have_same_element_type_and_it_is_blas_type_v<A, B, Out_t>,
                  "Matrices must have the same element type and it must be double, complex ...");

    if (a.extent(1) != b.extent(0)) NDA_RUNTIME_ERROR << "Dimension mismatch in gemm : a : " << a.extent(1) << " while b : " << b.extent(0);

    // first resize if necessary and possible
    resize_or_check_if_view(c, make_shape(a.extent(0), b.extent(1)));

    // now we use qcache instead of the matrix to make a copy if necessary ...
    // not optimal : if stride == 1, N ---> use LDA parameters
    // change the condition in the qcache construction....
    auto Cc = reflexive_qcache(c);

    auto v = Cc(); // FIXME : why can't I use v directly in the next constexpr ?

    if constexpr (v.indexmap().is_stride_order_C()) {
      //if (Cc().indexmap().is_stride_order_C()) {
      // then tC = tB tA !
      auto Cb      = qcache(a); // note the inversion  a <-> b
      auto Ca      = qcache(b); // note the inversion  a <-> b
      char trans_a = get_trans(Ca(), true);
      char trans_b = get_trans(Cb(), true);
      int m        = (trans_a == 'N' ? get_n_rows(Ca()) : get_n_cols(Ca()));
      int n        = (trans_b == 'N' ? get_n_cols(Cb()) : get_n_rows(Cb()));
      int k        = (trans_a == 'N' ? get_n_cols(Ca()) : get_n_rows(Ca()));
      f77::gemm(trans_a, trans_b, m, n, k, alpha, Ca().data_start(), get_ld(Ca()), Cb().data_start(), get_ld(Cb()), beta, Cc().data_start(),
                get_ld(Cc()));
    } else {
      auto Ca      = qcache(a);
      auto Cb      = qcache(b);
      char trans_a = get_trans(Ca(), false);
      char trans_b = get_trans(Cb(), false);
      int m        = (trans_a == 'N' ? get_n_rows(Ca()) : get_n_cols(Ca()));
      int n        = (trans_b == 'N' ? get_n_cols(Cb()) : get_n_rows(Cb()));
      int k        = (trans_a == 'N' ? get_n_cols(Ca()) : get_n_rows(Ca()));
      //NDA_PRINT(trans_a);
      //NDA_PRINT(trans_b);
      //NDA_PRINT(m);
      //NDA_PRINT(n);
      //NDA_PRINT(k);
      //NDA_PRINT(get_ld(Ca()));
      //NDA_PRINT(get_ld(Cb()));
      //NDA_PRINT(Ca().indexmap());
      //NDA_PRINT(Cb().indexmap());
      //NDA_PRINT(Cc().indexmap());
      //NDA_PRINT(Ca());
      //NDA_PRINT(Cb());
      //NDA_PRINT(Cb().indexmap().is_stride_order_Fortran());

      f77::gemm(trans_a, trans_b, m, n, k, alpha, Ca().data_start(), get_ld(Ca()), Cb().data_start(), get_ld(Cb()), beta, Cc().data_start(),
                get_ld(Cc()));
      //NDA_PRINT(c);
    }
  }

} // namespace nda::blas
