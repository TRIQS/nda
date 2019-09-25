#pragma once

#include "tools.hpp"
#include "blas_interface/cxx_interface.hpp"

namespace nda::blas {

  // make the generic version for non lapack types or more complex types
  // largely suboptimal
  template <typename A, typename B, typename Out>
  void gemv_generic(typename A::value_type alpha, A const &a, B const &b, typename A::value_type beta, Out &c) {
    EXPECTS(a.extent(1) == b.extent(0));
    EXPECTS(a.extent(0) == c.extent(0));
    c() = 0;
    for (int i = 0; i < a.extent(0); ++i)
      for (int k = 0; k < a.extent(1); ++k) c(i) = alpha * a(i, k) * b(k) + beta * c(i);
  }

  /**
   * Calls gemv on a matrix, matrix_view, array, array_view of rank 2
   * to compute c <- alpha a*b + beta * c
   *
   * @tparam A matrix, matrix_view, array, array_view of rank 2
   * @tparam B matrix, matrix_view, array, array_view of rank 1
   * @tparam C matrix, matrix_view, array, array_view of rank 1
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
   *
   *
   */
  template <typename A, typename B, typename C>
  void gemv(typename A::value_type alpha, A const &a, B const &b, typename A::value_type beta, C &&c) {

    using Out_t = std::decay_t<C>;
    static_assert(is_regular_or_view_v<Out_t>, "gemm: Out must be a matrix, matrix_view, array or array_view of rank 2");
    static_assert(A::rank == 2, "A must be of rank 2");
    static_assert(B::rank == 1, "B must be of rank 1");
    static_assert(Out_t::rank == 1, "C must be of rank 1");
    static_assert(have_same_element_type_and_it_is_blas_type_v<A, B, Out_t>,
                  "Matrices/vectors must have the same element type and it must be double, complex ...");

    EXPECTS(a.extent(1) == b.extent(0));
    EXPECTS(a.extent(0) == c.extent(0));

    char trans_a = get_trans(a, false);
    int m1       = get_n_rows(a);
    int m2       = get_n_cols(a);
    int lda      = get_ld(a);
    f77::gemv(trans_a, m1, m2, alpha, a.data_start(), lda, b.data_start(), b.indexmap().strides()[0], beta, c.data_start(),
              c.indexmap().strides()[0]);
  }

} // namespace nda::blas
