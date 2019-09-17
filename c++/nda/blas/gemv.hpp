#pragma once
#include "f77/cxx_interface.hpp"
#include "tools.hpp"
#include "qcache.hpp"

namespace nda::blas {

  namespace generic {
    // make the generic version for non lapack types or more complex types
    // largely suboptimal
    template <typename A, typename B, typename Out>
    template <typename MT, typename VT, typename VTOut>
    void gemv(typename A::value_type alpha, A const &a, B const &b, typename A::value_type beta, Out &c) {
      resize_or_check_if_view(C, make_shape(first_dim(A)));
      if (second_dim(A) != X.size()) NDA_RUNTIME_ERROR << "gemm generic : dimension mismatch " << second_dim(A) << " vs " << X.size();
      C() = 0;
      for (int i = 0; i < first_dim(A); ++i)
        for (int k = 0; k < second_dim(A); ++k) C(i) += A(i, k) * X(k);
    }
  } // namespace generic

  using namespace blas_lapack_tools;

  template <typename MT, typename VT, typename VTOut>
  struct use_blas_gemv {
    static_assert(is_amv_value_or_view_class<VTOut>::value, "output of matrix product must be a matrix or matrix_view");
    //static constexpr bool are_both_value_view = is_amv_value_or_view_class<MT>::value && is_amv_value_or_view_class<VT>::value;
    //static constexpr bool value = are_both_value_view && is_blas_lapack_type<typename MT::value_type>::value && have_same_value_type< MT, VT, VTOut>::value;
    static constexpr bool value = is_blas_lapack_type<typename MT::value_type>::value && have_same_value_type<MT, VT, VTOut>::value;
    // cf gemm comment
  };

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
    static_assert(is_regular_or_view_v<Out_t>, "gemm: Out must be an array  or array_view");
    static_assert(have_same_element_type_and_it_is_blas_type_v<A, B, Out_t>,
                  "Matrices/vectors must have the same element type and it must be double, complex ...");

    resize_or_check_if_view(c, make_shape(a.extent(0)));
    auto Ca = qcache(a);
    auto Cb = qcache(b);
    if (Ca().extent(1) != Cb().size())
      NDA_RUNTIME_ERROR << "Dimension mismatch in gemv : A : " << get_shape(Ca()) << " while X : " << get_shape(Cb());
    char trans_a = get_trans(Ca(), false);
    int m1 = get_n_rows(Ca()), m2 = get_n_cols(Ca());
    int lda = get_ld(Ca());
    f77::gemv(&trans_a, m1, m2, alpha, Ca().data_start(), lda, Cb().data_start(), Cb().stride(), beta, c.data_start(), c.stride());
  }

} // namespace nda::blas
