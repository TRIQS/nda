#pragma once
#include "f77/cxx_interface.hpp"
#include "tools.hpp"
#include "qcache.hpp"

namespace nda::blas {

  namespace generic {
    // make the generic version for non lapack types or more complex types
    // largely suboptimal
    template <typename A, typename B, typename Out>
    void gemv(typename A::value_type alpha, A const &a, B const &b, typename A::value_type beta, Out &c) {
      resize_or_check_if_view(c, make_shape(a.extent(0)));
      if (a.extent(1) != b.extent(0)) NDA_RUNTIME_ERROR << "gemv generic : dimension mismatch " << a.extent(1) << " vs " << b.extent(0);
      c() = 0;
      for (int i = 0; i < a.extent(0); ++i)
        for (int k = 0; k < a.extent(1); ++k) c(i) = alpha * a(i, k) * b(k) + beta * c(i);
    }
  } // namespace generic

  /**
   * Calls gemv on a matrix or view
   * @param alpha
   * @param a
   * @param b
   * @param beta
   * @param out  Note that out can be a temporary (view)
   */
  template <typename A, typename B, typename Out>
  void gemv(typename A::value_type alpha, A const &a, B const &b, typename A::value_type beta, Out &&c) {

    using Out_t = std::decay_t<Out>;
    static_assert(is_regular_or_view_v<Out_t>, "gemm: Out must be an array  or array_view");
    static_assert(have_same_element_type_and_it_is_blas_type_v<A, B, Out_t>,
                  "Matrices/vectors must have the same element type and it must be double, complex ...");

    resize_or_check_if_view(c, make_shape(a.extent(0)));
    auto Ca = qcache(a);
    auto Cb = qcache(b);
    if (Ca().extent(1) != Cb().size()) NDA_RUNTIME_ERROR << "Dimension mismatch in gemv : A : " << Ca().shape() << " while X : " << Cb().shape();
    char trans_a = get_trans(Ca(), false);
    int m1 = get_n_rows(Ca()), m2 = get_n_cols(Ca());
    int lda = get_ld(Ca());
    f77::gemv(&trans_a, m1, m2, alpha, Ca().data_start(), lda, Cb().data_start(), *Cb().indexmap().strides().data(), beta, c.data_start(),
              *c.indexmap().strides().data());
  }

} // namespace nda::blas
