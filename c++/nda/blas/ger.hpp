#pragma once

#include "f77/cxx_interface.hpp"
#include "tools.hpp"
#include "qcache.hpp"

namespace nda::blas {

  /**
  * Calls ger : m += alpha * x * ty
  * Takes care of making temporary copies if necessary
  */
  template <typename Vx, typename Vy, typename Out>
  void ger(typename Vx::value_type alpha, Vx const &x, Vy const &y, Out &&m) {

    using Out_t = std::decay_t<Out>;
    static_assert(is_regular_or_view_v<Out_t>, "ger: Out must be a matrix or matrix_view");
    static_assert(have_same_element_type_and_it_is_blas_type_v<Vx, Vy, Out_t>,
                  "Matrices must have the same element type and it must be double, complex ...");

    if ((m.extent(0) != y.size()) || (m.extent(1) != x.size()))
      NDA_RUNTIME_ERROR << "Dimension mismatch in ger : m : " << m.shape() << " while x : " << x.size() << " and y : " << y.size();

    auto Cx = qcache(x);
    auto Cy = qcache(y);
    auto Ca = reflexive_qcache(m);

    auto v = Ca();
    if constexpr (v.indexmap().is_stride_order_C())
      f77::ger(get_n_rows(Ca()), get_n_cols(Ca()), alpha, Cy().data_start(), *Cy().indexmap().strides().data(), Cx().data_start(), *Cx().indexmap().strides().data(), Ca().data_start(),
               get_ld(Ca()));
    else
      f77::ger(get_n_rows(Ca()), get_n_cols(Ca()), alpha, Cx().data_start(), *Cx().indexmap().strides().data(), Cy().data_start(), *Cy().indexmap().strides().data(), Ca().data_start(),
               get_ld(Ca()));
  }

} // namespace nda::blas
