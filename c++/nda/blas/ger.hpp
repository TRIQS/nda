#pragma once

#include "f77/cxx_interface.hpp"
#include "tools.hpp"
#include "qcache.hpp"

namespace nda::blas {

  /**
  * Calls ger : m += alpha * x * ty
  * Takes care of making temporary copies if necessary
  */
  template <typename Vx, typename Vy, typename M>
  void ger(typename Vx::value_type alpha, Vx const &x, Vy const &y, M &&m) {

    static_assert(is_regular_or_view_v<M>, "ger: Out must be a matrix or matrix_view");
    static_assert(have_same_element_type_and_it_is_blas_type_v<Vx, Vy, M>,
                  "Matrices must have the same element type and it must be double, complex ...");

    if ((first_dim(m) != y.size()) || (second_dim(m) != x.size()))
      NDA_RUNTIME_ERROR << "Dimension mismatch in ger : m : " << m.shape() << " while x : " << x.size() << " and y : " << y.size();

    auto Cx = qcache(x);
    auto Cy = qcache(y);
    auto Ca = reflexive_qcache(m);

    if constexpr (Ca().indexmap().is_stride_order_C())
      f77::ger(get_n_rows(Ca()), get_n_cols(Ca()), alpha, Cy().data_start(), Cy().stride(), Cx().data_start(), Cx().stride(), Ca().data_start(),
               get_ld(Ca()));
    else
      f77::ger(get_n_rows(Ca()), get_n_cols(Ca()), alpha, Cx().data_start(), Cx().stride(), Cy().data_start(), Cy().stride(), Ca().data_start(),
               get_ld(Ca()));
  }

} // namespace nda::blas
