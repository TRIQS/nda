/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2014 by O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include "../lapack/interface/lapack_cxx_interface.hpp"

namespace nda::linalg {

  template <typename M>
  // dispatch the implementation of invoke for T = double or complex
  auto _eigen_element_impl(M &&m, char compz) {

    EXPECTS((not m.is_empty()));
    EXPECTS(is_matrix_square(m, true));
    EXPECTS(m.indexmap().is_contiguous());

    int dim = first_dim(m);

    using T = typename std::decay_t<M>::value_type;

    array<double, 1> ev(dim);
    int lwork = 64 * dim;
    array<T, 1> work(lwork);
    array<double, 1> work2(is_complex_v<T> ? lwork : 0);

    int info = 0;
    if constexpr (not is_complex_v<T>) {
      lapack::f77::dsyev(compz, 'U', dim, m.data_start(), dim, ev.data_start(), work.data_start(), lwork, info);
    } else {
      lapack::f77::zheev(compz, 'U', dim, m.data_start(), dim, ev.data_start(), work.data_start(), lwork, work2.data_start(), info);
    }
    if (info) NDA_RUNTIME_ERROR << "Diagonalization error";
    return ev;
  }

  //--------------------------------

  /**
   * Simple diagonalization call, return all eigenelements.
   * Handles both real and complex case.
   * @param M : the matrix or view.
   */
  template <typename M>
  std::pair<array<double, 1>, matrix<typename M::value_type>> eigenelements(M const &m) {
    auto m_copy = make_regular(m);
    auto ev     = _eigen_element_impl(m_copy, 'V');
    if constexpr (is_complex_v<typename M::value_type>) {
      if constexpr (M::is_stride_order_C()) {
        return {ev, conj(m_copy)};
      } else {
        return {ev, m_copy.transpose()}; // the matrix mat is understood as a fortran matrix. After the lapack, in memory, it contains the
                                         // correct answer.
                                         // but since it is a fortran matrix, the C will see its transpose. We need to compensate this transpose (!).
      }
    } else {
      return {ev, m_copy};
    }
  }

  //--------------------------------

  /**
   * Simple diagonalization call, returning only the eigenvalues.
   * Handles both real and complex case.
   * @param M : the matrix VIEW : it MUST be contiguous
   */
  template <typename M>
  array<double, 1> eigenvalues(M const &m) {
    auto m_copy = make_regular(m);
    return _eigen_element_impl(m_copy, 'N');
  }

  //--------------------------------

  /**
   * Simple diagonalization call, returning only the eigenvalues.
   * Handles both real and complex case.
   * @param M : the matrix VIEW : it MUST be contiguous
   */
  template <typename M>
  array<double, 1> eigenvalues_in_place(M *&m) {
    return _eigen_element_impl(m, 'N');
  }

} // namespace nda::linalg
