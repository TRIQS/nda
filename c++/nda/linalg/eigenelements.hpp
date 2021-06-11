// Copyright (c) 2019-2020 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "../lapack/interface/lapack_cxx_interface.hpp"

namespace nda::linalg {

  template <typename M>
  // dispatch the implementation of invoke for T = double or complex
  auto _eigen_element_impl(M &&m, char compz) {

    EXPECTS((not m.empty()));
    EXPECTS(is_matrix_square(m, true));
    EXPECTS(m.indexmap().is_contiguous());

    int dim = m.extent(0);

    using T = typename std::decay_t<M>::value_type;

    array<double, 1> ev(dim);
    int lwork = 64 * dim;
    array<T, 1> work(lwork);
    array<double, 1> work2(is_complex_v<T> ? lwork : 0);

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    work2 = 0;
    work  = 0;
    ev    = 0;
#endif
#endif

    int info = 0;
    if constexpr (not is_complex_v<T>) {
      lapack::f77::syev(compz, 'U', dim, m.data(), dim, ev.data(), work.data(), lwork, info);
    } else {
      lapack::f77::heev(compz, 'U', dim, m.data(), dim, ev.data(), work.data(), lwork, work2.data(), info);
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
