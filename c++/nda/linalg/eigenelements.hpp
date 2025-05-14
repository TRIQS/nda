// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides eigenvalues and eigenvectors of a symmetric or hermitian matrix.
 */

#pragma once

#include "../basic_array.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../lapack/interface/cxx_interface.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../matrix_functions.hpp"
#include "../traits.hpp"

#include <type_traits>
#include <utility>

namespace nda::linalg {

  namespace detail {

    // Dispatch the call to the appropriate LAPACK routine based on the value type of the matrix.
    template <typename M>
    auto _eigen_element_impl(M &&m, char compz) { // NOLINT (temporary views are allowed here)
      using value_type = typename std::decay_t<M>::value_type;

      // runtime checks
      EXPECTS((not m.empty()));
      EXPECTS(is_matrix_square(m, true));
      EXPECTS(m.is_contiguous());
      EXPECTS(m.has_positive_strides());

      // set up the workspace
      int dim   = m.extent(0);
      int lwork = 64 * dim;
      array<double, 1> ev(dim);
      array<value_type, 1> work(lwork);
      array<double, 1> work2(is_complex_v<value_type> ? lwork : 0);

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
      work2 = 0;
      work  = 0;
      ev    = 0;
#endif
#endif

      // call the correct LAPACK routine
      int info = 0;
      if constexpr (not is_complex_v<value_type>) {
        lapack::f77::syev(compz, 'U', dim, m.data(), dim, ev.data(), work.data(), lwork, info);
      } else {
        lapack::f77::heev(compz, 'U', dim, m.data(), dim, ev.data(), work.data(), lwork, work2.data(), info);
      }
      if (info) NDA_RUNTIME_ERROR << "Error in nda::linalg::detail::_eigen_element_impl: Diagonalization error";
      return ev;
    }

  } // namespace detail

  /**
   * @addtogroup linalg_tools
   * @{
   */

  /**
   * @brief Find the eigenvalues and eigenvectors of a symmetric (real) or hermitian (complex) matrix/view.
   *
   * @details For a real symmetric matrix/view, it calls the LAPACK routine `syev`. For a complex hermitian matrix/view,
   * it calls the LAPACK routine `heev`.
   *
   * The given matrix/view is copied and the original is not modified.
   *
   * @tparam M Type of the matrix/view.
   * @param m Matrix/View to diagonalize.
   * @return std::pair consisting of the array of eigenvalues in ascending order and the matrix containing the
   * eigenvectors in its columns.
   */
  template <typename M>
  auto eigenelements(M const &m) {
    auto m_copy = matrix<typename M::value_type, F_layout>(m);
    auto ev     = detail::_eigen_element_impl(m_copy, 'V');
    return std::pair<array<double, 1>, typename M::regular_type>{ev, m_copy};
  }

  /**
   * @brief Find the eigenvalues of a symmetric (real) or hermitian (complex) matrix/view.
   *
   * @details For a real symmetric matrix/view, it calls the LAPACK routine `syev`. For a complex hermitian matrix/view,
   * it calls the LAPACK routine `heev`.
   *
   * The given matrix/view is copied and the original is not modified.
   *
   * @tparam M Type of the matrix/view.
   * @param m Matrix/View to diagonalize.
   * @return An nda::array of rank 1 containing the eigenvalues in ascending order.
   */
  template <typename M>
  auto eigenvalues(M const &m) {
    auto m_copy = matrix<typename M::value_type, F_layout>(m);
    return detail::_eigen_element_impl(m_copy, 'N');
  }

  /**
   * @brief Find the eigenvalues of a symmetric (real) or hermitian (complex) matrix/view.
   *
   * @details For a real symmetric matrix/view, it calls the LAPACK routine `syev`. For a complex hermitian matrix/view,
   * it calls the LAPACK routine `heev`.
   *
   * The given matrix/view will be modified by the diagonalization process.
   *
   * @tparam M Type of the matrix/view.
   * @param m Matrix/View to diagonalize.
   * @return An nda::array of rank 1 containing the eigenvalues in ascending order.
   */
  template <typename M>
  auto eigenvalues_in_place(M &m) {
    return detail::_eigen_element_impl(m, 'N');
  }

  /** @} */

} // namespace nda::linalg
