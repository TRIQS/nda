// Copyright (c) 2019-2024 Simons Foundation
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
//
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides eigenvalues and eigenvectors of a symmetric or hermitian matrix.
 */

#pragma once

#include "./det_and_inverse.hpp"
#include "../basic_array.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../lapack/interface/cxx_interface.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
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

    //--------- Generalized eigenproblem for symmetric matrices ----------
    // dispatch the implementation of invoke for T = double or complex
    template <typename M>
    auto _eigen_element_impl(M &&A, M &&B, char compz) {

      EXPECTS((not A.empty()));
      EXPECTS(is_matrix_square(A, true));
      EXPECTS(A.indexmap().is_contiguous());

      EXPECTS((not B.empty()));
      EXPECTS(is_matrix_square(B, true));
      EXPECTS(B.indexmap().is_contiguous());

      EXPECTS(A.extent(0) == B.extent(0));

      int dim = A.extent(0);

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
        lapack::f77::sygv(1, compz, 'U', dim, A.data(), dim, B.data(), dim, ev.data(), work.data(), lwork, info);
      } else {
        lapack::f77::hegv(1, compz, 'U', dim, A.data(), dim, B.data(), dim, ev.data(), work.data(), lwork, work2.data(), info);
      }
      if (info) NDA_RUNTIME_ERROR << "Diagonalization error";
      return ev;
    }

    //--------- eigen problem for nonsymmetric matrices ----------

    // dispatch the implementation of invoke for T = double or complex
    // require an additional copy for eigenvectors if compz = 'V'
    // only concern the right eigenvector v of matrix m: m * v = ev * v
    template <typename M>
    auto _geigen_element_impl(M &&m, char compz) {

      EXPECTS((not m.empty()));
      EXPECTS(is_matrix_square(m, true));
      EXPECTS(m.indexmap().is_contiguous());

      int dim = m.extent(0);

      using T = typename std::decay_t<M>::value_type;

      array<std::complex<double>, 1> ev(dim);
      auto vecs = matrix<T, F_layout>((compz=='V')? dim : 0, (compz=='V')? dim : 0);
      int lwork = 64 * dim;
      array<T, 1> work(lwork);
      array<double, 1> work2(is_complex_v<T> ? lwork : 0);

      int info = 0;
      if constexpr (not is_complex_v<T>) {
        array<double, 1> ev_r(dim);
        array<double, 1> ev_i(dim);
        lapack::f77::geev('N', compz, dim, m.data(), dim, ev_r.data(), ev_i.data(), vecs.data(), dim, vecs.data(), dim,
                          work.data(), lwork, info);
        for (long i = 0; i < dim; ++i) ev(i) = std::complex<double>(ev_r(i), ev_i(i));

      } else {
        lapack::f77::geev('N', compz, dim, m.data(), dim, ev.data(), vecs.data(), dim, vecs.data(), dim,
                          work.data(), lwork, work2.data(), info);
      }
      if (info) NDA_RUNTIME_ERROR << "Diagonalization error";
      if (compz=='V') m() = vecs;
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


  //--------------------------------

  /**
   * Find the eigenvalues and eigenvectors of a general real or complex matrix.
   * @param M The matrix or view.
   * @return Pair consisting of the array of eigenvalues and the matrix containing the eigenvectors as columns
   */
  template <typename M>
  std::pair<array<std::complex<double>, 1>, typename M::regular_type> geigenelements(M const &m) {
    auto m_copy = matrix<typename M::value_type, F_layout>(m);
    auto ev     = detail::_geigen_element_impl(m_copy, 'V');

    return {ev, m_copy};
  }

  //--------------------------------

  /**
   * Find the eigenvalues of a general complex matrix
   * Requires an additional copy
   * @param M The matrix or view.
   * @return The array of eigenvalues
   */
  template <typename M>
  array<std::complex<double>, 1> geigenvalues(M const &m) {
    auto m_copy = matrix<typename M::value_type, F_layout>(m);
    return detail::_geigen_element_impl(m_copy, 'N');
  }

  //--------------------------------

  /**
   * Find the eigenvalues of a general real or complex matrix.
   * Perform the operation in-place, avoiding a copy of the matrix,
   * but invalidating its contents.
   * @param M The matrix or view (must be contiguous and Fortran memory order)
   * @return The array of eigenvalues
   */
  template <typename M>
  array<std::complex<double>, 1> geigenvalues_in_place(M &m) {
    return detail::_geigen_element_impl(m, 'N');
  }

  //--------------------------------

  /**
   * Find the eigenvalues and eigenvectors of a generalized symmetric(real) or hermitian(complex) eigenproblem.
   * Requires an additional copy when M is stored in C memory order
   * @param M The matrix or view.
   * @return Pair consisting of the array of eigenvalues and the matrix containing the eigenvectors as columns
   */
  template <typename M>
  std::pair<array<double, 1>, typename M::regular_type> eigenelements(M const &A, M const &B) {
    auto A_copy = matrix<typename M::value_type, F_layout>(A);
    auto B_copy = matrix<typename M::value_type, F_layout>(B);
    auto ev     = detail::_eigen_element_impl(A_copy, B_copy, 'V');
    return {ev, A_copy};
  }

  //--------------------------------

  /**
   * Find the eigenvalues of a generalized symmetric(real) or hermitian(complex) eigenproblem.
   * @param M The matrix or view.
   * @return The array of eigenvalues
   */
  template <typename M>
  array<double, 1> eigenvalues(M const &A, M const& B) {
    auto A_copy = matrix<typename M::value_type, F_layout>(A);
    auto B_copy = matrix<typename M::value_type, F_layout>(B);
    return detail::_eigen_element_impl(A_copy, B_copy, 'N');
  }

  //--------------------------------

  /**
   * Find the eigenvalues of a generalized symmetric(real) or hermitian(complex) eigenproblem.
   * Perform the operation in-place, avoiding a copy of the matrix,
   * but invalidating its contents.
   * @param M The matrix or view (must be contiguous and Fortran memory order)
   * @return The array of eigenvalues
   */
  template <typename M>
  array<double, 1> eigenvalues_in_place(M *&A, M *&B) {
    return detail::_eigen_element_impl(A, B, 'N');
  }

  /** @} */

} // namespace nda::linalg
