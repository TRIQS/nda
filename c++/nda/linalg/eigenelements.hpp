// Copyright (c) 2019-2021 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell

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
   * Find the eigenvalues and eigenvectors of a symmetric(real) or hermitian(complex) matrix.
   * Requires an additional copy when M is stored in C memory order
   * @param M The matrix or view.
   * @return Pair consisting of the array of eigenvalues and the matrix containing the eigenvectors as columns
   */
  template <typename M>
  std::pair<array<double, 1>, typename M::regular_type> eigenelements(M const &m) {
    auto m_copy = matrix<typename M::value_type, F_layout>(m);
    auto ev     = _eigen_element_impl(m_copy, 'V');
    return {ev, m_copy};
  }

  //--------------------------------

  /**
   * Find the eigenvalues of a symmetric(real) or hermitian(complex) matrix.
   * @param M The matrix or view.
   * @return The array of eigenvalues
   */
  template <typename M>
  array<double, 1> eigenvalues(M const &m) {
    auto m_copy = matrix<typename M::value_type, F_layout>(m);
    return _eigen_element_impl(m_copy, 'N');
  }

  //--------------------------------

  /**
   * Find the eigenvalues of a symmetric(real) or hermitian(complex) matrix.
   * Perform the operation in-place, avoiding a copy of the matrix,
   * but invalidating its contents.
   * @param M The matrix or view (must be contiguous and Fortran memory order)
   * @return The array of eigenvalues
   */
  template <typename M>
  array<double, 1> eigenvalues_in_place(M *&m) {
    return _eigen_element_impl(m, 'N');
  }

  //--------------------------------

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

  /**
   * Find the eigenvalues and eigenvectors of a general real or complex matrix.
   * @param M The matrix or view.
   * @return Pair consisting of the array of eigenvalues and the matrix containing the eigenvectors as columns
   */
  template <typename M>
  std::pair<array<std::complex<double>, 1>, typename M::regular_type> geigenelements(M const &m) {
    auto m_copy = matrix<typename M::value_type, F_layout>(m);
    auto ev     = _geigen_element_impl(m_copy, 'V');

    return {ev, m_copy};
  }

  /**
   * Find the eigenvalues of a general complex matrix
   * Requires an additional copy
   * @param M The matrix or view.
   * @return The array of eigenvalues
   */
  template <typename M>
  array<std::complex<double>, 1> geigenvalues(M const &m) {
    auto m_copy = matrix<typename M::value_type, F_layout>(m);
    return _geigen_element_impl(m_copy, 'N');
  }

  /**
   * Find the eigenvalues of a general real or complex matrix.
   * Perform the operation in-place, avoiding a copy of the matrix,
   * but invalidating its contents.
   * @param M The matrix or view (must be contiguous and Fortran memory order)
   * @return The array of eigenvalues
   */
  template <typename M>
  array<std::complex<double>, 1> geigenvalues_in_place(M *&m) {
    return _geigen_element_impl(m, 'N');
  }


} // namespace nda::linalg
