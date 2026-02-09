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
 * @brief Provides functions to get the QR factorization of a matrix.
 */

#pragma once

#include "./utils.hpp"
#include "../basic_array.hpp"
#include "../basic_functions.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../lapack/geqp3.hpp"
#include "../lapack/orgqr.hpp"
#include "../lapack/ungqr.hpp"
#include "../layout/policies.hpp"
#include "../layout/range.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <algorithm>
#include <tuple>
#include <type_traits>

namespace nda::linalg {

  /**
   * @addtogroup linalg_tools
   * @{
   */

  /**
   * @brief Get the \f$ \mathbf{Q} \f$ and \f$ \mathbf{R} \f$ matrices from the output of nda::lapack::geqp3 or 
   * nda::lapack::geqrf.
   *
   * @details \f$ \mathbf{R} \f$ is simply the upper triangular (trapezoidal) part of the \f$ m \times n \f$ matrix \f$
   * \mathbf{A} \f$ returned by nda::lapack::geqp3 or nda::lapack::geqrf.
   *
   * \f$ \mathbf{Q} \f$ is computed from the elementary reflectors stored in the lower part of \f$ \mathbf{A} \f$ and
   * the vector of scalar factors \f$ \mathbf{\tau} \f$ using nda::lapack::orgqr or nda::lapack::ungqr.
   *
   * \f$ \mathbf{Q} \f$ has dimensions \f$ m \times k \f$ and \f$ \mathbf{R} \f$ has dimensions \f$ k \times n \f$,
   * where \f$ k \f$ depends on the mode of the factorization:
   * - **reduced mode** (default): \f$ k = \min(m, n) \f$.
   * - **complete mode**: \f$ k = m \f$.
   * 
   * The resulting matrices \f$ \mathbf{Q} \f$ and \f$ \mathbf{R} \f$ are always returned in nda::F_layout.
   * 
   * An exception is thrown, if the LAPACK call fails.
   *
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space.
   *
   * @tparam A nda::blas_lapack::BlasArray<2>type.
   * @tparam TAU nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @param a Input matrix \f$ \mathbf{A} \f$ containing the output of nda::lapack::geqp3.
   * @param tau Input vector containing the scalar factors of the elementary reflectors as returned by
   * nda::lapack::geqp3.
   * @param complete If `true`, retrieves the matrices for the complete QR factorization.
   * @return A tuple containing the \f$ \mathbf{Q} \f$ and \f$ \mathbf{R} \f$ matrices.
   */
  template <blas_lapack::BlasArray<2> A, blas_lapack::BlasArrayFor<A, 1> TAU>
    requires(mem::have_host_compatible_addr_space<A, TAU>)
  auto get_qr_matrices(A const &a, TAU const &tau, bool complete = false) {
    auto const [m, n] = a.shape();
    auto const min_mn = std::min(m, n);
    auto const k      = (complete ? m : min_mn);
    auto Q            = matrix<get_value_t<A>, F_layout>::zeros(m, k);
    auto R            = matrix<get_value_t<A>, F_layout>::zeros(k, n);

    // compute Q matrix
    Q(range::all, range(min_mn)) = a(range::all, range(min_mn));
    int info{};
    if constexpr (is_complex_v<get_value_t<A>>) {
      info = lapack::ungqr(Q, tau);
    } else {
      info = lapack::orgqr(Q, tau);
    }
    if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::get_qr_matrices: orgqr/ungqr returned a non-zero value: info = " << info;

    // extract R matrix
    for (int i = 0; i < min_mn; ++i) R(range(i + 1), i) = a(range(i + 1), i);
    for (int i = min_mn; i < n; ++i) R(range::all, i) = a(range::all, i);

    return std::make_tuple(Q, R);
  }

  /**
   * @brief Compute the QR factorization of a matrix in place.
   *
   * @details The function computes the QR factorization of a general \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$ using
   * nda::lapack::geqp3 and nda::lapack::orgqr or nda::lapack::ungqr. The factorization has the form
   * \f[
   *   \mathbf{A P} = \mathbf{Q R}
   * \f]
   * where \f$ \mathbf{P} \f$ is a \f$ k \times k \f$ permutation matrix, \f$ \mathbf{Q} \f$ is an \f$ m \times k \f$
   * (orthogonal/unitary if \f$ k = m \f$) matrix, and \f$ \mathbf{R} \f$ is a \f$ k \times n \f$ upper triangular
   * (trapezoidal if \f$ k < n \f$) matrix.
   *
   * The dimension \f$ k \f$ depends on the mode of the factorization:
   * - **reduced mode** (default): \f$ k = \min(m, n) \f$.
   * - **complete mode**: \f$ k = m \f$.
   *
   * \f$ \mathbf{P} \f$ is returned as a permutation vector \f$ \mathbf{\sigma} \f$ of size \f$ n \f$. See
   * nda::linalg::get_permutation_vector for more information.
   * 
   * The resulting matrices \f$ \mathbf{Q} \f$ and \f$ \mathbf{R} \f$ are always returned in nda::F_layout.
   * 
   * An exception is thrown, if one of the LAPACK calls fails.
   *
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space and to be stored in 
   * nda::F_layout. See nda::linalg::qr for a version that handles nda::C_layout.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @param a Input/Output matrix. On entry, the \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$. On exit, the result of
   * the nda::lapack::geqp3 call.
   * @param complete If `true`, computes the complete QR factorization.
   * @returns A tuple containing \f$ \mathbf{\sigma} \f$, \f$ \mathbf{Q} \f$ and \f$ \mathbf{R} \f$.
   */
  template <blas_lapack::BlasArray<2> A>
    requires(mem::have_host_compatible_addr_space<A> and blas_lapack::has_F_layout<A>)
  auto qr_in_place(A &&a, bool complete = false) { // NOLINT (temporary views are allowed here)
    auto const [m, n] = a.shape();

    // permutation and tau vector
    auto jpvt = vector<int>::zeros(n);
    auto tau  = vector<get_value_t<A>>(std::min(m, n));

    // call lapack geqp3
    int info = lapack::geqp3(a, jpvt, tau);
    if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::qr_in_place: geqp3 returned a non-zero value: info = " << info;

    // extract Q and R from the output
    auto [Q, R] = get_qr_matrices(a, tau, complete);

    // shift permutation vector to zero-based indexing
    jpvt -= 1;

    return std::make_tuple(jpvt, Q, R);
  }

  /**
   * @brief Compute the QR factorization of a matrix.
   *
   * @details It makes a copy of the input matrix \f$ \mathbf{A} \f$ and calls nda::linalg::qr_in_place.
   *
   * The resulting matrices \f$ \mathbf{Q} \f$ and \f$ \mathbf{R} \f$ have the same layout as the input matrix \f$ 
   * \mathbf{A} \f$.
   * 
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space and to have a value type
   * that satisfies nda::is_blas_lapack_v.
   *
   * @tparam A nda::Matrix type.
   * @param a Input matrix. The \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$ to be factorized.
   * @param complete If `true`, computes the complete QR factorization. See nda::linalg::qr_in_place for details.
   * @returns A tuple containing \f$ \mathbf{\sigma} \f$, \f$ \mathbf{Q} \f$ and \f$ \mathbf{R} \f$.
   */
  template <Matrix A>
    requires(mem::have_host_compatible_addr_space<A> and is_blas_lapack_v<get_value_t<A>>)
  auto qr(A const &a, bool complete = false) {
    auto a_copy = matrix<get_value_t<A>, F_layout>(a);
    if constexpr (blas_lapack::has_F_layout<A>) {
      return qr_in_place(a_copy, complete);
    } else {
      auto [sigma, Q, R] = qr_in_place(a_copy, complete);
      return std::make_tuple(sigma, matrix<get_value_t<A>, C_layout>{Q}, matrix<get_value_t<A>, C_layout>{R});
    }
  }

  /** @} */

} // namespace nda::linalg
