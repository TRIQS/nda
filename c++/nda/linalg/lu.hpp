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
 * @brief Provides functions to get the LU factorization of a matrix.
 */

#pragma once

#include "./utils.hpp"
#include "../basic_array.hpp"
#include "../basic_functions.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../lapack/getrf.hpp"
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
   * @brief Get the \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ matrices from the output of nda::lapack::getrf.
   *
   * @details Let the original matrix \f$ \mathbf{A} \f$ be of size \f$ m \times n \f$. Then the returned \f$ \mathbf{L}
   * \f$ matrix is of size \f$ m \times k \f$ and \f$ \mathbf{U} \f$ is of size \f$ k \times n \f$, where \f$ k =
   * \min(m, n) \f$.
   * 
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space and to be stored in 
   * nda::F_layout.
   *
   * @tparam LP Policy determining the memory layout of the \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ matrices.
   * @tparam M nda::MemoryMatrix type.
   * @param a Input matrix containing the output of nda::lapack::getrf.
   * @return A tuple containing the \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ matrices.
   */
  template <typename LP = F_layout, MemoryMatrix A>
    requires(mem::have_host_compatible_addr_space<A> and blas_lapack::has_F_layout<A>)
  auto get_lu_matrices(A const &a) {
    using output_t = matrix<get_value_t<A>, LP>;

    // copy the first k columns from A to L and U
    auto const [m, n] = a.shape();
    auto const k      = std::min(m, n);
    auto L            = output_t::zeros(m, k);
    auto U            = output_t::zeros(k, n);
    for (int i = 0; i < k; ++i) {
      L(i, i)               = get_value_t<A>{1};
      L(range(i + 1, m), i) = a(range(i + 1, m), i);
      U(range(i + 1), i)    = a(range(i + 1), i);
    }

    // in case of n > m, copy the remaining columns to U
    for (int i = k; i < n; ++i) U(range::all, i) = a(range::all, i);

    return std::make_tuple(L, U);
  }

  /**
   * @brief Compute the LU factorization of a matrix in place.
   *
   * @details The function computes the LU factorization of a general \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$ using
   * nda::lapack::getrf. The factorization has the form
   * \f[
   *   \mathbf{P A} = \mathbf{L U}
   * \f]
   * where \f$ \mathbf{P} \f$ is an \f$ m \times m \f$ permutation matrix, \f$ \mathbf{L} \f$ is an \f$ m \times k \f$
   * lower triangular (trapezoidal if \f$ m > n \f$) matrix with unit diagonal elements, and \f$ \mathbf{U} \f$ is a \f$
   * k \times n \f$ upper triangular (trapezoidal if \f$ m < n \f$) matrix. Here, \f$ k = \min(m, n) \f$.
   *
   * \f$ \mathbf{P} \f$ is returned as a permutation vector \f$ \mathbf{\sigma} \f$ of size \f$ m \f$. See
   * nda::linalg::get_permutation_vector for more information.
   * 
   * An exception is thrown, if the LAPACK call returns 
   * - a non-zero value (`allow_singular == false`) or
   * - a value \f$ > 0 \f$ (`allow_singular == true`).
   *
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space and to be stored in 
   * nda::F_layout. See nda::linalg::lu for a version that handles nda::C_layout.
   *
   * @tparam LP Policy determining the memory layout of the \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ matrices.
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @param a Input/Output matrix. On entry, the \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$. On exit, the result of 
   * the nda::lapack::getrf call.
   * @param allow_singular If `true`, allows factorization of singular matrices. If `false` (default), throws an error
   * when the matrix is detected to be singular.
   * @returns A tuple containing \f$ \mathbf{\sigma} \f$, \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$.
   */
  template <typename LP = F_layout, blas_lapack::BlasArray<2> A>
    requires(mem::have_host_compatible_addr_space<A> and blas_lapack::has_F_layout<A>)
  auto lu_in_place(A &&a, bool allow_singular = false) { // NOLINT (temporary views are allowed here)
    // pivot indices vector
    auto ipiv = vector<int>{};

    // call getrf
    int info = lapack::getrf(a, ipiv);
    if (info < 0) {
      NDA_RUNTIME_ERROR << "Error in nda::linalg::lu_in_place: getrf failed with invalid argument (info = " << info << ")";
    } else if (info > 0 and not allow_singular) {
      NDA_RUNTIME_ERROR << "Error in nda::linalg::lu_in_place: Matrix is singular, U(" << info << "," << info << ") is exactly zero";
    }

    // extract sigma, L, U from the output of getrf
    auto sigma  = get_permutation_vector(ipiv, a.extent(0));
    auto [L, U] = get_lu_matrices<LP>(a);

    return std::make_tuple(sigma, L, U);
  }

  /**
   * @brief Compute the LU factorization of a matrix.
   *
   * @details It makes a copy of the input matrix \f$ \mathbf{A} \f$ and calls nda::linalg::lu_in_place.
   *
   * The resulting \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ matrices have the same layout as the input matrix \f$ 
   * \mathbf{A} \f$.
   * 
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space and to have a value type
   * that satisfies nda::is_blas_lapack_v.
   *
   * @tparam A nda::Matrix type.
   * @param a Input matrix. The \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$ to be factorized.
   * @param allow_singular If `true`, allows factorization of singular matrices. If `false` (default), throws an error
   * when the matrix is detected to be singular.
   * @returns A tuple containing \f$ \mathbf{\sigma} \f$, \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$.
   */
  template <Matrix A>
    requires(mem::have_host_compatible_addr_space<A> and is_blas_lapack_v<get_value_t<A>>)
  auto lu(A const &a, bool allow_singular = false) {
    auto a_copy = matrix<get_value_t<A>, F_layout>(a);
    if constexpr (blas_lapack::has_F_layout<A>) {
      return lu_in_place(a_copy, allow_singular);
    } else {
      return lu_in_place<C_layout>(a_copy, allow_singular);
    }
  }

  /** @} */

} // namespace nda::linalg
