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
   * @details If the original matrix \f$ \mathbf{A} \f$ is of size \f$ m \times n \f$, then the returned \f$ \mathbf{L}
   * \f$ matrix is of size \f$ m \times k \f$ and \f$ \mathbf{U} \f$ is of size \f$ k \times n \f$, where \f$ k =
   * \min(m, n) \f$.
   *
   * @tparam LP Policy determining the memory layout of the \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ matrices.
   * @tparam M nda::MemoryMatrix type.
   * @param a nda::MemoryMatrix containing the output of nda::lapack::getrf.
   * @return A tuple containing the \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ matrices.
   */
  template <typename LP = F_layout, MemoryMatrix A>
    requires(nda::mem::have_host_compatible_addr_space<A> and nda::blas::has_F_layout<A>)
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
   * @note \f$ \mathbf{A} \f$ must be in nda::F_layout. See nda::linalg::lu for a version that handles C-layout input.
   * \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ can be returned in a different layout by specifying the `LP` template
   * parameter.
   *
   * @tparam LP Policy determining the memory layout of the \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ matrices.
   * @tparam M nda::MemoryMatrix type.
   * @param a Input/Output nda::MemoryMatrix. On entry, the \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$. On exit, the 
   * result of the nda::lapack::getrf call.
   * @returns A tuple containing \f$ \mathbf{\sigma} \f$, \f$ \mathbf{L} \f$, \f$ \mathbf{U} \f$ and the info value
   * returned by `getrf`.
   */
  template <typename LP = F_layout, MemoryMatrix A>
    requires(nda::mem::have_host_compatible_addr_space<A> and nda::blas::has_F_layout<A> and is_blas_lapack_v<get_value_t<A>>)
  auto lu_in_place(A &&a) { // NOLINT (temporary views are allowed here)
    // input, output types and static assertions

    // pivot indices vector
    auto ipiv = vector<int>{};

    // call lapack getrf
    int info = lapack::getrf(a, ipiv);

    // extract sigma, L, U from the output of getrf
    auto sigma  = get_permutation_vector(ipiv, a.extent(0));
    auto [L, U] = get_lu_matrices<LP>(a);

    return std::make_tuple(sigma, L, U, info);
  }

  /**
   * @brief Compute the LU factorization of a matrix.
   *
   * @details It makes a copy of the input matrix \f$ \mathbf{A} \f$ and calls nda::linalg::lu_in_place.
   * 
   * @note \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ have the same layout as the input matrix \f$ \mathbf{A} \f$.
   *
   * @tparam A nda::Matrix type.
   * @param a Input matrix. The \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$ to be factorized.
   * @returns A tuple containing \f$ \mathbf{\sigma} \f$, \f$ \mathbf{L} \f$, \f$ \mathbf{U} \f$ and the info value
   * returned by `getrf`.
   */
  template <Matrix A>
    requires(nda::mem::have_host_compatible_addr_space<A> and is_blas_lapack_v<get_value_t<A>>)
  auto lu(A const &a) {
    auto a_copy = matrix<get_value_t<A>, F_layout>(a);
    if constexpr (nda::blas::has_F_layout<A>) {
      return lu_in_place(a_copy);
    } else {
      return lu_in_place<C_layout>(a_copy);
    }
  }

  /** @} */

} // namespace nda::linalg
