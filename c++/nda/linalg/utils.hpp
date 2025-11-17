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
 * @brief Provides utility functions for the nda::linalg namespace.
 */

#pragma once

#include "../basic_array.hpp"
#include "../basic_functions.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <utility>

namespace nda::linalg {

  /**
   * @addtogroup linalg_tools
   * @{
   */

  /**
   * @brief Get the permutation vector \f$ \mathbf{\sigma} \f$ from the pivot indices returned by nda::lapack::getrf or
   * other LAPACK routines.
   *
   * @details The function constructs the permutation vector \f$ \mathbf{\sigma} \f$ of size \f$ m \f$ from the pivot
   * index vector `ipiv` of size \f$ l \f$. Starting from an identity permutation, i.e. \f$ \mathbf{\sigma} = (0, 1,
   * \ldots, m - 1) \f$, it interchanges \f$ \sigma_i \f$ with \f$ \sigma_{\mathrm{ipiv}_i - 1} \f$ for all \f$ i = 0,
   * 1, \dots, l - 1 \f$.
   *
   * @param ipiv nda::Vector containing the pivot indices returned by nda::lapack::getrf.
   * @param m Number of elements of the permutation vector \f$ \mathbf{\sigma} \f$.
   * @return Permutation vector \f$ \mathbf{\sigma} \f$ as an nda::vector.
   */
  auto get_permutation_vector(Vector auto const &ipiv, int m) {
    static_assert(nda::mem::have_host_compatible_addr_space<decltype(ipiv)>);
    EXPECTS(m >= ipiv.size());
    auto sigma = vector<int>{arange<int>(m)};
    for (int i = 0; i < ipiv.size(); ++i) std::swap(sigma(i), sigma(ipiv(i) - 1));
    return sigma;
  }

  /**
   * @brief Get the permutation matrix \f$ \mathbf{P} \f$ from a permutation vector \f$ \mathbf{\sigma} \f$.
   *
   * @details The function constructs the permutation matrix \f$ \mathbf{P} \f$ of size \f$ m \times m \f$ from the
   * permutation vector \f$ \sigma \f$ of size \f$ m \f$. 
   * 
   * The permutation matrix only has the following non-zero elements (for all \f$ i = 0, 1, \ldots, m - 1 \f$):
   * - \f$ \mathbf{P}_{i, \sigma_i} = 1 \f$ for row permutations,
   * - \f$ \mathbf{P}_{\sigma_i, i} = 1 \f$ for column permutations.
   *
   * @tparam T nda::Scalar value type of the permutation matrix.
   * @tparam LP Policy determining the memory layout of the permutation matrix.
   * @param sigma nda::Vector containing the permutation vector with values \f$ \in \{0, 1, \ldots, m - 1 \} \f$.
   * @param column_permutations If true, constructs the permutation matrix for column permutations.
   * @return Permutation matrix \f$ \mathbf{P} \f$ as an nda::matrix.
   */
  template <Scalar T, typename LP = F_layout>
  auto get_permutation_matrix(Vector auto const &sigma, bool column_permutations = false) {
    static_assert(nda::mem::have_host_compatible_addr_space<decltype(sigma)>);
    int m  = sigma.size();
    auto P = matrix<T, LP>::zeros(m, m);
    for (int i = 0; i < m; ++i) (column_permutations ? P(sigma(i), i) : P(i, sigma(i))) = T{1};
    return P;
  }

  /**
   * @brief Get the permutation matrix \f$ \mathbf{P} \f$ from the pivot indices returned by nda::lapack::getrf or other
   * LAPACK routines.
   *
   * @details It simply calls nda::linalg::get_permutation_vector to get the permutation vector from the pivot indices,
   * and then calls nda::linalg::get_permutation_matrix to get the permutation matrix.
   *
   * @tparam T nda::Scalar value type of the permutation matrix.
   * @tparam LP Policy determining the memory layout of the permutation matrix.
   * @param ipiv nda::Vector containing the pivot indices returned by nda::lapack::getrf.
   * @param m Number of rows/columns of the square permutation matrix \f$ \mathbf{P} \f$.
   * @return Permutation matrix \f$ \mathbf{P} \f$ as an nda::matrix.
   */
  template <Scalar T, typename LP = F_layout>
  auto get_permutation_matrix(Vector auto const &ipiv, int m) {
    return get_permutation_matrix<T, LP>(get_permutation_vector(ipiv, m));
  }

  /** @} */

} // namespace nda::linalg
