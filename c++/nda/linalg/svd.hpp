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
 * @brief Provides functions to compute the singular value decoomposition of a matrix.
 */

#pragma once

#include "../basic_array.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../lapack/gesvd.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#include <algorithm>
#include <tuple>
#include <type_traits>
#include <utility>

namespace nda::linalg {

  /**
   * @addtogroup linalg_tools
   * @{
   */

  /**
   * @brief Compute the singular value decomposition (SVD) of a matrix in place.
   *
   * @details The function computes the SVD of a given \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$:
   * \f[
   *   \mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{V}^H \; ,
   * \f]
   * where \f$ \mathbf{U} \f$ is a unitary \f$ m \times m \f$ matrix, \f$ \mathbf{V} \f$ is a unitary \f$ n \times n \f$
   * matrix and \f$ \mathbf{S} \f$ is an \f$ m \times n \f$ matrix with non-negative real numbers on the diagonal.
   *
   * It first constructs the output vector \f$ \mathbf{s} \f$, which contains the singular values, and the output
   * matrices \f$ \mathbf{U} \f$ and \f$ \mathbf{V}^H \f$. It then calls nda::lapack::gesvd to compute the SVD.
   *
   * An exception is thrown, if the LAPACK call returns a non-zero value.
   *
   * @note If the input matrix \f$ \mathbf{A} \f$ is in nda::F_layout, the output matrices \f$ \mathbf{U} \f$ and
   * \f$ \mathbf{V}^H \f$ are also in nda::F_layout. Otherwise, they are in nda::C_layout.
   *
   * @tparam A nda::MemoryMatrix type.
   * @param a Input/output matrix. On entry, the \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$. On exit, the contents of
   * \f$ \mathbf{A} \f$ are destroyed.
   * @return `std::tuple` containing \f$ \mathbf{U} \f$, \f$ \mathbf{s} \f$ and \f$ \mathbf{V}^H \f$.
   */
  template <MemoryMatrix A>
    requires(is_blas_lapack_v<get_value_t<A>>)
  auto svd_in_place(A &&a) { // NOLINT (temporary views are allowed here)
    using layout_policy       = nda::detail::layout_to_policy<typename std::remove_cvref_t<A>::layout_t>::type;
    constexpr auto addr_space = nda::mem::get_addr_space<A>;

    // vector s and matrices U and V^H
    auto const [m, n] = a.shape();
    auto s            = vector<get_fp_t<A>, heap<addr_space>>(std::min(m, n));
    auto U            = matrix<get_value_t<A>, layout_policy, heap<addr_space>>(m, m);
    auto VH           = matrix<get_value_t<A>, layout_policy, heap<addr_space>>(n, n);

    // call lapack gesvd
    int info = lapack::gesvd(a, s, U, VH);
    if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::svd_in_place: gesvd returned a non-zero value: info = " << info;

    return std::make_tuple(U, s, VH);
  }

  /**
   * @brief Compute the singular value decomposition (SVD) of a matrix.
   *
   * @details The function computes the SVD of a given \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$:
   * \f[
   *   \mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{V}^H \; ,
   * \f]
   * where \f$ \mathbf{U} \f$ is a unitary \f$ m \times m \f$ matrix, \f$ \mathbf{V} \f$ is a unitary \f$ n \times n \f$
   * matrix and \f$ \mathbf{S} \f$ is an \f$ m \times n \f$ matrix with non-negative real numbers on the diagonal.
   *
   * It calls nda::linalg::svd_in_place with a copy of the input matrix \f$ \mathbf{A} \f$.
   *
   * @note If the input matrix \f$ \mathbf{A} \f$ is in nda::F_layout, the output matrices \f$ \mathbf{U} \f$ and
   * \f$ \mathbf{V}^H \f$ are also in nda::F_layout. Otherwise, they are in nda::C_layout.
   * 
   * @warning This function makes copies of the input arrays/views. When working on the device memory space, this may
   * lead to runtime errors if the copying fails.
   *
   * @tparam A nda::MemoryMatrix type.
   * @param a Input matrix \f$ \mathbf{A} \f$.
   * @return `std::tuple` containing \f$ \mathbf{U} \f$, \f$ \mathbf{s} \f$ and \f$ \mathbf{V}^H \f$.
   */
  template <Matrix A>
    requires(is_blas_lapack_v<get_value_t<A>>)
  auto svd(A const &a) { // NOLINT (temporary views are allowed here)
    return svd_in_place(basic_array{a});
  }

  /** @} */

} // namespace nda::linalg
