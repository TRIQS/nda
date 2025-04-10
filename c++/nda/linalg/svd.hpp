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

namespace nda {

  /**
   * @addtogroup linalg_tools
   * @{
   */

  namespace detail {

    // Function to compute the singular value decomposition in place.
    template <MemoryMatrix A>
      requires(is_blas_lapack_v<get_value_t<A>>)
    auto svd_in_place(A &&a) { // NOLINT (temporary views are allowed here)
      using layout_policy       = detail::layout_to_policy<typename std::remove_cvref_t<A>::layout_t>::type;
      constexpr auto addr_space = mem::get_addr_space<A>;

      // vector s and matrices U and V^H
      auto s  = vector<double, heap<addr_space>>(std::min(a.extent(0), a.extent(1)));
      auto U  = matrix<get_value_t<A>, layout_policy, heap<addr_space>>(a.extent(0), a.extent(0));
      auto VH = matrix<get_value_t<A>, layout_policy, heap<addr_space>>(a.extent(1), a.extent(1));

      // call lapack gesvd
      int info = lapack::gesvd(a, s, U, VH);
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::svd_in_place: gesvd returned a non-zero value: info = " << info;

      return std::make_tuple(U, s, VH);
    }

  } // namespace detail

  /**
   * @brief Compute the singular value decomposition (SVD) of a matrix.
   *
   * @details The function computes the SVD of a given m-by-n matrix \f$ \mathbf{A} \f$:
   * \f[
   *   \mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{V}^H \; ,
   * \f]
   * where \f$ \mathbf{U} \f$ is a unitary m-by-m matrix, \f$ \mathbf{V} \f$ is a unitary n-by-n matrix and \f$
   * \mathbf{S} \f$ is an m-by-n matrix with non-negative real numbers on the diagonal.
   *
   * It first makes a copy of the input matrix \f$ \mathbf{A} \f$ and constructs the output vector \f$ \mathbf{s} \f$,
   * which contains the singular values, and the output matrices \f$ \mathbf{U} \f$ and \f$ \mathbf{V}^H \f$. The SVD is
   * performed by calling nda::lapack::gesvd.
   *
   * @note If the input matrix \f$ \mathbf{A} \f$ is in Fortran layout, the output matrices \f$ \mathbf{U} \f$ and
   * \f$ \mathbf{V}^H \f$ are also in Fortran layout. Otherwise, they are in C layout.
   *
   * @tparam A nda::MemoryMatrix type.
   * @param a Input matrix \f$ \mathbf{A} \f$.
   * @return `std::tuple` containing \f$ \mathbf{U} \f$, \f$ \mathbf{s} \f$ and \f$ \mathbf{V}^H \f$.
   */
  template <Matrix A>
    requires(is_blas_lapack_v<get_value_t<A>>)
  auto svd(A const &a) { // NOLINT (temporary views are allowed here)
    return detail::svd_in_place(basic_array{a});
  }

  /** @} */

} // namespace nda
