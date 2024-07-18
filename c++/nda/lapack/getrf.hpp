// Copyright (c) 2021-2024 Simons Foundation
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
// Authors: Thomas Hahn, Miguel Morales, Nils Wentzell

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `getrf` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../concepts.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <algorithm>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `getrf` routine.
   *
   * @details Computes an LU factorization of a general m-by-n matrix \f$ \mathbf{A} \f$ using partial pivoting with row
   * interchanges.
   *
   * The factorization has the form
   * \f[
   *   \mathbf{A} = \mathbf{P L U}
   * \f]
   * where \f$ \mathbf{P} \f$ is a permutation matrix, \f$ \mathbf{L} \f$ is lower triangular with unit diagonal
   * elements (lower trapezoidal if `m > n`), and \f$ \mathbf{U} \f$ is upper triangular (upper trapezoidal if `m < n`).
   *
   * This is the right-looking Level 3 BLAS version of the algorithm.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam IPIV nda::MemoryVector type.
   * @param a Input/output matrix. On entry, the m-by-n matrix to be factored. On exit, the factors \f$ \mathbf{L} \f$
   * and \f$ \mathbf{U} \f$ from the factorization \f$ \mathbf{A} = \mathbf{P L U} \f$; the unit diagonal elements of
   * \f$ \mathbf{L} \f$ are not stored.
   * @param ipiv Output vector. The pivot indices from `getrf`, i.e. for `1 <= i <= n`, row i of the matrix was
   * interchanged with row `ipiv(i)`.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector IPIV>
    requires(mem::have_compatible_addr_space<A, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  int getrf(A &&a, IPIV &&ipiv) { // NOLINT (temporary views are allowed here)
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Error in nda::lapack::getri: Pivoting array must have elements of type int");

    auto dm = std::min(a.extent(0), a.extent(1));
    if (ipiv.size() < dm) ipiv.resize(dm); // ipiv needs to be a regular array?

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    ipiv = 0;
#endif
#endif

    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space<A, IPIV>) {
#if defined(NDA_HAVE_DEVICE)
      device::getrf(a.extent(0), a.extent(1), a.data(), get_ld(a), ipiv.data(), info);
#else
      compile_error_no_gpu();
#endif
    } else {
      f77::getrf(a.extent(0), a.extent(1), a.data(), get_ld(a), ipiv.data(), info);
    }
    return info;
  }

} // namespace nda::lapack
