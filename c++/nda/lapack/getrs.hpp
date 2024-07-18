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
 * @brief Provides a generic interface to the LAPACK `getrs` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../concepts.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#ifndef NDA_HAVE_DEVICE
#include "../device.hpp"
#endif // NDA_HAVE_DEVICE

#include <algorithm>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `getrs` routine.
   *
   * @details Solves a system of linear equations
   *
   * - \f$ \mathbf{A X} = \mathbf{B} \f$,
   * - \f$ \mathbf{A}^T \mathbf{X} = \mathbf{B} \f$ or
   * - \f$ \mathbf{A}^H \mathbf{X} = \mathbf{B} \f$
   *
   * with a general n-by-n matrix \f$ \mathbf{A} \f$ using the LU factorization computed by `getrf`.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam B nda::MemoryMatrix type.
   * @tparam IPIV nda::MemoryVector type.
   * @param a Input matrix. The factors \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ from the factorization \f$ \mathbf{A}
   * = \mathbf{P L U} \f$ as computed by `getrf`.
   * @param b Input/output matrix. On entry, the right hand side matrix \f$ \mathbf{B} \f$. On exit, the solution matrix
   * \f$ \mathbf{X} \f$.
   * @param ipiv Input vector. The pivot indices from `getrf`, i.e. for `1 <= i <= n`, row i of the matrix was
   * interchanged with row `ipiv(i)`.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryMatrix B, MemoryVector IPIV>
    requires(have_same_value_type_v<A, B> and mem::have_compatible_addr_space<A, B, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  int getrs(A const &a, B &&b, IPIV const &ipiv) { // NOLINT (temporary views are allowed here)
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Error in nda::lapack::getrs: Pivoting array must have elements of type int");
    EXPECTS(ipiv.size() >= std::min(a.extent(0), a.extent(1)));

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

    // check for lazy expressions
    static constexpr bool conj_A = is_conj_array_expr<A>;
    char op_a                    = get_op<conj_A, /* transpose = */ has_C_layout<A>>;

    // perform actual library call
    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space<A, B, IPIV>) {
#if defined(NDA_HAVE_DEVICE)
      device::getrs(op_a, get_ncols(a), get_ncols(b), a.data(), get_ld(a), ipiv.data(), b.data(), get_ld(b), info);
#else
      compile_error_no_gpu();
#endif
    } else {
      f77::getrs(op_a, get_ncols(a), get_ncols(b), a.data(), get_ld(a), ipiv.data(), b.data(), get_ld(b), info);
    }
    return info;
  }

} // namespace nda::lapack
