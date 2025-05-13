// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `getrf` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../basic_functions.hpp"
#include "../concepts.hpp"
#include "../layout_transforms.hpp"
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
   * @brief Interface to the LAPACK `getrf` routine.
   *
   * @details Computes an LU factorization of a general \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$ using partial
   * pivoting with row interchanges.
   *
   * The factorization has the form
   * \f[
   *   \mathbf{A} = \mathbf{P L U}
   * \f]
   * where \f$ \mathbf{P} \f$ is a permutation matrix, \f$ \mathbf{L} \f$ is lower triangular with unit diagonal
   * elements (lower trapezoidal if \f$ m > n \f$), and \f$ \mathbf{U} \f$ is upper triangular (upper trapezoidal if \f$
   * m < n \f$).
   *
   * This is the right-looking Level 3 BLAS version of the algorithm.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam IPIV nda::MemoryVector type.
   * @param a Input/output matrix. On entry, the \f$ m \times n \f$ matrix to be factored. On exit, the factors \f$
   * \mathbf{L} \f$ and \f$ \mathbf{U} \f$ from the factorization \f$ \mathbf{A} = \mathbf{P L U} \f$; the unit diagonal
   * elements of \f$ \mathbf{L} \f$ are not stored.
   * @param ipiv Output vector. The pivot indices, i.e. for \f$ 1 \leq i \leq \min(m,n) \f$, row \f$ i \f$ of the matrix
   * was interchanged with row `ipiv(i-1)`.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector IPIV>
    requires(mem::have_compatible_addr_space<A, IPIV> and is_blas_lapack_v<get_value_t<A>> and std::is_same_v<get_value_t<IPIV>, int>)
  int getrf(A &&a, IPIV &&ipiv) { // NOLINT (temporary views are allowed here)
    // for C-layout arrays/views, call getrf with the transpose
    if constexpr (has_C_layout<A>) return getrf(transpose(a), ipiv);

    // check the dimensions of the input/output arrays/views and resize if necessary
    auto const [m, n] = a.shape();
    resize_or_check_if_view(ipiv, {std::min(m, n)});

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    ipiv = 0;
#endif
#endif

    // perform actual library call
    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space<A, IPIV>) {
#if defined(NDA_HAVE_DEVICE)
      device::getrf(m, n, a.data(), get_ld(a), ipiv.data(), info);
#else
      compile_error_no_gpu();
#endif
    } else {
      f77::getrf(m, n, a.data(), get_ld(a), ipiv.data(), info);
    }
    return info;
  }

} // namespace nda::lapack
