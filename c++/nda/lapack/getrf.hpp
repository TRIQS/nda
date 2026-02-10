// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK/cuSOLVER `getrf` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../basic_functions.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../device.hpp"
#include "../layout_transforms.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <algorithm>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK/cuSOLVER `%getrf` routine.
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
   * If the input arrays satisfy nda::mem::have_device_compatible_addr_space, the cuSOLVER implementation is used.
   * 
   * @note If \f$ \mathbf{A} \f$ is stored in nda::C_layout, the factorization is actually performed on \f$ \mathbf{A}^T 
   * \f$. When the result is further used in nda::lapack::getrs or nda::lapack::getri, this is automatically taken into 
   * account and works as expected.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @tparam IPIV nda::blas_lapack::PivotArrayFor<A, 1> type.
   * @tparam W nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @param a Input/output matrix. On entry, the \f$ m \times n \f$ matrix to be factored. On exit, the factors \f$
   * \mathbf{L} \f$ and \f$ \mathbf{U} \f$ from the factorization \f$ \mathbf{A} = \mathbf{P L U} \f$; the unit diagonal
   * elements of \f$ \mathbf{L} \f$ are not stored.
   * @param ipiv Output vector. The pivot indices, i.e. for \f$ 1 \leq i \leq \min(m,n) \f$, row \f$ i \f$ of the matrix
   * was interchanged with row `ipiv(i-1)`.
   * @param work Ouput vector. Workspace array only used by the cuSOLVER routine.
   * @return Integer return code from the LAPACK/cuSOLVER call.
   */
  template <BlasArray<2> A, PivotArrayFor<A, 1> IPIV, BlasArrayFor<A, 1> W = vector_value_t<A>>
  int getrf(A &&a, IPIV &&ipiv, [[maybe_unused]] W &&work = vector_value_t<A>{}) { // NOLINT (temporary views are allowed here)
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
    if constexpr (mem::have_device_compatible_addr_space<A, IPIV, W>) {
      // resize/check work buffer
      int const lwork = device::getrf_buffer_size(m, n, a.data(), get_ld(a));
      resize_or_check_work_buffer(work, lwork);

      device::getrf(m, n, a.data(), get_ld(a), work.data(), ipiv.data(), info);
    } else {
      f77::getrf(m, n, a.data(), get_ld(a), ipiv.data(), info);
    }
    return info;
  }

} // namespace nda::lapack
