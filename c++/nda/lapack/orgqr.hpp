// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK/cuSOLVER `orgqr` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../basic_array.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK/cuSOLVER `orgqr` routine.
   *
   * @details Generates an \f$ m \times n \f$ real matrix \f$ \mathbf{Q} \f$ with orthonormal columns, which is defined
   * as the first \f$ n \f$ columns of a product of \f$ k \f$ elementary reflectors of order \f$ m \f$
   * \f[
   *   \mathbf{Q} = \mathbf{H}(1) \mathbf{H}(2) \ldots \mathbf{H}(k) \; ,
   * \f]
   * as returned by nda::lapack::geqp3 or nda::lapack::geqrf.
   *
   * Each \f$ \mathbf{H}(i) \f$ has the form
   * \f[
   *   \mathbf{H}(i) = \mathbf{I} - \tau_i * \mathbf{v}_i \mathbf{v}_i^T
   * \f]
   * where \f$ \tau_i \f$ is a real scalar, and \f$ \mathbf{v}_i \f$ is a real vector with
   * - elements \f$ 1 \f$ to \f$ i - 1 \f$ equal to 0,
   * - element \f$ i \f$ equal to 1 and
   * - elements \f$ i + 1 \f$ to \f$ m \f$ stored in the elements \f$ i + 1 \f$ to \f$ m \f$ in column \f$ i \f$ of
   * matrix \f$ \mathbf{A} \f$.
   * 
   * If the input arrays satisfy nda::mem::have_device_compatible_addr_space, the cuSOLVER implementation is used.
   * 
   * @note \f$ \mathbf{A} \f$ is required to be stored in nda::F_layout.
   *
   * @tparam A nda::blas_lapack::BlasArrayReal<2> type.
   * @tparam TAU nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam W nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @param a Input/output matrix. On entry, the i<sup>th</sup> column must contain the vector which defines the
   * elementary reflector \f$ H(i) \; , i = 1,2,...,k \f$, as returned by nda::lapack::geqp3 or nda::lapack::geqrf. On 
   * exit, the \f$ m \times n \f$ matrix \f$ \mathbf{Q} \f$.
   * @param tau Input vector. \f$ \tau_i \f$ must contain the scalar factor of the elementary reflector \f$
   * \mathbf{H}(i) \f$, as returned by nda::lapack::geqp3 or nda::lapack::geqrf.
   * @param work Ouput vector. Workspace array used by the LAPACK/cuSOLVER routine.
   * @return Integer return code from the LAPACK/cuSOLVER call.
   */
  template <BlasArrayReal<2> A, BlasArrayFor<A, 1> TAU, BlasArrayFor<A, 1> W = vector_value_t<A>>
    requires(has_F_layout<A>)
  int orgqr(A &&a, TAU &&tau, W &&work = vector_value_t<A>{}) { // NOLINT (temporary views are allowed here)
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A, TAU, W>;

    // check the dimensions of the input/output arrays/views
    auto const [m, n] = a.shape();
    auto const k      = tau.size();
    EXPECTS(m >= n);
    EXPECTS(n >= k);

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    // first call to get the optimal buffer size
    auto tmp_lwork = get_value_t<A>{};
    int info       = 0;
    if constexpr (run_on_device) {
      tmp_lwork = device::orgqr_buffer_size(m, n, k, a.data(), get_ld(a), tau.data());
    } else {
      f77::orgqr(m, n, k, a.data(), get_ld(a), tau.data(), &tmp_lwork, -1, info);
    }
    int lwork = static_cast<int>(std::ceil(tmp_lwork));

    // resize/check work buffer
    resize_or_check_work_buffer(work, lwork);

    // perform actual library call
    if constexpr (run_on_device) {
      device::orgqr(m, n, k, a.data(), get_ld(a), tau.data(), work.data(), lwork, info);
    } else {
      f77::orgqr(m, n, k, a.data(), get_ld(a), tau.data(), work.data(), lwork, info);
    }

    return info;
  }

} // namespace nda::lapack
