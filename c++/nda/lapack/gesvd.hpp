// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK/cuSOLVER `gesvd` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../basic_array.hpp"
#include "../basic_functions.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../device.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <utility>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK/cuSOLVER `gesvd` routine.
   *
   * @details Computes the singular value decomposition (SVD) of an \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$. The
   * SVD is written as
   * \f[
   *   \mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{V}^H \; ,
   * \f]
   * where \f$ \mathbf{S} \f$ is an \f$ m \times n \f$ matrix which is zero except for its \f$ \min(m,n) \f$ diagonal
   * elements, \f$ \mathbf{U} \f$ is an \f$ m \times m \f$ unitary matrix, and \f$ \mathbf{V} \f$ is an \f$ n \times n
   * \f$ unitary matrix. The diagonal elements of \f$ \mathbf{S} \f$ are the singular values of \f$ \mathbf{A} \f$; they
   * are real and non-negative, and are returned in descending order. The first \f$ min(m,n) \f$ columns of \f$
   * \mathbf{U} \f$ and \f$ \mathbf{V} \f$ are the left and right singular vectors of \f$ \mathbf{A} \f$.
   * 
   * If the input arrays satisfy nda::mem::have_device_compatible_addr_space, the cuSOLVER implementation is used.
   * 
   * @note \f$ \mathbf{A} \f$, \f$ \mathbf{U} \f$ and \f$ \mathbf{V}^H \f$ are required to have the same memory layout.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @tparam S nda::blas_lapack::BlasArrayRealFor<A, 1> type.
   * @tparam U nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam VH nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam W1 nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam W2 nda::blas_lapack::BlasArrayRealFor<A, 1> type.
   * @param a Input/output matrix. On entry, the \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$. On exit, the contents of
   * \f$ \mathbf{A} \f$ are destroyed.
   * @param s Output vector. The singular values of \f$ \mathbf{A} \f$, sorted so that \f$ s_i \geq s_{i+1} \f$.
   * @param u Output matrix. It contains the \f$ m \times m \f$ unitary matrix \f$ \mathbf{U} \f$.
   * @param vh Output matrix. It contains the \f$ n \times n \f$ unitary matrix \f$ \mathbf{V}^H \f$.
   * @param work Ouput vector. Workspace array used by the LAPACK/cuSOLVER routine.
   * @param rwork Output vector. Workspace array used by the LAPACK/cuSOLVER routine.
   * @return Integer return code from the LAPACK/cuSOLVER call.
   */
  template <BlasArray<2> A, BlasArrayRealFor<A, 1> S, BlasArrayFor<A, 2> U, BlasArrayFor<A, 2> VH, BlasArrayFor<A, 1> W1 = vector_value_t<A>,
            BlasArrayRealFor<A, 1> W2 = vector_fp_t<A>>
    requires(has_C_layout<A> == has_C_layout<U> and has_C_layout<A> == has_C_layout<VH>)
  int gesvd(A &&a, S &&s, U &&u, VH &&vh, W1 &&work = vector_value_t<A>{}, W2 &&rwork = vector_fp_t<A>{}) { // NOLINT (tmp views)
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A, S, U, VH, W1, W2>;

    // check the dimensions of the output arrays/views and resize if necessary
    auto [m, n]  = a.shape();
    auto const k = std::min(m, n);
    resize_or_check_if_view(s, {k});
    resize_or_check_if_view(u, {m, m});
    resize_or_check_if_view(vh, {n, n});
    resize_or_check_work_buffer(rwork, 5 * k);

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(s.indexmap().min_stride() == 1);
    EXPECTS(u.indexmap().min_stride() == 1);
    EXPECTS(vh.indexmap().min_stride() == 1);

    // take care of C-layouts by swapping U and V^H
    auto u_data  = u.data();
    auto vh_data = vh.data();
    auto u_ld    = get_ld(u);
    auto vh_ld   = get_ld(vh);
    if constexpr (has_C_layout<A>) {
      std::swap(u_data, vh_data);
      std::swap(u_ld, vh_ld);
      std::swap(m, n);
    }

    // cusolverDn?gesvd only supports matrices with m >= n
    if constexpr (run_on_device) { EXPECTS(m >= n); }

    // first call to get the optimal buffer size
    auto tmp_lwork = get_value_t<A>{};
    int info       = 0;
    if constexpr (run_on_device) {
      tmp_lwork = device::gesvd_buffer_size(m, n, a.data());
    } else {
      f77::gesvd('A', 'A', m, n, a.data(), get_ld(a), s.data(), u_data, u_ld, vh_data, vh_ld, &tmp_lwork, -1, rwork.data(), info);
    }
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // resize/check work buffer
    resize_or_check_work_buffer(work, lwork);

    // perform actual library call
    if constexpr (run_on_device) {
      device::gesvd('A', 'A', m, n, a.data(), get_ld(a), s.data(), u_data, u_ld, vh_data, vh_ld, work.data(), lwork, rwork.data(), info);
    } else {
      f77::gesvd('A', 'A', m, n, a.data(), get_ld(a), s.data(), u_data, u_ld, vh_data, vh_ld, work.data(), lwork, rwork.data(), info);
    }

    return info;
  }

} // namespace nda::lapack
