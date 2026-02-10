// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK/cuSOLVER `geqrf` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../basic_array.hpp"
#include "../basic_functions.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK/cuSOLVER `%geqrf` routine.
   *
   * @details Computes a QR factorization of a matrix \f$ \mathbf{A} \f$:
   * \f[
   *   \mathbf{A} = \mathbf{Q R} \; .
   * \f]
   *
   * The matrix \f$ \mathbf{Q} \f$ is represented as a product of elementary reflectors
   * \f[
   *   \mathbf{Q} = \mathbf{H}(1) \mathbf{H}(2) \ldots \mathbf{H}(k) \; ,
   * \f]
   * where \f$ k = \min(m,n) \f$.
   *
   * Each \f$ \mathbf{H}(i) \f$ has the form
   * \f[
   *   \mathbf{H}(i) = \mathbf{I} - \tau_i * \mathbf{v}_i \mathbf{v}_i^H
   * \f]
   * where \f$ \tau_i \f$ is a real/complex scalar, and \f$ \mathbf{v}_i \f$ is a real/complex vector with
   * - elements \f$ 1 \f$ to \f$ i - 1 \f$ equal to 0,
   * - element \f$ i \f$ equal to 1 and
   * - elements \f$ i + 1 \f$ to \f$ m \f$ stored on exit in the elements \f$ i + 1 \f$ to \f$ m \f$ in the column \f$ i
   * \f$ of \f$ \mathbf{A} \f$.
   * 
   * If the input arrays satisfy nda::mem::have_device_compatible_addr_space, the cuSOLVER implementation is used.
   * 
   * @note \f$ \mathbf{A} \f$ is required to have nda::F_layout.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @tparam TAU nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam W nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @param a Input/output matrix. On entry, the \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$. On exit, the upper
   * triangle of the array contains the \f$ \min(m,n) \times n \f$ upper trapezoidal matrix \f$ \mathbf{R} \f$; the
   * elements below the diagonal, together with the array \f$ \mathbf{\tau} \f$, represent the unitary matrix \f$
   * \mathbf{Q} \f$ as a product of \f$ \min(m,n) \f$ elementary reflectors.
   * @param tau Output vector. The scalar factors \f$ \tau_i \f$ of the elementary reflectors \f$ \mathbf{H}(i) \f$.
   * @param work Ouput vector. Workspace array used by the LAPACK/cuSOLVER routine.
   * @return Integer return code from the LAPACK/cuSOLVER call.
   */
  template <BlasArray<2> A, BlasArrayFor<A, 1> TAU, BlasArrayFor<A, 1> W = vector_value_t<A>>
    requires(has_F_layout<A>)
  int geqrf(A &&a, TAU &&tau, W &&work = vector_value_t<A>{}) { // NOLINT (temporary views are allowed here)
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A, TAU, W>;

    // check the dimensions of the input/output arrays/views and resize if necessary
    auto const [m, n] = a.shape();
    resize_or_check_if_view(tau, {std::min(m, n)});

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    // first call to get the optimal buffer size
    auto tmp_lwork = get_value_t<A>{};
    int info       = 0;
    if constexpr (run_on_device) {
      tmp_lwork = device::geqrf_buffer_size(m, n, a.data(), get_ld(a));
    } else {
      f77::geqrf(m, n, a.data(), get_ld(a), tau.data(), &tmp_lwork, -1, info);
    }
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // resize/check work buffer
    resize_or_check_work_buffer(work, lwork);

    // perform actual library call
    if constexpr (run_on_device) {
      device::geqrf(m, n, a.data(), get_ld(a), tau.data(), work.data(), lwork, info);
    } else {
      f77::geqrf(m, n, a.data(), get_ld(a), tau.data(), work.data(), lwork, info);
    }

    return info;
  }

} // namespace nda::lapack
