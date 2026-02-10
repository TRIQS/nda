// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `geqp3` routine.
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
   * @brief Interface to the LAPACK `%geqp3` routine.
   *
   * @details Computes a QR factorization of a matrix \f$ \mathbf{A} \f$ with column pivoting:
   * \f[
   *   \mathbf{A P} = \mathbf{Q R}
   * \f]
   * using Level 3 BLAS.
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
   * @note All input arrays are required to satisfy nda::mem::have_host_compatible_addr_space and \f$ \mathbf{A} \f$ is 
   * required to have nda::F_layout.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @tparam JPVT nda::blas_lapack::PivotArrayFor<A, 1> type.
   * @tparam TAU nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam W1 nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam W2 nda::blas_lapack::BlasArrayRealFor<A, 1> type.
   * @param a Input/output matrix. On entry, the \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$. On exit, the upper
   * triangle of the array contains the \f$ \min(m,n) \times n \f$ upper trapezoidal matrix \f$ \mathbf{R} \f$; the
   * elements below the diagonal, together with the array \f$ \mathbf{\tau} \f$, represent the unitary matrix \f$
   * \mathbf{Q} \f$ as a product of \f$ \min(m,n) \f$ elementary reflectors.
   * @param jpvt Input/output vector. On entry, if the j<sup>th</sup> element is \f$ \neq 0 \f$, the j<sup>th</sup>
   * column of \f$ \mathbf{A} \f$ is permuted to the front of \f$ \mathbf{A P} \f$ (a leading column); if the j<sup>th
   * </sup> element is equal 0, the j<sup>th</sup> column of \f$ \mathbf{A} \f$ is a free column. On exit, if the
   * j<sup>th</sup> element is equal \f$ k \f$, then the j<sup>th</sup>  column of \f$ \mathbf{A P} \f$ was the
   * k<sup>th</sup> column of \f$ \mathbf{A} \f$.
   * @param tau Output vector. The scalar factors \f$ \tau_i \f$ of the elementary reflectors \f$ \mathbf{H}(i) \f$.
   * @param work Ouput vector. Workspace array used by the LAPACK routine.
   * @param rwork Ouput vector. Workspace array used by the LAPACK routine.
   * @return Integer return code from the LAPACK call.
   */
  template <BlasArray<2> A, PivotArrayFor<A, 1> JPVT, BlasArrayFor<A, 1> TAU, BlasArrayFor<A, 1> W1 = vector_value_t<A>,
            BlasArrayRealFor<A, 1> W2 = vector_fp_t<A>>
    requires(mem::have_host_compatible_addr_space<A> and has_F_layout<A>)
  int geqp3(A &&a, JPVT &&jpvt, TAU &&tau, W1 &&work = vector_value_t<A>{}, W2 &&rwork = vector_fp_t<A>{}) { // NOLINT
    // check the dimensions of the input/output arrays/views and resize if necessary
    auto const [m, n] = a.shape();
    EXPECTS(jpvt.size() == n);
    resize_or_check_if_view(tau, {std::min(m, n)});
    resize_or_check_work_buffer(rwork, 2 * n);

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(jpvt.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    // first call to get the optimal buffer size
    auto tmp_lwork = get_value_t<A>{};
    int info       = 0;
    lapack::f77::geqp3(m, n, a.data(), get_ld(a), jpvt.data(), tau.data(), &tmp_lwork, -1, rwork.data(), info);
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // resize/check work buffer
    resize_or_check_work_buffer(work, lwork);

    // perform actual library call
    lapack::f77::geqp3(m, n, a.data(), get_ld(a), jpvt.data(), tau.data(), work.data(), lwork, rwork.data(), info);

    return info;
  }

} // namespace nda::lapack
