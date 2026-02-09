// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `hegv` routine.
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
#include <concepts>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `hegv` routine.
   *
   * @details Computes all eigenvalues \f$ \lambda_i \f$ and, optionally, eigenvectors \f$ \mathbf{v}_i \f$ of a complex
   * generalized Hermitian-definite eigenvalue problem of the form
   * 
   * - \f$ \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{B} \mathbf{v}_i \f$ (`itype = 1`),
   * - \f$ \mathbf{A} \mathbf{B} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 2`) or
   * - \f$ \mathbf{B} \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 3`).
   *
   * Here \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are assumed to be Hermitian and \f$ \mathbf{B} \f$ is also positive
   * definite.
   * 
   * @note \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are required to satisfy nda::mem::have_host_compatible_addr_space 
   * and to have nda::F_layout.
   *
   * @tparam A nda::blas_lapack::BlasArrayCplx<2> type.
   * @tparam B nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam W nda::blas_lapack::BlasArrayRealFor<A, 1> type.
   * @tparam W1 nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam W2 nda::blas_lapack::BlasArrayRealFor<A, 1> type.
   * @param a Input/output matrix. On entry, the Hermitian matrix \f$ \mathbf{A} \f$. On exit, if `jobz = V`, \f$
   * \mathbf{A} \f$ contains the matrix \f$ \mathbf{V} \f$ of normalized eigenvectors such that \f$ \mathbf{V}^H
   * \mathbf{B} \mathbf{V} = \mathbf{I} \f$ (if `itype = 1` or `itype = 2`) or \f$ \mathbf{V}^H \mathbf{B}^{-1}
   * \mathbf{V} = \mathbf{I} \f$ (if `itype = 3`). If `jobz = N`, then on exit \f$ \mathbf{A} \f$ is destroyed.
   * @param b Input/output matrix. On entry, the symmetric positive definite matrix \f$ \mathbf{B} \f$. On exit, the
   * part of \f$ \mathbf{B} \f$ containing the matrix is overwritten by the triangular factor \f$ \mathbf{U} \f$ or
   * \f$ \mathbf{L} \f$ from a Cholesky factorization.
   * @param w Output vector. The eigenvalues \f$ \lambda_i \f$ in ascending order.
   * @param jobz Character indicating whether to compute eigenvectors and eigenvalues ('V') or eigenvalues only ('N').
   * @param itype Specifies the problem to be solved.
   * @param work Ouput vector. Workspace array used by the LAPACK routine.
   * @param rwork Ouput vector. Workspace array used by the LAPACK routine.
   * @return Integer return code from the LAPACK call.
   */
  template <BlasArrayCplx<2> A, BlasArrayFor<A, 2> B, BlasArrayRealFor<A, 1> W, BlasArrayFor<A, 1> W1 = vector_value_t<A>,
            BlasArrayRealFor<A, 1> W2 = vector_fp_t<A>>
    requires(mem::have_host_compatible_addr_space<A> and has_F_layout<A, B>)
  int hegv(A &&a, B &&b, W &&w, char jobz = 'V', int itype = 1, W1 &&work = vector_value_t<A>{}, W2 &&rwork = vector_fp_t<A>{}) { // NOLINT
    // check the dimensions of the input/output arrays/views and resize if necessary
    auto const [m, n] = a.shape();
    EXPECTS(m == n);
    EXPECTS(m == b.shape()[0]);
    EXPECTS(n == b.shape()[1]);
    resize_or_check_if_view(w, {n});
    resize_or_check_work_buffer(rwork, std::max(1l, 3 * n - 2));

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(w.indexmap().min_stride() == 1);

    // check other input parameters for consistency
    EXPECTS(itype == 1 or itype == 2 or itype == 3);
    EXPECTS(jobz == 'V' or jobz == 'N');

    // first call to get the optimal buffer size
    auto tmp_lwork = get_value_t<A>{};
    int info       = 0;
    lapack::f77::hegv(itype, jobz, 'U', n, a.data(), get_ld(a), b.data(), get_ld(b), w.data(), &tmp_lwork, -1, rwork.data(), info);
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // resize/check work buffer
    resize_or_check_work_buffer(work, lwork);

    // perform actual library call
    lapack::f77::hegv(itype, jobz, 'U', n, a.data(), get_ld(a), b.data(), get_ld(b), w.data(), work.data(), lwork, rwork.data(), info);

    return info;
  }

} // namespace nda::lapack
