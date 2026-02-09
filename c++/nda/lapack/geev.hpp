// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `geev` routine.
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

#include <cmath>
#include <complex>
#include <concepts>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `geev` routine for real matrices.
   *
   * @details Computes all eigenvalues \f$ \lambda_j \f$ and, optionally, right/left eigenvectors \f$ \mathbf{v}_j \f$/
   * \f$ \mathbf{u}_j \f$ of a real eigenvalue problem.
   *
   * The right eigenvector \f$ \mathbf{v}_j \f$ of \f$ \mathbf{A} \f$ satisfies
   * \f[
   *   \mathbf{A} \mathbf{v}_j = \lambda_j \mathbf{v}_j
   * \f]
   * where \f$ \lambda_j \f$ is its eigenvalue.
   *
   * The left eigenvector \f$ \mathbf{u}_j \f$ of \f$ \mathbf{A} \f$ satisfies
   * \f[
   *   \mathbf{u}_j^T \mathbf{A} = \lambda_j \mathbf{u}_j^T
   * \f]
   * where \f$ \mathbf{u}_j^T \f$ denotes the transpose of \f$ \mathbf{u}_j \f$.
   *
   * The computed eigenvectors are normalized to have Euclidean norm equal to 1 and largest component real.
   *
   * For real matrices, complex eigenvalues always occur in complex conjugate pairs and the corresponding eigenvectors 
   * are stored in a special packed format (see nda::linalg::get_geev_eigenvectors).
   * 
   * @note \f$ \mathbf{A} \f$, \f$ \mathbf{V}_L \f$ and \f$ \mathbf{V}_R \f$ are required to satisfy 
   * nda::mem::have_host_compatible_addr_space and to have nda::F_layout.
   *
   * @tparam A nda::blas_lapack::BlasArrayReal<2> type.
   * @tparam WR nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam WI nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam VL nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam VR nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam W1 nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, \f$ \mathbf{A} \f$ is overwritten.
   * @param wr Output vector. The real parts of the computed eigenvalues, i.e. \f$ \mathrm{Re}(\lambda_j) \f$.
   * @param wi Output vector. The imaginary parts of the computed eigenvalues, i.e. \f$ \mathrm{Im}(\lambda_j) \f$.
   * @param vl Output matrix. If `jobvl = V`, matrix \f$ \mathbf{V}_L \f$ containing the left eigenvectors (in packed 
   * format for complex pairs). If `jobvl = N`, \f$ \mathbf{V}_L \f$ is not referenced.
   * @param vr Output matrix. If `jobvr = V`, matrix \f$ \mathbf{V}_R \f$ containging the right eigenvectors (in packed 
   * format for complex pairs). If `jobvr = N`, \f$ \mathbf{V}_R \f$ is not referenced.
   * @param jobvl Character indicating whether to compute left eigenvectors ('V') or not ('N').
   * @param jobvr Character indicating whether to compute right eigenvectors ('V') or not ('N').
   * @param work Ouput vector. Workspace array used by the LAPACK routine.
   * @return Integer return code from the LAPACK call.
   */
  template <BlasArrayReal<2> A, BlasArrayFor<A, 1> WR, BlasArrayFor<A, 1> WI, BlasArrayFor<A, 2> VL, BlasArrayFor<A, 2> VR,
            BlasArrayFor<A, 1> W1 = vector_value_t<A>>
    requires(mem::have_host_compatible_addr_space<A> and has_F_layout<A, VL, VR>)
  int geev(A &&a, WR &&wr, WI &&wi, VL &&vl, VR &&vr, char jobvl = 'N', char jobvr = 'V', W1 &&work = vector_value_t<A>{}) { // NOLINT
    // check the dimensions of the input/output arrays/views
    auto const [m, n] = a.shape();
    EXPECTS(m == n);
    resize_or_check_if_view(wr, {n});
    resize_or_check_if_view(wi, {n});

    // check other input parameters for consistency
    EXPECTS(jobvl == 'V' or jobvl == 'N');
    EXPECTS(jobvr == 'V' or jobvr == 'N');

    // resize eigenvector matrices if needed
    int ldvl = 1;
    int ldvr = 1;
    if (jobvl == 'V') {
      resize_or_check_if_view(vl, {n, n});
      ldvl = get_ld(vl);
    }
    if (jobvr == 'V') {
      resize_or_check_if_view(vr, {n, n});
      ldvr = get_ld(vr);
    }

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(wr.indexmap().min_stride() == 1);
    EXPECTS(wi.indexmap().min_stride() == 1);
    EXPECTS(jobvl == 'N' or vl.indexmap().min_stride() == 1);
    EXPECTS(jobvr == 'N' or vr.indexmap().min_stride() == 1);

    // first call to get the optimal buffer size
    auto tmp_lwork = get_value_t<A>{};
    int info       = 0;
    lapack::f77::geev(jobvl, jobvr, n, a.data(), get_ld(a), wr.data(), wi.data(), vl.data(), ldvl, vr.data(), ldvr, &tmp_lwork, -1, info);
    int lwork = static_cast<int>(std::ceil(tmp_lwork));

    // resize/check work buffer
    resize_or_check_work_buffer(work, lwork);

    // perform actual library call
    lapack::f77::geev(jobvl, jobvr, n, a.data(), get_ld(a), wr.data(), wi.data(), vl.data(), ldvl, vr.data(), ldvr, work.data(), lwork, info);

    return info;
  }

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `geev` routine for complex matrices.
   *
   * @details Computes all eigenvalues \f$ \lambda_j \f$ and, optionally, right/left eigenvectors \f$ \mathbf{v}_j \f$/
   * \f$ \mathbf{u}_j \f$ of a complex eigenvalue problem. 
   *
   * The right eigenvector \f$ \mathbf{v}_j \f$ of \f$ \mathbf{A} \f$ satisfies
   * \f[
   *   \mathbf{A} \mathbf{v}_j = \lambda_j \mathbf{v}_j
   * \f]
   * where \f$ \lambda_j \f$ is its eigenvalue.
   *
   * The left eigenvector \f$ \mathbf{u}_j \f$ of \f$ \mathbf{A} \f$ satisfies
   * \f[
   *   \mathbf{u}_j^H \mathbf{A} = \lambda_j \mathbf{u}_j^H
   * \f]
   * where \f$ \mathbf{u}_j^H \f$ denotes the conjugate-transpose of \f$ \mathbf{u}_j \f$.
   *
   * The computed eigenvectors are normalized to have Euclidean norm equal to 1 and largest component real.
   * 
   * @note \f$ \mathbf{A} \f$, \f$ \mathbf{V}_L \f$ and \f$ \mathbf{V}_R \f$ are required to satisfy 
   * nda::mem::have_host_compatible_addr_space and to have nda::F_layout.
   *
   * @tparam A nda::blas_lapack::BlasArrayCplx<2> type.
   * @tparam W nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam VL nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam VR nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam W1 nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam W2 nda::blas_lapack::BlasArrayRealFor<A, 1> type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, \f$ \mathbf{A} \f$ is overwritten.
   * @param w Output vector. The computed eigenvalues \f$ \lambda_j \f$.
   * @param vl Output matrix. If `jobvl = V`, matrix \f$ \mathbf{V}_L \f$ containing the left eigenvectors. If 
   * `jobvl = N`, \f$ \mathbf{V}_L \f$ is not referenced. 
   * @param vr Output matrix. If `jobvr = V`, matrix \f$ \mathbf{V}_R \f$ containging the right eigenvectors. If 
   * `jobvr = N`, \f$ \mathbf{V}_R \f$ is not referenced.
   * @param jobvl Character indicating whether to compute left eigenvectors ('V') or not ('N').
   * @param jobvr Character indicating whether to compute right eigenvectors ('V') or not ('N').
   * @param work Ouput vector. Workspace array used by the LAPACK routine.
   * @param rwork Ouput vector. Workspace array used by the LAPACK routine.
   * @return Integer return code from the LAPACK call.
   */
  template <BlasArrayCplx<2> A, BlasArrayFor<A, 1> W, BlasArrayFor<A, 2> VL, BlasArrayFor<A, 2> VR, BlasArrayFor<A, 1> W1 = vector_value_t<A>,
            BlasArrayRealFor<A, 1> W2 = vector_fp_t<A>>
    requires(mem::have_host_compatible_addr_space<A> and has_F_layout<A, VL, VR>)
  int geev(A &&a, W &&w, VL &&vl, VR &&vr, char jobvl = 'N', char jobvr = 'V', W1 &&work = vector_value_t<A>{}, // NOLINT
           W2 &&rwork = vector_fp_t<A>{}) {                                                                     // NOLINT
    // check the dimensions of the input/output arrays/views
    auto const [m, n] = a.shape();
    EXPECTS(m == n);
    resize_or_check_if_view(w, {n});
    resize_or_check_work_buffer(rwork, 2 * n);

    // check parameters
    EXPECTS(jobvl == 'V' or jobvl == 'N');
    EXPECTS(jobvr == 'V' or jobvr == 'N');

    // resize eigenvector matrices if needed
    int ldvl = 1;
    int ldvr = 1;
    if (jobvl == 'V') {
      resize_or_check_if_view(vl, {n, n});
      ldvl = get_ld(vl);
    }
    if (jobvr == 'V') {
      resize_or_check_if_view(vr, {n, n});
      ldvr = get_ld(vr);
    }

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(w.indexmap().min_stride() == 1);
    EXPECTS(jobvl == 'N' or vl.indexmap().min_stride() == 1);
    EXPECTS(jobvr == 'N' or vr.indexmap().min_stride() == 1);

    // first call to get the optimal buffer size
    auto tmp_lwork = get_value_t<A>{};
    int info       = 0;
    lapack::f77::geev(jobvl, jobvr, n, a.data(), get_ld(a), w.data(), vl.data(), ldvl, vr.data(), ldvr, &tmp_lwork, -1, rwork.data(), info);
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // resize/check work buffer
    resize_or_check_work_buffer(work, lwork);

    // perform actual library call
    lapack::f77::geev(jobvl, jobvr, n, a.data(), get_ld(a), w.data(), vl.data(), ldvl, vr.data(), ldvr, work.data(), lwork, rwork.data(), info);

    return info;
  }

} // namespace nda::lapack
