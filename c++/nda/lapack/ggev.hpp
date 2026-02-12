// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `ggev` routine.
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
   * @brief Interface to the LAPACK `%ggev` routine for real matrices.
   *
   * @details Computes the generalized eigenvalues \f$ \lambda_j = \alpha_j / \beta_j \f$ and, optionally, right/left
   * eigenvectors \f$ \mathbf{v}_j \f$/\f$ \mathbf{u}_j \f$ of a real generalized eigenvalue problem.
   *
   * The right eigenvector \f$ \mathbf{v}_j \f$ satisfies
   * \f[
   *   \mathbf{A} \mathbf{v}_j = \lambda_j \mathbf{B} \mathbf{v}_j \; ,
   * \f]
   * whereas the left eigenvector \f$ \mathbf{u}_j \f$ satisfies
   * \f[
   *   \mathbf{u}_j^T \mathbf{A} = \lambda_j \mathbf{u}_j^T \mathbf{B} \; .
   * \f]
   * Here, \f$ \mathbf{u}_j^T \f$ denotes the transpose of \f$ \mathbf{u}_j \f$.
   *
   * The eigenvalues are stored as real triples \f$ \alpha^{(r)}_j \f$, \f$ \alpha^{(i)}_j \f$ and \f$ \beta_j \f$ such 
   * that \f$ \alpha_j = \alpha^{(r)}_j + i \alpha^{(i)}_j \f$. If \f$ \alpha^{(i)}_j = 0 \f$ and \f$ \beta_j \neq 0 
   * \f$, then the eigenvalue is real.
   *
   * The quotients \f$ \alpha^{(r)}_j / \beta_j \f$ and \f$ \alpha^{(i)}_j / \beta_j \f$ may easily over- or underflow,
   * and \f$ \beta_j \f$ may even be zero. Thus, the user should avoid naively computing the ratios. However, \f$
   * \alpha^{(r)}_j \f$ and \f$ \alpha^{(i)}_j \f$ will be always less than and usually comparable with \f$ ||A|| \f$
   * in magnitude, and \f$ \beta_j \f$ always less than and usually comparable with \f$ ||B|| \f$.
   *
   * The computed eigenvectors are scaled so that their largest components \f$ z \f$ satisfy \f$ |\mathrm{Re}(z)| +
   * |\mathrm{Im}(z)| = 1 \f$.
   *
   * For real matrices, complex eigenvalues always occur in complex conjugate pairs and the corresponding eigenvectors
   * are stored in a special packed format (see nda::linalg::unpack_eigenvectors).
   *
   * @note All input arrays are required to satisfy nda::mem::have_host_compatible_addr_space and all input matrices
   * are required to have nda::F_layout.
   *
   * @tparam A nda::blas_lapack::BlasArrayReal<2> type.
   * @tparam B nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam AR nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam AI nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam B2 nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam VL nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam VR nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam W1 nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, \f$ \mathbf{A} \f$ is overwritten.
   * @param b Input/output matrix. On entry, the matrix \f$ \mathbf{B} \f$. On exit, \f$ \mathbf{B} \f$ is overwritten.
   * @param alphar Output vector \f$ \boldsymbol{\alpha}^{(r)} \f$. The real parts of the numerator of the computed 
   * eigenvalues, i.e. \f$ \alpha^{(r)}_j = \mathrm{Re}(\alpha_j) \f$.
   * @param alphai Output vector \f$ \boldsymbol{\alpha}^{(i)} \f$. The imaginary parts of the numerator of the computed 
   * eigenvalues, i.e. \f$ \alpha^{(i)}_j = \mathrm{Im}(\alpha_j) \f$.
   * @param beta Output vector \f$ \boldsymbol{\beta} \f$. The denominators of the computed eigenvalues, i.e. \f$ 
   * \beta_j \f$.
   * @param vl Output matrix \f$ \mathbf{V}_L \f$. If `jobvl = V`, the matrix contains the left eigenvectors (in packed
   * format for complex pairs). If `jobvl = N`, \f$ \mathbf{V}_L \f$ is not referenced.
   * @param vr Output matrix \f$ \mathbf{V}_R \f$. If `jobvr = V`, the matrix contains the right eigenvectors (in packed
   * format for complex pairs). If `jobvr = N`, \f$ \mathbf{V}_R \f$ is not referenced.
   * @param jobvl Character indicating whether to compute left eigenvectors ('V') or not ('N').
   * @param jobvr Character indicating whether to compute right eigenvectors ('V') or not ('N').
   * @param work Output vector. Workspace array used by the LAPACK routine.
   * @return Integer return code from the LAPACK call.
   */
  template <BlasArrayReal<2> A, BlasArrayFor<A, 2> B, BlasArrayFor<A, 1> AR, BlasArrayFor<A, 1> AI, BlasArrayFor<A, 1> B2, BlasArrayFor<A, 2> VL,
            BlasArrayFor<A, 2> VR, BlasArrayFor<A, 1> W1 = vector_value_t<A>>
    requires(mem::have_host_compatible_addr_space<A> and has_F_layout<A, B, VL, VR>)
  int ggev(A &&a, B &&b, AR &&alphar, AI &&alphai, B2 &&beta, VL &&vl, VR &&vr, char jobvl = 'N', char jobvr = 'V', // NOLINT
           W1 &&work = vector_value_t<A>{}) {                                                                       // NOLINT
    // check the dimensions of the input/output arrays/views
    auto const [m, n] = a.shape();
    EXPECTS(m == n);
    EXPECTS(b.shape() == a.shape());
    resize_or_check_if_view(alphar, {n});
    resize_or_check_if_view(alphai, {n});
    resize_or_check_if_view(beta, {n});

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
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(alphar.indexmap().min_stride() == 1);
    EXPECTS(alphai.indexmap().min_stride() == 1);
    EXPECTS(beta.indexmap().min_stride() == 1);
    EXPECTS(jobvl == 'N' or vl.indexmap().min_stride() == 1);
    EXPECTS(jobvr == 'N' or vr.indexmap().min_stride() == 1);

    // first call to get the optimal buffer size
    auto tmp_lwork = get_value_t<A>{};
    int info       = 0;
    lapack::f77::ggev(jobvl, jobvr, n, a.data(), get_ld(a), b.data(), get_ld(b), alphar.data(), alphai.data(), beta.data(), vl.data(), ldvl,
                      vr.data(), ldvr, &tmp_lwork, -1, info);
    int lwork = static_cast<int>(std::ceil(tmp_lwork));

    // resize/check work buffer
    resize_or_check_work_buffer(work, lwork);

    // perform actual library call
    lapack::f77::ggev(jobvl, jobvr, n, a.data(), get_ld(a), b.data(), get_ld(b), alphar.data(), alphai.data(), beta.data(), vl.data(), ldvl,
                      vr.data(), ldvr, work.data(), lwork, info);

    return info;
  }

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `%ggev` routine for complex matrices.
   *
   * @details Computes the generalized eigenvalues \f$ \lambda_j = \alpha_j / \beta_j \f$ and, optionally, right/left
   * eigenvectors \f$ \mathbf{v}_j \f$/\f$ \mathbf{u}_j \f$ of a complex generalized eigenvalue problem.
   *
   * The right eigenvector \f$ \mathbf{v}_j \f$ satisfies
   * \f[
   *   \mathbf{A} \mathbf{v}_j = \lambda_j \mathbf{B} \mathbf{v}_j \; ,
   * \f]
   * whereas the left eigenvector \f$ \mathbf{u}_j \f$ satisfies
   * \f[
   *   \mathbf{u}_j^H \mathbf{A} = \lambda_j \mathbf{u}_j^H \mathbf{B} \; .
   * \f]
   * Here, \f$ \mathbf{u}_j^H \f$ denotes the conjugate-transpose of \f$ \mathbf{u}_j \f$.
   *
   * The eigenvalues are stored as complex pairs \f$ \alpha_j \f$ and \f$ \beta_j \f$. If \f$ \beta_j \neq 0 \f$, the
   * eigenvalue is \f$ \lambda_j = \alpha_j / \beta_j \f$.
   *
   * The quotient \f$ \alpha_j / \beta_j \f$ may easily over- or underflow, and \f$ \beta_j \f$ may even be zero. Thus,
   * the user should avoid naively computing the ratio. However, \f$ \alpha_j \f$ will be always less than and usually
   * comparable with \f$ ||A|| \f$ in magnitude, and \f$ \beta_j \f$ always less than and usually comparable with \f$
   * ||B|| \f$.
   *
   * The computed eigenvectors are scaled so that their largest components \f$ z \f$ satisfy \f$ |\mathrm{Re}(z)| +
   * |\mathrm{Im}(z)| = 1 \f$.
   *
   * @note All input arrays are required to satisfy nda::mem::have_host_compatible_addr_space and all input matrices
   * are required to have nda::F_layout.
   *
   * @tparam A nda::blas_lapack::BlasArrayCplx<2> type.
   * @tparam B nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam A2 nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam B2 nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam VL nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam VR nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam W1 nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam W2 nda::blas_lapack::BlasArrayRealFor<A, 1> type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, \f$ \mathbf{A} \f$ is overwritten.
   * @param b Input/output matrix. On entry, the matrix \f$ \mathbf{B} \f$. On exit, \f$ \mathbf{B} \f$ is overwritten.
   * @param alpha Output vector \f$ \boldsymbol{\alpha} \f$. The numerators of the computed eigenvalues, i.e. \f$ 
   * \alpha_j \f$.
   * @param beta Output vector \f$ \boldsymbol{\beta} \f$. The denominators of the computed eigenvalues, i.e. \f$ 
   * \beta_j \f$.
   * @param vl Output matrix \f$ \mathbf{V}_L \f$. If `jobvl = V`, the matrix contains the left eigenvectors. If
   * `jobvl = N`, \f$ \mathbf{V}_L \f$ is not referenced.
   * @param vr Output matrix \f$ \mathbf{V}_R \f$. If `jobvr = V`, the matrix contains the right eigenvectors. If 
   * `jobvr = N`, \f$ \mathbf{V}_R \f$ is not referenced.
   * @param jobvl Character indicating whether to compute left eigenvectors ('V') or not ('N').
   * @param jobvr Character indicating whether to compute right eigenvectors ('V') or not ('N').
   * @param work Output vector. Workspace array used by the LAPACK routine.
   * @param rwork Output vector. Workspace array used by the LAPACK routine.
   * @return Integer return code from the LAPACK call.
   */
  template <BlasArrayCplx<2> A, BlasArrayFor<A, 2> B, BlasArrayFor<A, 1> A2, BlasArrayFor<A, 1> B2, BlasArrayFor<A, 2> VL, BlasArrayFor<A, 2> VR,
            BlasArrayFor<A, 1> W1 = vector_value_t<A>, BlasArrayRealFor<A, 1> W2 = vector_fp_t<A>>
    requires(mem::have_host_compatible_addr_space<A> and has_F_layout<A, B, VL, VR>)
  int ggev(A &&a, B &&b, A2 &&alpha, B2 &&beta, VL &&vl, VR &&vr, char jobvl = 'N', char jobvr = 'V', // NOLINT
           W1 &&work = vector_value_t<A>{}, W2 &&rwork = vector_fp_t<A>{}) {                          // NOLINT
    // check the dimensions of the input/output arrays/views
    auto const [m, n] = a.shape();
    EXPECTS(m == n);
    EXPECTS(b.shape() == a.shape());
    resize_or_check_if_view(alpha, {n});
    resize_or_check_if_view(beta, {n});
    resize_or_check_work_buffer(rwork, 8 * n);

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
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(alpha.indexmap().min_stride() == 1);
    EXPECTS(beta.indexmap().min_stride() == 1);
    EXPECTS(jobvl == 'N' or vl.indexmap().min_stride() == 1);
    EXPECTS(jobvr == 'N' or vr.indexmap().min_stride() == 1);

    // first call to get the optimal buffer size
    auto tmp_lwork = get_value_t<A>{};
    int info       = 0;
    lapack::f77::ggev(jobvl, jobvr, n, a.data(), get_ld(a), b.data(), get_ld(b), alpha.data(), beta.data(), vl.data(), ldvl, vr.data(), ldvr,
                      &tmp_lwork, -1, rwork.data(), info);
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // resize/check work buffer
    resize_or_check_work_buffer(work, lwork);

    // perform actual library call
    lapack::f77::ggev(jobvl, jobvr, n, a.data(), get_ld(a), b.data(), get_ld(b), alpha.data(), beta.data(), vl.data(), ldvl, vr.data(), ldvr,
                      work.data(), lwork, rwork.data(), info);

    return info;
  }

} // namespace nda::lapack
