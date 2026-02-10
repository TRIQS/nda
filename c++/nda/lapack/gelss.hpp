// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `gelss` routine.
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
   * @brief Interface to the LAPACK `%gelss` routine.
   *
   * @details Computes the minimum norm solution to a linear least squares problem:
   * \f[
   *   \min_{\mathbf{x}} | \mathbf{b} - \mathbf{A x} |_2
   * \f]
   * using the singular value decomposition (SVD) of \f$ \mathbf{A} \f$, an \f$ m \times n \f$ matrix which may be 
   * rank-deficient.
   *
   * Several right hand side vectors \f$ \mathbf{b} \f$ and solution vectors \f$ \mathbf{x} \f$ can be handled in a
   * single call; they are stored as the columns of the \f$ m \times n_{\mathrm{rhs}} \f$ right hand side matrix \f$
   * \mathbf{B} \f$ and the \f$ n \times n_{\mathrm{rhs}} \f$ solution matrix \f$ \mathbf{X} \f$.
   *
   * The effective rank of \f$ \mathbf{A} \f$ is determined by treating as zero those singular values which are less
   * than \f$ r_{\mathrm{cond}} \f$ times the largest singular value.
   * 
   * @note All input arrays are required to satisfy nda::mem::have_host_compatible_addr_space and all input matrices
   * are required to have nda::F_layout. Since we do not resize the input array representing the right hand side, it is
   * required to be large enough to hold the solution, i.e. it needs to have at least \f$ \max(m, n) \f$ rows.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @tparam B nda::blas_lapack::BlasArrayFor<A> type.
   * @tparam S nda::blas_lapack::BlasArrayRealFor<A, 1> type.
   * @tparam W1 nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @tparam W2 nda::blas_lapack::BlasArrayRealFor<A, 1> type.
   * @param a Input/output matrix. On entry, the \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$. On exit, the first \f$
   * \min(m,n) \f$ rows of \f$ \mathbf{A} \f$ are overwritten with its right singular vectors, stored rowwise.
   * @param b Input/output array. On entry, the \f$ m \times n_{\mathrm{rhs}} \f$ right hand side matrix \f$ \mathbf{B}
   * \f$ or vector \f$ \mathbf{b} \f$. On exit, it is overwritten by the \f$ n \times n_{\mathrm{rhs}} \f$ solution
   * matrix \f$ \mathbf{X} \f$ or vector \f$ \mathbf{x} \f$. If \f$ m \geq n \f$ and if the effective rank is equal \f$
   * n \f$, the residual sum-of-squares for the solution in the i<sup>th</sup> column is given by the sum of squares of
   * the modulus of elements \f$ n + 1 \f$ to \f$ m \f$ in that column.
   * @param s Output vector. The singular values of \f$ \mathbf{A} \f$ in decreasing order. The condition number of \f$
   * \mathbf{A} \f$ in the 2-norm is \f$ s_1 / s_{min(m,n)} \f$.
   * @param rcond It is used to determine the effective rank of \f$ \mathbf{A} \f$. Singular values \f$ s_i \leq
   * r_{\mathrm{cond}} s_1 \f$ are treated as zero. If \f$ r_{\mathrm{cond}} < 0 \f$, machine precision is used instead.
   * @param rank Output variable. The effective rank of \f$ \mathbf{A} \f$, i.e. the number of singular values which
   * are greater than \f$ r_{\mathrm{cond}} s_1 \f$.
   * @param work Ouput vector. Workspace array used by the LAPACK routine.
   * @param rwork Ouput vector. Workspace array used by the LAPACK routine.
   * @return Integer return code from the LAPACK call.
   */
  template <BlasArray<2> A, BlasArrayFor<A> B, BlasArrayRealFor<A, 1> S, BlasArrayFor<A, 1> W1 = vector_value_t<A>,
            BlasArrayRealFor<A, 1> W2 = vector_fp_t<A>>
    requires(mem::have_host_compatible_addr_space<A> and (get_rank<B> == 1 or get_rank<B> == 2) and has_F_layout<A, B>)
  int gelss(A &&a, B &&b, S &&s, get_fp_t<A> rcond, int &rank, W1 &&work = vector_value_t<A>{}, W2 &&rwork = vector_fp_t<A>{}) { // NOLINT
    // check the dimensions of the input/output arrays/views and resize if necessary
    auto const [m, n] = a.shape();
    auto const k      = std::min(m, n);
    resize_or_check_if_view(s, {k});
    resize_or_check_work_buffer(rwork, 5 * k);
    EXPECTS(b.extent(0) >= std::max(m, n));

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(s.indexmap().min_stride() == 1);

    // first call to get the optimal buffer size
    auto tmp_lwork = get_value_t<A>{};
    int info       = 0;
    int nrhs       = (get_rank<B> == 2 ? b.extent(1) : 1);
    f77::gelss(m, n, nrhs, a.data(), get_ld(a), b.data(), get_ld(b), s.data(), rcond, rank, &tmp_lwork, -1, rwork.data(), info);
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // resize/check work buffer
    resize_or_check_work_buffer(work, lwork);

    // perform actual library call
    f77::gelss(m, n, nrhs, a.data(), get_ld(a), b.data(), get_ld(b), s.data(), rcond, rank, work.data(), lwork, rwork.data(), info);

    return info;
  }

} // namespace nda::lapack
