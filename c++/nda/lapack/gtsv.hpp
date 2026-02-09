// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `gtsv` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `gtsv` routine.
   *
   * @details Solves a system of linear equations
   *
   * - \f$ \mathbf{A} \mathbf{X} = \mathbf{B} \f$ or
   * - \f$ \mathbf{A} \mathbf{x} = \mathbf{b} \f$,
   *
   * with a tridiagonal \f$ n \times n \f$ matrix \f$ \mathbf{A} \f$ and either \f$ n \times n_{\mathrm{rhs}} \f$
   * matrices \f$ \mathbf{X} \f$ and \f$ \mathbf{B} \f$ or vectors \f$ \mathbf{x} \f$ and \f$ \mathbf{b} \f$ of size
   * \f$ n \f$. It uses Gaussian elimination with partial pivoting.
   *
   * @note The input arrays/views are required to satisfy nda::mem::have_host_compatible_addr_space and \f$ \mathbf{B}
   * \f$ has to be in nda::F_layout.
   *
   * @tparam DL nda::blas_lapack::BlasArray<1> type.
   * @tparam D nda::blas_lapack::BlasArrayFor\<DL, 1\> type.
   * @tparam DU nda::blas_lapack::BlasArrayFor\<DL, 1\> type.
   * @tparam B nda::blas_lapack::BlasArrayFor\<DL\> type.
   * @param dl Input/Output vector. On entry, it must contain the \f$ n - 1 \f$ subdiagonal elements of \f$ \mathbf{A}
   * \f$. On exit, it is overwritten by the \f$ n - 2 \f$ elements of the second superdiagonal of the upper triangular
   * matrix \f$ \mathbf{U} \f$ from the LU factorization of \f$ \mathbf{A} \f$.
   * @param d Input/Output vector. On entry, it must contain the diagonal elements of \f$ \mathbf{A} \f$. On exit, it is
   * overwritten by the \f$ n \f$ diagonal elements of \f$ \mathbf{U} \f$.
   * @param du Input/Output vector. On entry, it must contain the \f$ n - 1 \f$ superdiagonal elements of \f$ \mathbf{A}
   * \f$. On exit, it is overwritten by the \f$ n - 1 \f$ elements of the first superdiagonal of \f$ \mathbf{U} \f$ .
   * @param b Input/Output array. On entry, the \f$ n \times n_{\mathrm{rhs}} \f$ right hand side matrix \f$ \mathbf{B}
   * \f$ or the vector \f$ \mathbf{b} \f$. On exit, the \f$ n \times n_{\mathrm{rhs}} \f$ solution matrix \f$ \mathbf{X}
   * \f$ or the vector \f$ \mathbf{x} \f$.
   * @return Integer return code from the LAPACK call.
   */
  template <BlasArray<1> DL, BlasArrayFor<DL, 1> D, BlasArrayFor<DL, 1> DU, BlasArrayFor<DL> B>
    requires(mem::have_host_compatible_addr_space<DL> and (get_rank<B> == 1 or get_rank<B> == 2) and has_F_layout<B>)
  int gtsv(DL &&dl, D &&d, DU &&du, B &&b) { // NOLINT (temporary views are allowed here)
    // check the dimensions of the input/output arrays/views
    auto const n = d.size();
    EXPECTS(dl.size() == n - 1);
    EXPECTS(du.size() == n - 1);
    EXPECTS(b.extent(0) == n);

    // perform actual library call
    int info = 0;
    f77::gtsv(n, (get_rank<B> == 2 ? b.extent(1) : 1), dl.data(), d.data(), du.data(), b.data(), get_ld(b), info);

    return info;
  }

} // namespace nda::lapack
