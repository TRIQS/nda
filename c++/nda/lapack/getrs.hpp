// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK/cuSOLVER `getrs` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../device.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK/cuSOLVER `%getrs` routine.
   *
   * @details Solves a system of linear equations
   * 
   * - \f$ \mathbf{A} \mathbf{X} = \mathbf{B} \f$ or
   * - \f$ \mathbf{A} \mathbf{x} = \mathbf{b} \f$,
   * 
   * with a general \f$ n \times n \f$ matrix \f$ \mathbf{A} \f$ and either \f$ n \times n_{\mathrm{rhs}} \f$ matrices 
   * \f$ \mathbf{X} \f$ and \f$ \mathbf{B} \f$ or vectors \f$ \mathbf{x} \f$ and \f$ \mathbf{b} \f$ of size \f$ n \f$.
   * 
   * If the input arrays satisfy nda::mem::have_device_compatible_addr_space, the cuSOLVER implementation is used.
   * 
   * @note \f$ \mathbf{A} \f$ is allowed to be a lazy conjugate expression (see nda::blas_lapack::is_conj_array_expr), 
   * in which case it is required to be in nda::C_layout. \f$ \mathbf{B} \f$ is required to be in nda::F_layout.
   *
   * @tparam A nda::blas_lapack::BlasArrayOrConj<2> type.
   * @tparam B nda::blas_lapack::BlasArrayFor<A> type.
   * @tparam IPIV nda::blas_lapack::PivotArrayFor<A, 1> type.
   * @param a Input matrix. The factors \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ from the factorization \f$ \mathbf{A}
   * = \mathbf{P L U} \f$ as computed by nda::lapack::getrf.
   * @param b Input/output matrix/vector. On entry, the right hand side matrix \f$ \mathbf{B} \f$ or vector \f$ 
   * \mathbf{b} \f$. On exit, the solution matrix \f$ \mathbf{X} \f$ or vector \f$ \mathbf{x} \f$.
   * @param ipiv Input vector. The pivot indices from nda::lapack::getrf, i.e. for \f$ 1 \leq i \leq n \f$, row \f$ i 
   * \f$ of the matrix was interchanged with row `ipiv(i-1)`.
   * @return Integer return code from the LAPACK call.
   */
  template <BlasArrayOrConj<2> A, BlasArrayFor<A> B, PivotArrayFor<A, 1> IPIV>
    requires((get_rank<B> == 1 or get_rank<B> == 2) and has_F_layout<B>)
  int getrs(A const &a, B &&b, IPIV const &ipiv) { // NOLINT (temporary views are allowed here)
    // get underlying matrix in case it is given as a lazy conjugate expression
    auto &a_mat = get_array(a);

    // check the dimensions of the input/output arrays/views
    EXPECTS(a_mat.extent(0) == a_mat.extent(1));
    EXPECTS(b.extent(0) == a_mat.extent(0));
    EXPECTS(ipiv.size() == a_mat.extent(0));

    // arrays/views must be LAPACK compatible
    EXPECTS(a_mat.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

    // perform actual library call
    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space<A, B, IPIV>) {
      device::getrs(get_op<A>, get_ncols(a_mat), get_ncols(b), a_mat.data(), get_ld(a_mat), ipiv.data(), b.data(), get_ld(b), info);
    } else {
      f77::getrs(get_op<A>, get_ncols(a_mat), get_ncols(b), a_mat.data(), get_ld(a_mat), ipiv.data(), b.data(), get_ld(b), info);
    }

    return info;
  }

} // namespace nda::lapack
