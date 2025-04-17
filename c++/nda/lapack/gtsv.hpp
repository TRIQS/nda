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
#include "../concepts.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `gtsv` routine.
   *
   * @details Solves the equation
   * \f[
   *   \mathbf{A} \mathbf{X} = \mathbf{B},
   * \f]
   * where \f$ \mathbf{A} \f$ is an n-by-n tridiagonal matrix, by Gaussian elimination with partial pivoting.
   *
   * Note that the equation \f$ \mathbf{A}^T \mathbf{X} = \mathbf{B} \f$ may be solved by interchanging the order of the
   * arguments containing the subdiagonal elements.
   *
   * @tparam DL nda::MemoryVector type.
   * @tparam D nda::MemoryVector type.
   * @tparam DU nda::MemoryVector type.
   * @tparam B nda::MemoryArray type.
   * @param dl Input/Output vector. On entry, it must contain the (n-1) subdiagonal elements of \f$ \mathbf{A} \f$. On
   * exit, it is overwritten by the (n-2) elements of the second superdiagonal of the upper triangular matrix
   * \f$ \mathbf{U} \f$ from the LU factorization of \f$ \mathbf{A} \f$.
   * @param d Input/Output vector. On entry, it must contain the diagonal elements of \f$ \mathbf{A} \f$. On exit, it is
   * overwritten by the n diagonal elements of \f$ \mathbf{U} \f$.
   * @param du Input/Output vector. On entry, it must contain the (n-1) superdiagonal elements of \f$ \mathbf{A} \f$. On
   * exit, it is overwritten by the (n-1) elements of the first superdiagonal of \f$ \mathbf{U} \f$ .
   * @param b Input/Output array. On entry, the n-by-nrhs right hand side matrix \f$ \mathbf{B} \f$. On exit, if
   * `INFO == 0`, the n-by-nrhs solution matrix \f$ \mathbf{X} \f$.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryVector DL, MemoryVector D, MemoryVector DU, MemoryArray B>
    requires(have_same_value_type_v<DL, D, DU, B> and mem::on_host<DL, D, DU, B> and is_blas_lapack_v<get_value_t<DL>>)
  int gtsv(DL &&dl, D &&d, DU &&du, B &&b) { // NOLINT (temporary views are allowed here)
    static_assert((get_rank<B> == 1 or get_rank<B> == 2), "Error in nda::lapack::gtsv: B must be an matrix/array/view of rank 1 or 2");

    // get and check dimensions of input arrays
    EXPECTS(dl.extent(0) == d.extent(0) - 1); // "gtsv : dimension mismatch between sub-diagonal and diagonal vectors "
    EXPECTS(du.extent(0) == d.extent(0) - 1); // "gtsv : dimension mismatch between super-diagonal and diagonal vectors "
    EXPECTS(b.extent(0) == d.extent(0));      // "gtsv : dimension mismatch between diagonal vector and RHS matrix, "

    // perform actual library call
    int N    = d.extent(0);
    int NRHS = (get_rank<B> == 2 ? b.extent(1) : 1);
    int info = 0;
    f77::gtsv(N, NRHS, dl.data(), d.data(), du.data(), b.data(), N, info);
    return info;
  }

} // namespace nda::lapack
