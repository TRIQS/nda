// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `getrs` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../concepts.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#ifndef NDA_HAVE_DEVICE
#include "../device.hpp"
#endif // NDA_HAVE_DEVICE

#include <algorithm>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `getrs` routine.
   *
   * @details Solves a system of linear equations
   *
   * - \f$ \mathbf{A X} = \mathbf{B} \f$,
   * - \f$ \mathbf{A}^T \mathbf{X} = \mathbf{B} \f$ or
   * - \f$ \mathbf{A}^H \mathbf{X} = \mathbf{B} \f$
   *
   * with a general n-by-n matrix \f$ \mathbf{A} \f$ using the LU factorization computed by `getrf`.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam B nda::MemoryMatrix type.
   * @tparam IPIV nda::MemoryVector type.
   * @param a Input matrix. The factors \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ from the factorization \f$ \mathbf{A}
   * = \mathbf{P L U} \f$ as computed by `getrf`.
   * @param b Input/output matrix. On entry, the right hand side matrix \f$ \mathbf{B} \f$. On exit, the solution matrix
   * \f$ \mathbf{X} \f$.
   * @param ipiv Input vector. The pivot indices from `getrf`, i.e. for `1 <= i <= n`, row i of the matrix was
   * interchanged with row `ipiv(i)`.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryMatrix B, MemoryVector IPIV>
    requires(have_same_value_type_v<A, B> and mem::have_compatible_addr_space<A, B, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  int getrs(A const &a, B &&b, IPIV const &ipiv) { // NOLINT (temporary views are allowed here)
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Error in nda::lapack::getrs: Pivoting array must have elements of type int");
    EXPECTS(ipiv.size() >= std::min(a.extent(0), a.extent(1)));

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

    // check for lazy expressions
    static constexpr bool conj_A = is_conj_array_expr<A>;
    char op_a                    = get_op<conj_A, /* transpose = */ has_C_layout<A>>;

    // perform actual library call
    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space<A, B, IPIV>) {
#if defined(NDA_HAVE_DEVICE)
      device::getrs(op_a, get_ncols(a), get_ncols(b), a.data(), get_ld(a), ipiv.data(), b.data(), get_ld(b), info);
#else
      compile_error_no_gpu();
#endif
    } else {
      f77::getrs(op_a, get_ncols(a), get_ncols(b), a.data(), get_ld(a), ipiv.data(), b.data(), get_ld(b), info);
    }
    return info;
  }

  /**
   * Solves a system of linear equations
   *    A * X = B,  A**T * X = B,  or  A**H * X = B
   * with a general N-by-N matrix A using the LU factorization computed
   * by getrf.
   *
   * [in]      a is real/complex array, dimension (LDA,N)
   *           The factors L and U from the factorization A = P*L*U
   *           as computed by ZGETRF.
   *
   * [in]      ipiv is INTEGER array, dimension (N)
   *           The pivot indices from ZGETRF; for 1<=i<=N, row i of the
   *           matrix was interchanged with row ipiv(i).
   *
   * [in,out]  b is real/complex array, dimension (LDB,NRHS)
   *           On entry, the right hand side matrix B.
   *           On exit, the solution matrix X.
   *
   * [return]  info is INTEGER
   *           = 0:  successful exit
   *           < 0:  if info = -i, the i-th argument had an illegal value
   */
  template <MemoryMatrix A, MemoryMatrix B, MemoryVector IPIV>
    requires(have_same_value_type_v<A, B> and mem::have_compatible_addr_space<A, B, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  int getrs(char op, A const &a, B &b, IPIV const &ipiv) { // NOLINT (temporary views are allowed here)
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Error in nda::lapack::getrs: Pivoting array must have elements of type int");
    EXPECTS(ipiv.size() >= std::min(a.extent(0), a.extent(1)));

    // Must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space<A, B, IPIV>) {
#if defined(NDA_HAVE_DEVICE)
      device::getrs(op, a.extent(1), b.extent(1), a.data(), get_ld(a), ipiv.data(), b.data(), get_ld(b), info);
#else
      compile_error_no_gpu();
#endif
    } else {
      f77::getrs(op, a.extent(1), b.extent(1), a.data(), get_ld(a), ipiv.data(), b.data(), get_ld(b), info);
    }
    return info;
  }
} // namespace nda::lapack
