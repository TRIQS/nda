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
#include "../declarations.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#ifndef NDA_HAVE_DEVICE
#include "../device.hpp"
#endif // NDA_HAVE_DEVICE

#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `getrs` routine.
   *
   * @details Solves a system of linear equations
   * - \f$ \mathbf{A X} = \mathbf{B} \f$ or
   * - \f$ \mathbf{A}^* \mathbf{X} = \mathbf{B} \f$.
   *
   * with a general \f$ n \times n \f$ matrix \f$ \mathbf{A} \f$ using the LU factorization computed by
   * nda::lapack::getrf.
   *
   * @tparam A nda::Matrix type.
   * @tparam B nda::MemoryArray type.
   * @tparam IPIV nda::MemoryVector type.
   * @param a Input matrix. The factors \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ from the factorization \f$ \mathbf{A}
   * = \mathbf{P L U} \f$ as computed by nda::lapack::getrf.
   * @param b Input/output matrix. On entry, the right hand side matrix \f$ \mathbf{B} \f$. On exit, the solution matrix
   * \f$ \mathbf{X} \f$.
   * @param ipiv Input vector. The pivot indices from nda::lapack::getrf, i.e. for \f$ 1 \leq i \leq n \f$, row i of the
   * matrix was interchanged with row `ipiv(i)`.
   * @return Integer return code from the LAPACK call.
   */
  template <Matrix A, MemoryArray B, MemoryVector IPIV>
    requires((MemoryMatrix<A> or is_conj_array_expr<A>)
             and have_same_value_type_v<A, B> and mem::have_compatible_addr_space<A, B, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  int getrs(A const &a, B &&b, IPIV const &ipiv) { // NOLINT (temporary views are allowed here)
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Error in nda::lapack::getrs: Pivoting array must have elements of type int");
    static_assert(get_rank<B> == 1 || get_rank<B> == 2, "Error in nda::lapack::getrs: Right hand side must have rank 1 or 2");
    static_assert(has_F_layout<B>, "Error in nda::lapack::getrs: B must have Fortran layout");

    // get underlying matrix in case it is given as a lazy conjugate expression
    auto &a_mat = get_array(a);

    // check the dimensions of the input/output arrays/views
    EXPECTS(a_mat.shape()[0] == a_mat.shape()[1]);
    EXPECTS(b.extent(0) == a_mat.shape()[0]);
    EXPECTS(ipiv.size() == a_mat.shape()[0]);

    // arrays/views must be LAPACK compatible
    EXPECTS(a_mat.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

    // check for conjugate lazy expressions and C-layouts
    char op_a = get_op<is_conj_array_expr<A>, has_C_layout<A>>;

    // perform actual library call
    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space<A, B, IPIV>) {
#if defined(NDA_HAVE_DEVICE)
      device::getrs(op_a, get_ncols(a_mat), get_ncols(b), a_mat.data(), get_ld(a_mat), ipiv.data(), b.data(), get_ld(b), info);
#else
      compile_error_no_gpu();
#endif
    } else {
      f77::getrs(op_a, get_ncols(a_mat), get_ncols(b), a_mat.data(), get_ld(a_mat), ipiv.data(), b.data(), get_ld(b), info);
    }

    return info;
  }

} // namespace nda::lapack
