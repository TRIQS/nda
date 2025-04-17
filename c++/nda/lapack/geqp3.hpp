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
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
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
   * @brief Interface to the LAPACK `geqp3` routine.
   *
   * @details Computes a QR factorization with column pivoting of a matrix \f$ \mathbf{A} \f$:
   * \f[
   *   \mathbf{A P} = \mathbf{Q R}
   * \f]
   * using Level 3 BLAS.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam JPVT nda::MemoryVector type.
   * @tparam TAU nda::MemoryVector type.
   * @param a Input/output matrix. On entry, the m-by-n matrix \f$ \mathbf{A} \f$. On exit, the upper triangle of the
   * array contains the `min(m,n)`-by-n upper trapezoidal matrix \f$ \mathbf{R} \f$; the elements below the diagonal,
   * together with the array `tau`, represent the unitary matrix \f$ \mathbf{Q} \f$ as a product of `min(m,n)`
   * elementary reflectors.
   * @param jpvt Input/output vector. On entry, if `jpvt(j) != 0`, the j-th column of \f$ \mathbf{A} \f$ is permuted to
   * the front of \f$ \mathbf{A P} \f$ (a leading column); if `jpvt(j) == 0`, the j-th column of \f$ \mathbf{A} \f$ is a
   * free column. On exit, if `jpvt(j) == k`, then the j-th column of \f$ \mathbf{A P} \f$ was the the k-th column of
   * \f$ \mathbf{A} \f$.
   * @param tau Output vector. The scalar factors of the elementary reflectors.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector JPVT, MemoryVector TAU>
    requires(mem::on_host<A> and is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, TAU>
             and mem::have_compatible_addr_space<A, JPVT, TAU>)
  int geqp3(A &&a, JPVT &&jpvt, TAU &&tau) { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A>, "Error in nda::lapack::geqp3: C order not supported");
    static_assert(std::is_same_v<get_value_t<JPVT>, int>, "Error in nda::lapack::geqp3: Pivoting array must have elements of type int");
    static_assert(mem::have_host_compatible_addr_space<A, JPVT, TAU>, "Error in nda::lapack::geqp3: Only CPU is supported");

    auto [m, n] = a.shape();
    EXPECTS(tau.size() >= std::min(m, n));

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(jpvt.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    // first call to get the optimal buffersize
    using value_type = get_value_t<A>;
    value_type bufferSize_T{};
    int info = 0;
    array<double, 1> rwork(2 * n);
    lapack::f77::geqp3(m, n, a.data(), get_ld(a), jpvt.data(), tau.data(), &bufferSize_T, -1, rwork.data(), info);
    int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

    // allocate work buffer and perform actual library call
    nda::array<value_type, 1> work(bufferSize);
    lapack::f77::geqp3(m, n, a.data(), get_ld(a), jpvt.data(), tau.data(), work.data(), bufferSize, rwork.data(), info);
    jpvt -= 1; // Shift to 0-based indexing

    if (info) NDA_RUNTIME_ERROR << "Error in nda::lapack::geqp3: info = " << info;
    return info;
  }

} // namespace nda::lapack
