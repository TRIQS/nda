// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `ungqr` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../basic_array.hpp"
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
   * @brief Interface to the LAPACK `ungqr` routine.
   *
   * @details Generates an \f$ m \times \min(m,n) = k \f$ complex matrix \f$ \mathbf{Q} \f$ with orthonormal columns,
   * which is defined as the first \f$ k \f$ columns of a product of \f$ k \f$ elementary reflectors of order \f$ m \f$
   * \f[
   *   \mathbf{Q} = \mathbf{H}(1) \mathbf{H}(2) \ldots \mathbf{H}(k) \; ,
   * \f]
   * as returned by nda::lapack::geqp3.
   *
   * Each \f$ \mathbf{H}(i) \f$ has the form
   * \f[
   *   \mathbf{H}(i) = \mathbf{I} - \tau_i * \mathbf{v}_i \mathbf{v}_i^H
   * \f]
   * where \f$ \tau_i \f$ is a complex scalar, and \f$ \mathbf{v}_i \f$ is a complex vector with
   * - elements \f$ 1 \f$ to \f$ i - 1 \f$ equal to 0,
   * - element \f$ i \f$ equal to 1 and
   * - elements \f$ i + 1 \f$ to \f$ m \f$ stored in the elements \f$ i + 1 \f$ to \f$ m \f$ in column \f$ i \f$ of
   * matrix \f$ \mathbf{A} \f$.
   *
   * @tparam A nda::MemoryMatrix with complex value type.
   * @tparam TAU nda::MemoryVector with complex value type.
   * @param a Input/output matrix. On entry, the i<sup>th</sup> column must contain the vector which defines the
   * elementary reflector \f$ H(i) \; , i = 1,2,...,k \f$, as returned by nda::lapack::geqp3 in the first \f$ k \f$
   * columns. On exit, the \f$ m \times \min(m,n) = k \f$ matrix \f$ \mathbf{Q} \f$.
   * @param tau Input vector. \f$ \tau_i \f$ must contain the scalar factor of the elementary reflector \f$
   * \mathbf{H}(i) \f$, as returned by nda::lapack::geqp3.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector TAU>
    requires(mem::have_host_compatible_addr_space<A> and is_complex_v<get_value_t<A>> and have_same_value_type_v<A, TAU>)
  int ungqr(A &&a, TAU &&tau) { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A>, "Error in nda::lapack::ungqr: A must have Fortran layout");

    // check the dimensions of the input/output arrays/views
    auto const [m, n] = a.shape();
    auto const k      = std::min(m, n);
    EXPECTS(tau.size() == k);

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    // first call to get the optimal buffer size
    using value_type = get_value_t<A>;
    value_type tmp_lwork{};
    int info = 0;
    lapack::f77::ungqr(m, k, k, a.data(), get_ld(a), tau.data(), &tmp_lwork, -1, info);
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // allocate work buffer and perform actual library call
    nda::array<value_type, 1> work(lwork);
    lapack::f77::ungqr(m, k, k, a.data(), get_ld(a), tau.data(), work.data(), lwork, info);

    return info;
  }

} // namespace nda::lapack
