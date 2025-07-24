// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `syev` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../basic_array.hpp"
#include "../basic_functions.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <cmath>
#include <concepts>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `syev` routine.
   *
   * @details Computes all eigenvalues \f$ \lambda_i \f$ and, optionally, eigenvectors \f$ \mathbf{v}_i \f$ of a real 
   * symmetric eigenvalue problem of the form
   * \f[
   *   \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \; ,
   * \f]
   * for a given real symmetric matrix \f$ \mathbf{A} \f$.
   *
   * @tparam A nda::MemoryMatrix with float or double value type.
   * @param a Input/output matrix. On entry, the symmetric matrix \f$ \mathbf{A} \f$. On exit, if `jobz = V`, \f$ 
   * \mathbf{A} \f$ contains the orthonormal eigenvectors of the matrix \f$ \mathbf{A} \f$. If `jobz = N`, then on 
   * exit \f$ \mathbf{A} \f$ is destroyed.
   * @param w Output vector. The eigenvalues in ascending order.
   * @param jobz Character indicating whether to compute eigenvectors and eigenvalues ('V') or eigenvalues only ('N').
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector W>
    requires(mem::have_host_compatible_addr_space<A> and (std::is_same_v<float, get_value_t<A>> or std::is_same_v<double, get_value_t<A>>)
             and have_same_value_type_v<A, W>)
  int syev(A &&a, W &&w, char jobz = 'V') { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A>, "Error in nda::lapack::syev: A must have Fortran layout");

    // check the dimensions of the input/output arrays/views and resize if necessary
    auto const [m, n] = a.shape();
    EXPECTS(m == n);
    resize_or_check_if_view(w, {n});

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(w.indexmap().min_stride() == 1);

    // check other input parameters for consistency
    EXPECTS(jobz == 'V' or jobz == 'N');

    // first call to get the optimal buffer size
    using value_type = get_value_t<A>;
    value_type tmp_lwork{};
    int info = 0;
    lapack::f77::syev(jobz, 'U', n, a.data(), get_ld(a), w.data(), &tmp_lwork, -1, info);
    int lwork = static_cast<int>(std::ceil(tmp_lwork));

    // allocate work buffer and perform actual library call
    array<value_type, 1> work(lwork);
    lapack::f77::syev(jobz, 'U', n, a.data(), get_ld(a), w.data(), work.data(), lwork, info);

    return info;
  }

} // namespace nda::lapack
