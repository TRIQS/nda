// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `heev` routine.
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

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `heev` routine.
   *
   * @details Computes all eigenvalues \f$ \lambda_i \f$ and, optionally, eigenvectors \f$ \mathbf{v}_i \f$ of a complex 
   * hermitian matrix eigenvalue problem of the form
   * \f[
   *   \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \; ,
   * \f]
   * for a given complex hermitian matrix \f$ \mathbf{A} \f$.
   *
   * @tparam A nda::MemoryMatrix with complex value type.
   * @param a Input/output matrix. On entry, the hermitian matrix \f$ \mathbf{A} \f$. On exit, if `jobz = V`, \f$ 
   * \mathbf{A} \f$ contains the orthonormal eigenvectors of the matrix \f$ \mathbf{A} \f$. If `jobz = N`, then on 
   * exit \f$ \mathbf{A} \f$ is destroyed.
   * @param w Output vector. The eigenvalues in ascending order.
   * @param jobz Character indicating whether to compute eigenvectors and eigenvalues ('V') or eigenvalues only ('N').
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector W>
    requires(mem::have_host_compatible_addr_space<A> and is_complex_v<get_value_t<A>> and std::same_as<get_fp_t<A>, get_value_t<W>>)
  int heev(A &&a, W &&w, char jobz = 'V') { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A>, "Error in nda::lapack::heev: A must have Fortran layout");

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
    using fp_type = get_value_t<W>;
    array<fp_type, 1> rwork(std::max(1l, 3 * n - 2));
    std::complex<fp_type> tmp_lwork{};
    int info = 0;
    lapack::f77::heev(jobz, 'U', n, a.data(), get_ld(a), w.data(), &tmp_lwork, -1, rwork.data(), info);
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // allocate work buffer and perform actual library call
    array<std::complex<fp_type>, 1> work(lwork);
    lapack::f77::heev(jobz, 'U', n, a.data(), get_ld(a), w.data(), work.data(), lwork, rwork.data(), info);

    return info;
  }

} // namespace nda::lapack
