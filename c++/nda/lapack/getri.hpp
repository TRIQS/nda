// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `getri` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../basic_array.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <cmath>
#include <complex>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `getri` routine.
   *
   * @details Computes the inverse of an \f$ n \times n \f$ matrix \f$ \mathbf{A} \f$ matrix using the LU factorization
   * computed by nda::lapack::getrf.
   *
   * This method inverts \f$ \mathbf{U} \f$ and then computes \f$ \mathrm{inv}(\mathbf{A}) \f$ by solving the system
   * \f$ \mathrm{inv}(\mathbf{A}) L = \mathrm{inv}(\mathbf{U}) \f$ for \f$ \mathrm{inv}(\mathbf{A}) \f$.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam IPIV nda::MemoryVector type.
   * @param a Input/output matrix. On entry, the factors \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ from the
   * factorization \f$ \mathbf{A} = \mathbf{P L U} \f$ as computed by nda::lapack::getrf. On exit, if `INFO == 0`, the
   * inverse of the original matrix \f$ \mathbf{A} \f$.
   * @param ipiv Input vector. The pivot indices from nda::lapack::getrf, i.e. for \f$ 1 \leq i \leq n \f$, row i of the
   * matrix was interchanged with row `ipiv(i)`.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector IPIV>
    requires(mem::have_host_compatible_addr_space<A, IPIV> and is_blas_lapack_v<get_value_t<A>> and std::is_same_v<get_value_t<IPIV>, int>)
  int getri(A &&a, IPIV const &ipiv) { // NOLINT (temporary views are allowed here)
    // check the dimensions of the input/output arrays/views
    auto const [m, n] = a.shape();
    EXPECTS(m == n);
    EXPECTS(ipiv.size() == n);

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

    // first call to get the optimal buffer size
    using value_type = get_value_t<A>;
    int info         = 0;
    value_type tmp_lwork{};
    f77::getri(n, a.data(), get_ld(a), ipiv.data(), &tmp_lwork, -1, info);
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // allocate work buffer and perform actual library call
    array<value_type, 1> work(lwork);
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    work = 0;
#endif
#endif
    f77::getri(n, a.data(), get_ld(a), ipiv.data(), work.data(), lwork, info);

    return info;
  }

} // namespace nda::lapack
