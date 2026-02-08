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
#include "../blas/tools.hpp"
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
   * @details Computes the inverse of an \f$ n \times n \f$ matrix \f$ \mathbf{A} \f$ using the LU factorization 
   * computed by nda::lapack::getrf.
   *
   * This method inverts \f$ \mathbf{U} \f$ and then computes \f$ \mathrm{inv}(\mathbf{A}) \f$ by solving the system
   * \f$ \mathrm{inv}(\mathbf{A}) L = \mathrm{inv}(\mathbf{U}) \f$ for \f$ \mathrm{inv}(\mathbf{A}) \f$.
   * 
   * @note \f$ \mathbf{A} \f$ is required to satisfy nda::mem::have_host_compatible_addr_space.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @tparam IPIV nda::blas_lapack::PivotArrayFor<A, 1> type.
   * @tparam W nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @param a Input/output matrix. On entry, the factors \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ from the
   * factorization \f$ \mathbf{A} = \mathbf{P L U} \f$ as computed by nda::lapack::getrf. On exit, the inverse of the 
   * original matrix \f$ \mathbf{A} \f$.
   * @param ipiv Input vector. The pivot indices from nda::lapack::getrf, i.e. for \f$ 1 \leq i \leq n \f$, row i of the
   * matrix was interchanged with row `ipiv(i)`.
   * @param work Ouput vector. Workspace array used by the LAPACK routine.
   * @return Integer return code from the LAPACK call.
   */
  template <BlasArray<2> A, PivotArrayFor<A, 1> IPIV, BlasArrayFor<A, 1> W = vector_value_t<A>>
    requires(mem::have_host_compatible_addr_space<A>)
  int getri(A &&a, IPIV const &ipiv, W &&work = vector_value_t<A>{}) { // NOLINT (temporary views are allowed here)
    // check the dimensions of the input/output arrays/views
    auto const [m, n] = a.shape();
    EXPECTS(m == n);
    EXPECTS(ipiv.size() == n);

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

    // first call to get the optimal buffer size
    auto tmp_lwork = get_value_t<A>{};
    int info       = 0;
    f77::getri(n, a.data(), get_ld(a), ipiv.data(), &tmp_lwork, -1, info);
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // resize/check work buffer
    resize_or_check_work_buffer(work, lwork);

    // perform actual library call
    f77::getri(n, a.data(), get_ld(a), ipiv.data(), work.data(), lwork, info);

    return info;
  }

} // namespace nda::lapack
