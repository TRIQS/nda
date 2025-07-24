// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `hegv` routine.
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
   * @brief Interface to the LAPACK `hegv` routine.
   *
   * @details Computes all eigenvalues \f$ \lambda_i \f$ and, optionally, eigenvectors \f$ \mathbf{v}_i \f$ of a complex 
   * generalized Hermitian-definite eigenvalue problem of the form
   * - \f$ \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{B} \mathbf{v}_i \f$ (`itype = 1`),
   * - \f$ \mathbf{A} \mathbf{B} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 2`) or
   * - \f$ \mathbf{B} \mathbf{A} \mathbf{v}_i = \lambda_i \mathbf{v}_i \f$ (`itype = 3`).
   *
   * Here \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are assumed to be Hermitian and \f$ \mathbf{B} \f$ is also positive 
   * definite.
   *
   * @tparam A nda::MemoryMatrix with complex value type.
   * @tparam B nda::MemoryMatrix with complex value type.
   * @param a Input/output matrix. On entry, the Hermitian matrix \f$ \mathbf{A} \f$. On exit, if `jobz = V`, \f$ 
   * \mathbf{A} \f$ contains the matrix \f$ \mathbf{V} \f$ of normalized eigenvectors such that \f$ \mathbf{V}^H 
   * \mathbf{B} \mathbf{V} = \mathbf{I} \f$ (if `itype = 1` or `itype = 2`) or \f$ \mathbf{V}^H \mathbf{B}^{-1} 
   * \mathbf{V} = \mathbf{I} \f$ (if `itype = 3`). If `jobz = N`, then on exit \f$ \mathbf{A} \f$ is destroyed.
   * @param b Input/output matrix. On entry, the symmetric positive definite matrix \f$ \mathbf{B} \f$. On exit, the 
   * part of \f$ \mathbf{B} \f$ containing the matrix is overwritten by the triangular factor \f$ \mathbf{U} \f$ or 
   * \f$ \mathbf{L} \f$ from a Cholesky factorization.
   * @param w Output vector. The eigenvalues in ascending order.
   * @param jobz Character indicating whether to compute eigenvectors and eigenvalues ('V') or eigenvalues only ('N').
   * @param itype Specifies the problem to be solved.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryMatrix B, MemoryVector W>
    requires(mem::have_host_compatible_addr_space<A, B> and is_complex_v<get_value_t<A>> and have_same_value_type_v<A, B>
             and std::same_as<get_fp_t<A>, get_value_t<W>>)
  int hegv(A &&a, B &&b, W &&w, char jobz = 'V', int itype = 1) { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A> and has_F_layout<B>, "Error in nda::lapack::hegv: A and B must have Fortran layout");

    // check the dimensions of the input/output arrays/views and resize if necessary
    auto const [m, n] = a.shape();
    EXPECTS(m == n);
    EXPECTS(m == b.shape()[0]);
    EXPECTS(n == b.shape()[1]);
    resize_or_check_if_view(w, {n});

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(w.indexmap().min_stride() == 1);

    // check other input parameters for consistency
    EXPECTS(itype == 1 or itype == 2 or itype == 3);
    EXPECTS(jobz == 'V' or jobz == 'N');

    // first call to get the optimal buffer size
    using fp_type = get_fp_t<A>;
    array<fp_type, 1> rwork(std::max(1l, 3 * n - 2));
    std::complex<fp_type> tmp_lwork{};
    int info = 0;
    lapack::f77::hegv(itype, jobz, 'U', n, a.data(), get_ld(a), b.data(), get_ld(b), w.data(), &tmp_lwork, -1, rwork.data(), info);
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // allocate work buffer and perform actual library call
    array<std::complex<fp_type>, 1> work(lwork);
    lapack::f77::hegv(itype, jobz, 'U', n, a.data(), get_ld(a), b.data(), get_ld(b), w.data(), work.data(), lwork, rwork.data(), info);

    return info;
  }

} // namespace nda::lapack
