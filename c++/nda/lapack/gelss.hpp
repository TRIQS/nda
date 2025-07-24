// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `gelss` routine.
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
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `gelss` routine.
   *
   * @details Computes the minimum norm solution to a complex linear least squares problem:
   * \f[
   *   \min_{\mathbf{x}} | \mathbf{b} - \mathbf{A x} |_2
   * \f]
   * using the singular value decomposition (SVD) of \f$ \mathbf{A} \f$. \f$ \mathbf{A} \f$ is an \f$ m \times n \f$
   * matrix which may be rank-deficient.
   *
   * Several right hand side vectors \f$ \mathbf{b} \f$ and solution vectors \f$ \mathbf{x} \f$ can be handled in a
   * single call; they are stored as the columns of the \f$ m \times n_{\mathrm{rhs}} \f$ right hand side matrix \f$
   * \mathbf{B} \f$ and the \f$ n \times n_{\mathrm{rhs}} \f$ solution matrix \f$ \mathbf{X} \f$.
   *
   * The effective rank of \f$ \mathbf{A} \f$ is determined by treating as zero those singular values which are less
   * than \f$ r_{\mathrm{cond}} \f$ times the largest singular value.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam B nda::MemoryArray type.
   * @tparam S nda::MemoryVector type.
   * @param a Input/output matrix. On entry, the \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$. On exit, the first \f$
   * \min(m,n) \f$ rows of \f$ \mathbf{A} \f$ are overwritten with its right singular vectors, stored rowwise.
   * @param b Input/output array. On entry, the \f$ m \times n_{\mathrm{rhs}} \f$ right hand side matrix \f$ \mathbf{B}
   * \f$. On exit, \f$ \mathbf{B} \f$ is overwritten by the \f$ n \times n_{\mathrm{rhs}} \f$ solution matrix \f$
   * \mathbf{X} \f$. If \f$ m \geq n \f$ and if the effective rank is equal \f$ n \f$, the residual sum-of-squares for
   * the solution in the i<sup>th</sup> column is given by the sum of squares of the modulus of elements \f$ n + 1 \f$
   * to \f$ m \f$ in that column.
   * @param s Output vector. The singular values of \f$ \mathbf{A} \f$ in decreasing order. The condition number of \f$
   * \mathbf{A} \f$ in the 2-norm is \f$ s_1 / s_{min(m,n)} \f$.
   * @param rcond It is used to determine the effective rank of \f$ \mathbf{A} \f$. Singular values \f$ s_i \leq
   * r_{\mathrm{cond}} s_1 \f$ are treated as zero. If \f$ r_{\mathrm{cond}} < 0 \f$, machine precision is used instead.
   * @param rank Output variable. The effective rank of \f$ \mathbf{A} \f$, i.e. the number of singular values which
   * are greater than \f$ r_{\mathrm{cond}} s_1 \f$.
   * @return Integer return code.
   */
  template <MemoryMatrix A, MemoryArray B, MemoryVector S>
    requires(have_same_value_type_v<A, B> and mem::have_host_compatible_addr_space<A, B, S> and is_blas_lapack_v<get_value_t<A>>)
  int gelss(A &&a, B &&b, S &&s, get_fp_t<A> rcond, int &rank) { // NOLINT (temporary views are allowed here)
    static_assert(std::is_same_v<get_value_t<S>, float> || std::is_same_v<get_value_t<S>, double>,
                  "Error in nda::lapack::gelss: Singular value array must have elements of type float or double");
    static_assert(has_F_layout<A> and has_F_layout<B>, "Error in nda::lapack::gelss: Matrices/arrays must have Fortran layout");
    static_assert(get_rank<B> == 1 || get_rank<B> == 2, "Error in nda::lapack::gelss: Right hand side must have rank 1 or 2");

    // check the dimensions of the input/output arrays/views and resize if necessary
    auto const [m, n] = a.shape();
    auto const k      = std::min(m, n);
    resize_or_check_if_view(s, {k});
    EXPECTS(b.extent(0) == m);

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(s.indexmap().min_stride() == 1);

    // first call to get the optimal buffer size
    using value_type = get_value_t<A>;
    using fp_type    = get_fp_t<A>;
    value_type tmp_lwork{};
    auto rwork = array<fp_type, 1>(5 * k);
    int info   = 0;
    int nrhs   = (get_rank<B> == 2 ? b.extent(1) : 1);
    f77::gelss(m, n, nrhs, a.data(), get_ld(a), b.data(), get_ld(b), s.data(), rcond, rank, &tmp_lwork, -1, rwork.data(), info);
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // allocate work buffer and perform actual library call
    array<value_type, 1> work(lwork);
    f77::gelss(m, n, nrhs, a.data(), get_ld(a), b.data(), get_ld(b), s.data(), rcond, rank, work.data(), lwork, rwork.data(), info);

    return info;
  }

} // namespace nda::lapack
