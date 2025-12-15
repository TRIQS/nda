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
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <algorithm>
#include <cmath>
#include <complex>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `gelss` routine.
   *
   * @details Computes the minimum norm solution to a complex linear least squares problem:
   * \f[
   *   \min_x | \mathbf{b} - \mathbf{A x} |_2
   * \f]
   * using the singular value decomposition (SVD) of \f$ \mathbf{A} \f$. \f$ \mathbf{A} \f$ is an m-by-n matrix which
   * may be rank-deficient.
   *
   * Several right hand side vectors \f$ \mathbf{b} \f$ and solution vectors \f$ \mathbf{x} \f$ can be handled in a
   * single call; they are stored as the columns of the m-by-nrhs right hand side matrix \f$ \mathbf{B} \f$ and the
   * n-by-nrhs solution matrix \f$ \mathbf{X} \f$.
   *
   * The effective rank of \f$ \mathbf{A} \f$ is determined by treating as zero those singular values which are less
   * than `rcond` times the largest singular value.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam B nda::MemoryArray type.
   * @tparam S nda::MemoryVector type.
   * @param a Input/output matrix. On entry, the m-by-n matrix \f$ \mathbf{A} \f$. On exit, the first `min(m,n)` rows of
   * \f$ \mathbf{A} \f$ are overwritten with its right singular vectors, stored rowwise.
   * @param b Input/output array. On entry, the m-by-nrhs right hand side matrix \f$ \mathbf{B} \f$. On exit,
   * \f$ \mathbf{B} \f$ is overwritten by the n-by-nrhs solution matrix \f$ \mathbf{X} \f$. If `m >= n` and `RANK == n`,
   * the residual sum-of-squares for the solution in the i-th column is given by the sum of squares of the modulus of
   * elements `n+1:m` in that column.
   * @param s Output vector. The singular values of \f$ \mathbf{A} \f$ in decreasing order. The condition number of A in
   * the 2-norm is `s(1)/s(min(m,n))`.
   * @param rcond It is used to determine the effective rank of \f$ \mathbf{A} \f$. Singular values `s(i) <= rcond *
   * s(1)` are treated as zero. If `rcond < 0`, machine precision is used instead.
   * @param rank Output variable of the effective rank of \f$ \mathbf{A} \f$, i.e., the number of singular values which
   * are greater than `rcond * s(1)`.
   * @return Integer return code.
   */
  template <MemoryMatrix A, MemoryArray B, MemoryVector S>
    requires(have_same_value_type_v<A, B> and mem::on_host<A, B, S> and is_blas_lapack_v<get_value_t<A>>)
  int gelss(A &&a, B &&b, S &&s, remove_complex_t<get_value_t<A>> rcond, int &rank) { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A> and has_F_layout<B>, "Error in nda::lapack::gelss: C order not supported");
    static_assert(MemoryVector<B> or MemoryMatrix<B>, "Error in nda::lapack::gelss: B must be a vector or a matrix");

    auto dm = std::min(a.extent(0), a.extent(1));
    if (s.size() < dm) s.resize(dm);

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(s.indexmap().min_stride() == 1);

    // first call to get the optimal bufferSize
    using value_type = get_value_t<A>;
    value_type bufferSize_T{};
    auto rwork = array<remove_complex_t<value_type>, 1>(5 * dm);
    int info   = 0;
    int nrhs = 1, ldb = b.size(); // defaults for B MemoryVector
    if constexpr (MemoryMatrix<B>) {
      nrhs = b.extent(1);
      ldb  = get_ld(b);
    }
    f77::gelss(a.extent(0), a.extent(1), nrhs, a.data(), get_ld(a), b.data(), ldb, s.data(), rcond, rank, &bufferSize_T, -1, rwork.data(), info);
    int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

    // allocate work buffer and perform actual library call
    array<value_type, 1> work(bufferSize);
    f77::gelss(a.extent(0), a.extent(1), nrhs, a.data(), get_ld(a), b.data(), ldb, s.data(), rcond, rank, work.data(), bufferSize, rwork.data(),
               info);

    if (info) NDA_RUNTIME_ERROR << "Error in nda::lapack::gelss: info = " << info;
    return info;
  }

} // namespace nda::lapack
