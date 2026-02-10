// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functions to solve linear systems of equations.
 */

#pragma once

#include "../basic_array.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../lapack/getrf.hpp"
#include "../lapack/getrs.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#include <type_traits>

namespace nda::linalg {

  /**
   * @addtogroup linalg_solve
   * @{
   */

  /**
   * @brief Solve a system of linear equations in place.
   *
   * @details The function solves a system of linear equations
   *
   * - \f$ \mathbf{A X} = \mathbf{B} \f$ or
   * - \f$ \mathbf{A x} = \mathbf{b} \f$,
   *
   * with a general \f$ n \times n \f$ matrix \f$ \mathbf{A} \f$ and either \f$ n \times n_{\mathrm{rhs}} \f$ matrices
   * \f$ \mathbf{X} \f$ and \f$ \mathbf{B} \f$ or vectors \f$ \mathbf{x} \f$ and \f$ \mathbf{b} \f$ of size \f$ n \f$.
   *
   * It uses nda::lapack::getrf to compute the LU factorization of the matrix \f$ \mathbf{A} \f$ and then
   * nda::lapack::getrs to solve the system of linear equations.
   *
   * An exception is thrown, if a LAPACK/cuSOLVER call fails.
   *
   * @note \f$ \mathbf{B} \f$ is required to have nda::F_layout.
   *
   * @tparam A nda::blas_lapack::BlasArray<2> type.
   * @tparam B nda::blas_lapack::BlasArrayFor<A> type of rank 1 or 2.
   * @param a Input/Output matrix. On entry, the \f$ n \times n \f$ matrix \f$ \mathbf{A} \f$ determining the linear
   * system. On exit, its LU factorization as calculated by nda::lapack::getrf.
   * @param b Input/Output array. On entry, the right hand side matrix \f$ \mathbf{B} \f$ or vector \f$ \mathbf{b} \f$.
   * On exit, the solution matrix \f$ \mathbf{X} \f$ or vector \f$ \mathbf{x} \f$.
   */
  template <blas_lapack::BlasArray<2> A, blas_lapack::BlasArrayFor<A> B>
    requires((get_rank<B> == 1 || get_rank<B> == 2) and blas_lapack::has_F_layout<B>)
  void solve_in_place(A &&a, B &&b) { // NOLINT (temporary views are allowed here)
    // check the dimensions of the input/output arrays/views
    EXPECTS_WITH_MESSAGE(a.extent(0) == a.extent(1), "Error in nda::linalg::solve_in_place: Matrix A is not square");
    EXPECTS_WITH_MESSAGE(a.extent(0) == b.extent(0), "Error in nda::linalg::solve_in_place: Dimension mismatch between matrix A and B");

    // pivot indices vector
    auto ipiv = vector<int, heap<mem::common_addr_space<A, B>>>(a.extent(0));

    // call getrf to compute LU factorization
    int info = lapack::getrf(a, ipiv);
    if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::solve_in_place: getrf returned a non-zero value: info = " << info;

    // call getrs to solve AX=B using the LU factorization
    info = lapack::getrs(a, b, ipiv);
    if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::linalg::solve_in_place: getrs returned a non-zero value: info = " << info;
  }

  /**
   * @brief Solve a system of linear equations.
   *
   * @details It makes a copy of the input matrix \f$ \mathbf{A} \f$ and the right hand side matrix \f$ \mathbf{B} \f$
   * or vector \f$ \mathbf{b} \f$ and calls nda::linalg::solve_in_place.
   *
   * The solution matrix \f$ \mathbf{X} \f$ is always in nda::F_layout.
   *
   * This function makes copies of the input arrays/views. When working on the device memory space, this may lead to
   * runtime errors if the copying fails.
   *
   * @note \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$/\f$ \mathbf{b} \f$ are required to satisfy
   * nda::mem::have_compatible_addr_space and to have the same value type that satisfies nda::is_blas_lapack_v.
   *
   * @tparam A nda::Matrix type.
   * @tparam B nda::Array type of rank 1 or 2.
   * @param a Input matrix. The \f$ n \times n \f$ matrix \f$ \mathbf{A} \f$ determining the linear system.
   * @param b Input array. Right hand side matrix \f$ \mathbf{B} \f$ or vector \f$ \mathbf{b} \f$.
   * @return Solution matrix \f$ \mathbf{X} \f$ or vector \f$ \mathbf{x} \f$.
   */
  template <Matrix A, Array B>
    requires(have_same_value_type_v<A, B> and mem::have_compatible_addr_space<A, B> and is_blas_lapack_v<get_value_t<A>>)
  auto solve(A const &a, B const &b) { // NOLINT (temporary views are allowed here)
    // copy A and preserve its layout
    using a_layout_policy = nda::detail::layout_to_policy<typename std::remove_cvref_t<A>::layout_t>::type;
    auto a_copy           = matrix<get_value_t<A>, a_layout_policy, heap<mem::common_addr_space<A, B>>>(a);

    // copy B and enforce Fortran layout for the matrix case
    using vector_t = vector<get_value_t<A>, heap<mem::common_addr_space<A, B>>>;
    using matrix_t = matrix<get_value_t<A>, F_layout, heap<mem::common_addr_space<A, B>>>;
    using b_type   = std::conditional_t<get_rank<B> == 1, vector_t, matrix_t>;
    auto b_copy    = b_type(b);

    // call solve_in_place with the copies
    solve_in_place(a_copy, b_copy);
    return b_copy;
  }

  /** @} */

} // namespace nda::linalg
