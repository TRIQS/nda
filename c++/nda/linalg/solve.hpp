// Copyright (c) 2019-2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides solver functions for linear systems of equations.
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

namespace nda {

  /**
   * @addtogroup linalg_tools
   * @{
   */

  /**
   * @brief Solve a system of linear equations in place.
   *
   * @details The function solves a system of linear equations
   *
   * - \f$ \mathbf{A X} = \mathbf{B} \f$ or
   * - \f$ \mathbf{A} \mathbf{x} = \mathbf{b} \f$,
   *
   * with a general n-by-n matrix \f$ \mathbf{A} \f$ and n-by-m  matrices \f$ \mathbf{X} \f$ and \f$ \mathbf{B} \f$ or
   * vectors \f$ \mathbf{x} \f$ and \f$ \mathbf{b} \f$.
   *
   * It uses nda::lapack::getrf to compute the LU factorization of the matrix \f$ \mathbf{A} \f$ and then
   * nda::lapack::getrs to solve the system of linear equations. An exception is thrown, if the LAPACK calls return a
   * non-zero value.
   *
   * @note If the right hand side is a C-layout matrix \f$ \mathbf{B} \f$, it will create a temporary copy with Fortran
   * layout inside the nda::lapack::getrs call, which is then copied back into the original matrix. This might be
   * inefficient for large matrices and it is recommended to use Fortran layout.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam B nda::MemoryArray type of rank 1 or 2.
   * @param a Input/output matrix. On entry, the left hand side matrix \f$ \mathbf{A} \f$. On exit, the result from the
   * nda::lapack::getrf call.
   * @param b Input/output matrix. On entry, the right hand side matrix \f$ \mathbf{B} \f$ (vector \f$ \mathbf{b} \f$).
   * On exit, the solution matrix \f$ \mathbf{X} \f$ (vector \f$ \mathbf{x} \f$).
   */
  template <MemoryMatrix A, MemoryArray B>
    requires(have_same_value_type_v<A, B> and mem::have_compatible_addr_space<A, B> and is_blas_lapack_v<get_value_t<A>>)
  void solve_in_place(A &&a, B &&b) { // NOLINT (temporary views are allowed here)
    constexpr auto addr_space = mem::common_addr_space<A, B>;

    // check dimensions
    EXPECTS_WITH_MESSAGE(a.shape()[0] == a.shape()[1], "Error in nda::solve_in_place: Matrix A is not square");
    EXPECTS_WITH_MESSAGE(a.shape()[1] == b.shape()[0], "Error in nda::solve_in_place: Dimension mismatch between matrix A and B");

    // pivot indices vector
    auto ipiv = vector<int, heap<addr_space>>(a.shape()[0]);

    // call lapack getrf
    int info = lapack::getrf(a, ipiv);
    if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::solve_in_place: getrf returned a non-zero value: info = " << info;

    // call lapack getrs
    info = lapack::getrs(a, b, ipiv);
    if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::solve_in_place: getrs returned a non-zero value: info = " << info;
  }

  /**
   * @brief Solve a system of linear equations.
   *
   * @details The function solves a system of linear equations
   *
   * - \f$ \mathbf{A X} = \mathbf{B} \f$ or
   * - \f$ \mathbf{A} \mathbf{x} = \mathbf{b} \f$,
   *
   * with a general n-by-n matrix \f$ \mathbf{A} \f$ and n-by-m  matrices \f$ \mathbf{X} \f$ and \f$ \mathbf{B} \f$ or
   * vectors \f$ \mathbf{x} \f$ and \f$ \mathbf{b} \f$.
   *
   * It makes a copy of the input matrix \f$ \mathbf{A} \f$ and the input matrix/vector \f$ \mathbf{B} \f$/\f$
   * \mathbf{b} \f$ and then calls nda::solve_in_place with the copies.
   *
   * @note If the right hand side is a matrix, the solution matrix \f$ \mathbf{X} \f$ is always returned in Fortran
   * layout.
   *
   * @tparam A nda::Matrix type.
   * @tparam B nda::Array type of rank 1 or 2.
   * @param a Left hand side matrix \f$ \mathbf{A} \f$.
   * @param b Right hand side matrix \f$ \mathbf{B} \f$ (vector \f$ \mathbf{b} \f$).
   * @return Solution matrix \f$ \mathbf{X} \f$ (vector \f$ \mathbf{x} \f$).
   */
  template <Matrix A, Array B>
    requires(have_same_value_type_v<A, B> and is_blas_lapack_v<get_value_t<A>>)
  auto solve(A const &a, B const &b) { // NOLINT (temporary views are allowed here)
    using b_type = std::conditional_t<get_rank<B> == 1, vector<get_value_t<B>>, matrix<get_value_t<B>, F_layout>>;
    auto a_copy  = matrix<get_value_t<A>, F_layout>(a);
    auto b_copy  = b_type(b);
    solve_in_place(a_copy, b_copy);
    return b_copy;
  }

  /** @} */

} // namespace nda
