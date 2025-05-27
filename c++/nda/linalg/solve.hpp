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
   * @addtogroup linalg_tools
   * @{
   */

  /**
   * @brief Solve a system of linear equations.
   *
   * @details The function solves a system of linear equations
   *
   * - \f$ \mathbf{A X} = \mathbf{B} \f$ or
   * - \f$ \mathbf{A x} = \mathbf{b} \f$,
   *
   * with a general \f$ n \times n \f$ matrix \f$ \mathbf{A} \f$ and either
   * 
   * - \f$ n \times m \f$  matrices \f$ \mathbf{X} \f$ and \f$ \mathbf{B} \f$ or 
   * - vectors \f$ \mathbf{x} \f$ and \f$ \mathbf{b} \f$ of size \f$ n \f$.
   *
   * It uses nda::lapack::getrf to compute the LU factorization of the matrix \f$ \mathbf{A} \f$ and then
   * nda::lapack::getrs to solve the system of linear equations.
   *
   * An exception is thrown, if the LAPACK calls return a non-zero value.
   *
   * @note Right hand side matrix \f$ \mathbf{B} \f$ must have nda::F_layout.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam B nda::MemoryArray type of rank 1 or 2.
   * @param a Input/Output matrix. On entry, the \f$ n \times n \f$ matrix \f$ \mathbf{A} \f$ determining the linear
   * system. On exit, the result of the LU factorization from nda::lapack::getrf.
   * @param b Input/Output matrix. On entry, the right hand side matrix \f$ \mathbf{B} \f$ (vector \f$ \mathbf{b} \f$). 
   * On exit, the solution matrix \f$ \mathbf{X} \f$ (vector \f$ \mathbf{x} \f$).
   */
  template <MemoryMatrix A, MemoryArray B>
    requires(have_same_value_type_v<A, B> and nda::mem::have_compatible_addr_space<A, B> and is_blas_lapack_v<get_value_t<A>>)
  void solve_in_place(A &&a, B &&b) { // NOLINT (temporary views are allowed here)
    constexpr auto addr_space = nda::mem::common_addr_space<A, B>;

    // check the dimensions of the input/output arrays/views
    EXPECTS_WITH_MESSAGE(a.shape()[0] == a.shape()[1], "Error in nda::solve_in_place: Matrix A is not square");
    EXPECTS_WITH_MESSAGE(a.shape()[0] == b.extent(0), "Error in nda::solve_in_place: Dimension mismatch between matrix A and B");

    // pivot indices vector
    auto ipiv = vector<int, heap<addr_space>>(a.extent(0));

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
   * - \f$ \mathbf{A x} = \mathbf{b} \f$,
   * 
   * with a general \f$ n \times n \f$ matrix \f$ \mathbf{A} \f$ and either
   *
   * - \f$ n \times m \f$ matrices \f$ \mathbf{X} \f$ and \f$ \mathbf{B} \f$ or 
   * - vectors \f$ \mathbf{x} \f$ and \f$ \mathbf{b} \f$ of size \f$ n \f$.
   *
   * It calls nda::linalg::solve_in_place with a copy of the input matrix \f$ \mathbf{A} \f$ and input matrix/vector \f$
   * \mathbf{B} \f$/\f$ \mathbf{b} \f$.
   *
   * @note The solution matrix \f$ \mathbf{X} \f$ is always in nda::F_layout.
   * 
   * @warning This function makes copies of the input arrays/views. When working on the device memory space, this may
   * lead to runtime errors if the copying fails.
   *
   * @tparam A nda::Matrix type.
   * @tparam B nda::Array type of rank 1 or 2.
   * @param a Input matrix. The \f$ n \times n \f$ matrix \f$ \mathbf{A} \f$ determining the linear system.
   * @param b Input matrix. The right hand side matrix \f$ \mathbf{B} \f$ (vector \f$ \mathbf{b} \f$).
   * @return Solution matrix \f$ \mathbf{X} \f$ (vector \f$ \mathbf{x} \f$).
   */
  template <Matrix A, Array B>
    requires(have_same_value_type_v<A, B> and nda::mem::have_compatible_addr_space<A, B> and is_blas_lapack_v<get_value_t<A>>)
  auto solve(A const &a, B const &b) { // NOLINT (temporary views are allowed here)
    // copy A and preserve its layout
    using a_layout_policy = nda::detail::layout_to_policy<typename std::remove_cvref_t<A>::layout_t>::type;
    auto a_copy           = matrix<get_value_t<A>, a_layout_policy, heap<nda::mem::common_addr_space<A, B>>>(a);

    // copy B and enforce Fortran layout if it is a matrix
    using vector_t = vector<get_value_t<A>, heap<nda::mem::common_addr_space<A, B>>>;
    using matrix_t = matrix<get_value_t<A>, F_layout, heap<nda::mem::common_addr_space<A, B>>>;
    using b_type   = std::conditional_t<get_rank<B> == 1, vector_t, matrix_t>;
    auto b_copy    = b_type(b);

    // call solve_in_place with the copies
    solve_in_place(a_copy, b_copy);
    return b_copy;
  }

  /** @} */

} // namespace nda::linalg
