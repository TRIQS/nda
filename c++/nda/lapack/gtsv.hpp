// Copyright (c) 2021-2024 Simons Foundation
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
// Authors: Thomas Hahn, Nils Wentzell

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `gtsv` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `gtsv` routine.
   *
   * @details Solves the equation
   * \f[
   *   \mathbf{A} \mathbf{X} = \mathbf{B},
   * \f]
   * where \f$ \mathbf{A} \f$ is an n-by-n tridiagonal matrix, by Gaussian elimination with partial pivoting.
   *
   * Note that the equation \f$ \mathbf{A}^T \mathbf{X} = \mathbf{B} \f$ may be solved by interchanging the order of the
   * arguments containing the subdiagonal elements.
   *
   * @note If the array \f$ \mathbf{B} \f$ is a matrix in C-layout, it will create a temporary copy of it with Fortran
   * layout before calling the LAPACK routine. After the call, the result will be copied back to the original array.
   * This might be inefficient for large arrays and it is recommended to use Fortran layout for input arrays.
   *
   * @tparam DL nda::MemoryVector type.
   * @tparam D nda::MemoryVector type.
   * @tparam DU nda::MemoryVector type.
   * @tparam B nda::MemoryArray type.
   * @param dl Input/Output vector. On entry, it must contain the (n-1) subdiagonal elements of \f$ \mathbf{A} \f$. On
   * exit, it is overwritten by the (n-2) elements of the second superdiagonal of the upper triangular matrix
   * \f$ \mathbf{U} \f$ from the LU factorization of \f$ \mathbf{A} \f$.
   * @param d Input/Output vector. On entry, it must contain the diagonal elements of \f$ \mathbf{A} \f$. On exit, it is
   * overwritten by the n diagonal elements of \f$ \mathbf{U} \f$.
   * @param du Input/Output vector. On entry, it must contain the (n-1) superdiagonal elements of \f$ \mathbf{A} \f$. On
   * exit, it is overwritten by the (n-1) elements of the first superdiagonal of \f$ \mathbf{U} \f$ .
   * @param b Input/Output array. On entry, the n-by-nrhs right hand side matrix \f$ \mathbf{B} \f$. On exit, if
   * `INFO == 0`, the n-by-nrhs solution matrix \f$ \mathbf{X} \f$.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryVector DL, MemoryVector D, MemoryVector DU, MemoryArray B>
    requires(have_same_value_type_v<DL, D, DU, B> and mem::on_host<DL, D, DU, B> and is_blas_lapack_v<get_value_t<DL>>)
  int gtsv(DL &&dl, D &&d, DU &&du, B &&b) { // NOLINT (temporary views are allowed here)
    static_assert((get_rank<B> == 1 or get_rank<B> == 2), "Error in nda::lapack::gtsv: B must be an matrix/array/view of rank 1 or 2");
    static_assert(has_F_layout<B> or get_rank<B> == 1, "Error in nda::lapack::getrs: B must have Fortran layout or rank 1");

    // check the dimensions of the input/output arrays/views
    EXPECTS(dl.extent(0) == d.extent(0) - 1);
    EXPECTS(du.extent(0) == d.extent(0) - 1);
    EXPECTS(b.extent(0) == d.extent(0));

    // perform actual library call
    int info = 0;
    f77::gtsv(d.extent(0), (get_rank<B> == 2 ? b.extent(1) : 1), dl.data(), d.data(), du.data(), b.data(), get_ld(b), info);
    return info;
  }

} // namespace nda::lapack
