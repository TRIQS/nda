/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2012-2017 by O. Parcollet
 * Copyright (C) 2018 by Simons Foundation
 *   author : O. Parcollet, P. Dumitrescu, N. Wentzell
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once

#include <complex>
#include "f77/cxx_interface.hpp"
#include "../blas/tools.hpp"

namespace nda::lapack {

  using blas::get_ld;
  using blas::get_n_cols;
  using blas::get_n_rows;
  using blas::is_blas_lapack_v;

  /**
   * LU decomposition of a matrix_view
   * The matrix m is modified during the operation.
   * The matrix is interpreted as FORTRAN ordered, and then LU decomposed.
   * NB : for some operation, like det, inversion, it is fine to be transposed, 
   *      for some it may not be ... 
   *
   * @tparam T Element type
   * @tparam L Layout
   * @param m  matrix to be LU decomposed. It is destroyed by the operation
   * @param ipiv  Gauss Pivot, cf lapack doc
   *
   */
  template <typename T, typename L>
  int getrf(matrix_view<T, L> &m, array<int, 1> &ipiv) {
    static_assert(is_blas_lapack_v<T>, "Matrices must have the same element type and it must be double, complex ...");
    auto dm = std::min(m.extent(0), m.extent(1));
    if (ipiv.size() < dm) ipiv.resize(dm);
    int info = 0;
    f77::getrf(get_n_rows(m), get_n_cols(m), m.data_start(), get_ld(m), ipiv.data_start(), info);
    return info;
  }

} // namespace nda::lapack
