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
#include "../blas/qcache.hpp"

namespace nda::lapack {

  /**
   * Computes the inverse of an LU-factored general matrix.
   * The matrix m is modified during the operation.
   * The matrix is interpreted as FORTRAN orderedf getrf.
   *
   * @tparam T Element type
   * @tparam L Layout
   * @param m  matrix to be LU decomposed. It is destroyed by the operation
   * @param ipiv  Gauss Pivot, cf lapack doc
   *
   */
  template <typename M>
  int getri(M &m, array<int, 1> &ipiv) {
    static_assert(is_blas_lapack_v<typename M::value_type>, "Matrices must have the same element type and it must be double, complex ...");

    auto dm = std::min(m.extent(0), m.extent(1));
    EXPECTS(ipiv.size() >= dm);

    int info = 0;
    typename M::value_type work1[2];

    // first call to get the optimal lwork
    f77::getri(get_n_rows(m), m.data_start(), get_ld(m), ipiv.data_start(), work1, -1, info);
    int lwork;
    if constexpr (is_complex_v<typename M::value_type>)
      lwork = std::round(std::real(work1[0])) + 1;
    else
      lwork = std::round(work1[0]) + 1;

    array<typename M::value_type, 1> work(lwork);

    f77::getri(get_n_rows(m), m.data_start(), get_ld(m), ipiv.data_start(), work.data_start(), lwork, info);
    return info;
  }
} // namespace nda::lapack
