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

  using blas::is_blas_lapack_v;
  using blas::get_n_cols;
  using blas::get_n_rows;
  using blas::get_ld;

  /**
   * Calls getrf on a matrix or view
   * @tparam M
   * @param m
   * @param ipiv
   * @param assert_fortran_order Ensure the matrix is in Fortran Order 
   */
  template <typename M>
  int getrf(M &m, array<int, 1> &ipiv, bool assert_fortran_order = false) REQUIRES(is_regular_or_view_v<M> and (M::rank == 2)) {
    static_assert(is_blas_lapack_v<typename M::value_type>, "Matrices must have the same element type and it must be double, complex ...");
    if (assert_fortran_order && m.indexmap().is_stride_order_Fortran()) NDA_RUNTIME_ERROR << "matrix passed to getrf is not in Fortran order";
    auto Ca = reflexive_qcache(m);
    auto dm = std::min(Ca().extent(0), Ca().extent(1));
    if (ipiv.size() < dm) ipiv.resize(dm);
    int info;
    f77::getrf(get_n_rows(Ca()), get_n_cols(Ca()), Ca().data_start(), get_ld(Ca()), ipiv.data_start(), info);
    return info;
  }

} // namespace nda::lapack
