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
   * Calls getri on a matrix or view
   * @tparam M
   * @param m
   * @param ipiv
   */
  template <typename M>
  int getri(M &m, array<int, 1> &ipiv) {
    static_assert(is_blas_lapack_v<typename M::value_type>, "Matrices must have the same element type and it must be double, complex ...");

    auto Ca = reflexive_qcache(m);
    auto dm = std::min(Ca().extent(0), Ca().extent(1));
    if (ipiv.size() < dm) NDA_RUNTIME_ERROR << "getri : error in ipiv size : found " << ipiv.size() << " while it should be at least" << dm;

    int info;
    typename M::value_type work1[2];
    // first call to get the optimal lwork
    f77::getri(get_n_rows(Ca()), Ca().data_start(), get_ld(Ca()), ipiv.data_start(), work1, -1, info);
    int lwork;
    if constexpr (is_complex_v<typename M::value_type>)
      lwork = std::round(std::real(work1[0])) + 1;
    else
      lwork = std::round(work1[0]) + 1;

    array<typename M::value_type, 1> work(lwork);

    f77::getri(get_n_rows(Ca()), Ca().data_start(), get_ld(Ca()), ipiv.data_start(), work.data_start(), lwork, info);
    return info;
  }
} // namespace nda::lapack
