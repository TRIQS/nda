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
#include "tools.hpp"
#include "qcache.hpp"

namespace triqs::arrays::lapack {

  using namespace blas_lapack_tools;

  inline size_t r_round(double x) { return std::round(x) + 1; }
  inline size_t r_round(std::complex<double> x) { return std::round(std::real(x)) + 1; }
  /**
  * Calls getri on a matrix or view
  * Takes care of making temporary copies if necessary
  */
  template <typename MT>
  typename std::enable_if<is_blas_lapack_type<typename MT::value_type>::value, int>::type getri(MT &A, arrays::vector<int> &ipiv) {
    //getri (MT & A, arrays::vector<int> & ipiv, arrays::vector<typename MT::value_type> & work ) {
    reflexive_qcache<MT> Ca(A);
    auto dm = std::min(first_dim(Ca()), second_dim(Ca()));
    if (ipiv.size() < dm) TRIQS_RUNTIME_ERROR << "getri : error in ipiv size : found " << ipiv.size() << " while it should be at least" << dm;
    int info;
    typename MT::value_type work1[2];
    // first call to get the optimal lwork
    f77::getri(get_n_rows(Ca()), Ca().data_start(), get_ld(Ca()), ipiv.data_start(), work1, -1, info);
    int lwork = r_round(work1[0]);
    //std::cerr << " Optimisation lwork "<< lwork<< std::endl;
    arrays::vector<typename MT::value_type> work(lwork);
    f77::getri(get_n_rows(Ca()), Ca().data_start(), get_ld(Ca()), ipiv.data_start(), work.data_start(), lwork, info);
    return info;
  }
} // namespace triqs::arrays::lapack
