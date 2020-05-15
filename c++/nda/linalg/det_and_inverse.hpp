/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011 by O. Parcollet
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

#include "../lapack.hpp"

namespace nda {

  // ---------- is_matrix_square  -------------------------

  template <typename A>
  bool is_matrix_square(A const &a, bool print_error = false) {
    bool r = (a.extent(0) == a.extent(1));
    if (not r and print_error) std::cerr << "Error non-square matrix. Dimensions are :(" << a.extent(0) << "," << a.extent(1) << ")\n  " << std::endl;
    return r;
  }

  // ----------  Determinant -------------------------

  template <typename T, typename Layout>
  T determinant_impl(matrix_view<T const, Layout> m, array<int, 1> const &ipiv) {
    T det         = 1;
    const int dim = m.extent(0);
    for (int i = 0; i < dim; i++) det *= m(i, i);
    int flip = 0; // compute the sign of the permutation
    for (int i = 0; i < dim; i++) flip += (ipiv(i) != i + 1 ? 1 : 0);
    det = ((flip % 2 == 1) ? -det : det);
    return det;
  }

  template <typename A>
  typename A::value_type determinant(A const &m, array<int, 1> const &ipiv) NDA_REQUIRES(is_matrix_or_view_v<A>) {
    return determinant_impl(make_const_view(m), ipiv);
  }

  template <typename A>
  auto determinant_in_place(A &a) {
    static_assert(not std::is_const_v<A>, "determinant_in_place can not be const. It destroys its argument");
    array<int, 1> ipiv(a.extent(0));
    int info = lapack::getrf(a, ipiv); // it is ok to be in C order. Lapack compute the inverse of the transpose.
    if (info != 0) NDA_RUNTIME_ERROR << "Error in determinant. Info lapack is" << info;
    return determinant(a, ipiv);
  }

  template <typename A>
  auto determinant(A const &a) {
    auto a_copy = make_regular(a);
    return determinant_in_place(a_copy);
  }

  // ----------  inverse -------------------------

  template <typename A>
  void inverse_in_place(A &a) NDA_REQUIRES(is_regular_or_view_v<A> and (get_algebra<A> == 'M') and (get_rank<A> == 2)) {
    EXPECTS(is_matrix_square(a, true));
    array<int, 1> ipiv(a.extent(0));
    int info = lapack::getrf(a, ipiv); // it is ok to be in C order. Lapack compute the inverse of the transpose.
    if (info != 0) NDA_RUNTIME_ERROR << "Inverse/Det error : matrix is not invertible. Step 1. Lapack error : " << info;
    info = lapack::getri(a, ipiv);
    if (info != 0) NDA_RUNTIME_ERROR << "Inverse/Det error : matrix is not invertible. Step 2. Lapack error : " << info;
  }

  template <typename A>
  auto inverse(A &a) NDA_REQUIRES(is_ndarray_v<A> and (get_algebra<A> == 'M') and (get_rank<A> == 2)) {
    EXPECTS(is_matrix_square(a, true));
    auto r = make_regular(a);
    inverse_in_place(r);
    return r;
  }

} // namespace nda

namespace clef {
  CLEF_MAKE_FNT_LAZY(determinant)
}
