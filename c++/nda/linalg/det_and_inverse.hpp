// Copyright (c) 2019-2020 Simons Foundation
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

#pragma once

#include "../lapack.hpp"

namespace nda {

  // ---------- is_matrix_square  -------------------------

  template <typename A>
  bool is_matrix_square(A const &a, bool print_error = false) {
    bool r = (a.shape()[0] == a.shape()[1]);
    if (not r and print_error) std::cerr << "Error non-square matrix. Dimensions are :(" << a.shape()[0] << "," << a.shape()[1] << ")\n  " << std::endl;
    return r;
  }

  // ----------  Determinant -------------------------

  namespace impl {

    template <typename T, typename Layout>
    T determinant_from_view(matrix_view<T const, Layout> m, array<int, 1> const &ipiv) {
      T det         = 1;
      const int dim = m.extent(0);
      for (int i = 0; i < dim; i++) det *= m(i, i);
      int flip = 0; // compute the sign of the permutation
      for (int i = 0; i < dim; i++) flip += (ipiv(i) != i + 1 ? 1 : 0);
      det = ((flip % 2 == 1) ? -det : det);
      return det;
    }
  } // namespace impl

  template <typename A>
  typename A::value_type determinant(A const &m, array<int, 1> const &ipiv) REQUIRES(is_matrix_or_view_v<A>) {
    return impl::determinant_from_view(make_const_view(m), ipiv);
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

  template <typename T, typename L, typename AP, typename OP>
  void inverse_in_place(basic_array_view<T, 2, L, 'M', AP, OP> a) {
    EXPECTS(is_matrix_square(a, true));
    array<int, 1> ipiv(a.extent(0));
    int info = lapack::getrf(a, ipiv); // it is ok to be in C order. Lapack compute the inverse of the transpose.
    if (info != 0) NDA_RUNTIME_ERROR << "Inverse/Det error : matrix is not invertible. Step 1. Lapack error : " << info;
    info = lapack::getri(a, ipiv);
    if (info != 0) NDA_RUNTIME_ERROR << "Inverse/Det error : matrix is not invertible. Step 2. Lapack error : " << info;
  } // namespace nda

  template <typename T, typename L, typename CP>
  void inverse_in_place(basic_array<T, 2, L, 'M', CP> &a) {
    inverse_in_place(a());
  }

  template <typename A>
  auto inverse(A const &a) REQUIRES(is_ndarray_v<A> and (get_algebra<A> == 'M') and (get_rank<A> == 2)) {
    EXPECTS(is_matrix_square(a, true));
    auto r = make_regular(a);
    inverse_in_place(r);
    return r;
  }

} // namespace nda

namespace nda::clef {
  CLEF_MAKE_FNT_LAZY(determinant)
}
