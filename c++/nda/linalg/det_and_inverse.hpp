// Copyright (c) 2019-2021 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once

#include "../lapack.hpp"
#include "../layout_transforms.hpp"

namespace nda {

  // ---------- is_matrix_square  -------------------------

  template <typename A>
  bool is_matrix_square(A const &a, bool print_error = false) {
    bool r = (a.shape()[0] == a.shape()[1]);
    if (not r and print_error)
      std::cerr << "Error non-square matrix. Dimensions are :(" << a.shape()[0] << "," << a.shape()[1] << ")\n  " << std::endl;
    return r;
  }

  // ----------  Determinant -------------------------

  template <typename M>
  auto determinant_in_place(M &m) requires(is_matrix_or_view_v<M>) {
    using value_t = get_value_t<M>;
    static_assert(std::is_convertible_v<value_t, double> or std::is_convertible_v<value_t, std::complex<double>>,
	"determinant requires a matrix of values that can be implicitly converted to double or std::complex<double>");
    static_assert(not std::is_const_v<M>, "determinant_in_place can not be const. It destroys its argument");

    if(m.empty()) return value_t{1};

    if(m.extent(0) != m.extent(1))
      NDA_RUNTIME_ERROR << "Error in determinant. Matrix is not square but has shape " << m.shape();
    const int dim = m.extent(0);

    // Calculate the LU decomposition using lapack getrf
    basic_array<int, 1, C_layout, 'A', sso<100>> ipiv(dim);
    int info = lapack::getrf(m, ipiv); // it is ok to be in C order. Lapack compute the inverse of the transpose.
    if (info < 0) NDA_RUNTIME_ERROR << "Error in determinant. Info lapack is " << info;

    // Calculate the determinant from the LU decomposition
    auto det = value_t{1};
    int n_flips = 0;
    for (int i = 0; i < dim; i++){
      det *= m(i, i);
      // Count the number of column interchanges performed by getrf
      if(ipiv(i) != i + 1) ++n_flips;
    }

    return ((n_flips % 2 == 1) ? -det : det);
  }

  template <typename M>
  auto determinant(M const &m) {
    auto m_copy = make_regular(m);
    return determinant_in_place(m_copy);
  }

  // For small matrices (2x2 and 3x3), we directly
  // compute the matrix inversion rather than calling the
  // LaPack routine
  // ---------- Small Inverse Benchmarks ---------
  //          Run on (16 X 2400 MHz CPUs) (see benchmarks/small_inv.cpp)
  // ---------------------------------------------
  // Matrix Size      Time (old)        Time (new)
  //     1             502 ns            59.0 ns
  //     2             595 ns            61.7 ns
  //     3             701 ns            67.5 ns
  
  // ----------  inverse (1x1) ---------------------
  template <typename T, typename L, typename AP, typename OP>
  void inverse1_in_place(basic_array_view<T, 2, L, 'M', AP, OP> a) {
    if (a(0,0) == 0.0) NDA_RUNTIME_ERROR << "Inverse/Det error : matrix is not invertible.";
    a(0,0) = 1.0/a(0,0);
  }
  
  // ----------  inverse (2x2) ---------------------
  template <typename T, typename L, typename AP, typename OP>
  void inverse2_in_place(basic_array_view<T, 2, L, 'M', AP, OP> a) {

    // calculate the adjoint of the matrix
    std::swap(a(0,0), a(1,1));

    // calculate the inverse determinant of the matrix
    auto det = (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0));
    if (det == 0.0) NDA_RUNTIME_ERROR << "Inverse/Det error : matrix is not invertible.";
    auto detinv = 1.0/det;

    a(0,0) *= +detinv; 
    a(1,1) *= +detinv;
    a(1,0) *= -detinv;
    a(0,1) *= -detinv;
  }

  // ----------  inverse (3x3) ---------------------
  template <typename T, typename L, typename AP, typename OP>
  void inverse3_in_place(basic_array_view<T, 2, L, 'M', AP, OP> a) {

    // calculate the adjoint of the matrix
    auto b00 = +a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1);
    auto b10 = -a(1, 0) * a(2, 2) + a(1, 2) * a(2, 0);
    auto b20 = +a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0);
    auto b01 = -a(0, 1) * a(2, 2) + a(0, 2) * a(2, 1);
    auto b11 = +a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0);
    auto b21 = -a(0, 0) * a(2, 1) + a(0, 1) * a(2, 0);
    auto b02 = +a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1);
    auto b12 = -a(0, 0) * a(1, 2) + a(0, 2) * a(1, 0);
    auto b22 = +a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);

    // calculate the inverse determinant of the matrix
    auto det = a(0,0)*b00 + a(0,1)*b10 + a(0,2)*b20;
    if (det == 0.0) NDA_RUNTIME_ERROR << "Inverse/Det error : matrix is not invertible.";
    auto detinv = 1.0/det;

    // fill the matrix
    a(0,0) = detinv * b00; a(0,1) = detinv * b01; a(0,2) = detinv * b02;
    a(1,0) = detinv * b10; a(1,1) = detinv * b11; a(1,2) = detinv * b12;
    a(2,0) = detinv * b20; a(2,1) = detinv * b21; a(2,2) = detinv * b22;
  }

  // ----------  inverse ----------------
  template <typename T, typename L, typename AP, typename OP>
  void inverse_in_place(basic_array_view<T, 2, L, 'M', AP, OP> a) {
    EXPECTS(is_matrix_square(a, true));
    if (a.empty()) return;

    if (a.shape()[0] == 1) {
      inverse1_in_place(a);
      return;
    }

    if (a.shape()[0] == 2) {
      inverse2_in_place(a);
      return;
    }

    if (a.shape()[0] == 3) {
      inverse3_in_place(a);
      return;
    }

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

  template <Array A>
  auto inverse(A const &a) requires(get_algebra<A> == 'M') {
    static_assert(get_rank<A> == 2, "inverse: array must have rank two");
    EXPECTS(is_matrix_square(a, true));
    auto r = make_regular(a);
    inverse_in_place(r);
    return r;
  }

  template <Array A>
  auto inverse(A const &a) {
      auto r = make_regular(a);
      auto long_axis = stdutil::mpop<2>(r.indexmap().lengths());
      for_each(long_axis, [&r, _ = range()](auto &&... i) {inverse_in_place(make_matrix_view(r(i..., _, _))); });
      return r;
  }

} // namespace nda

namespace nda::clef {
  CLEF_MAKE_FNT_LAZY(determinant)
}
