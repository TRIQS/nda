// Copyright (c) 2022 Simons Foundation
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
// Authors: Nils Wentzell

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <algorithm>
#include <complex>

// Test the CULAPACK gesvd function.
template <typename value_t>
void test_gesvd() {
  using matrix_t = nda::matrix<value_t, nda::F_layout>;

  auto A      = matrix_t{{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}}};
  auto [m, n] = A.shape();

  auto U  = matrix_t(m, m);
  auto VT = matrix_t(n, n);

  auto s = nda::vector<double>(std::min(m, n));

  auto A_d  = to_device(A);
  auto s_d  = to_device(s);
  auto U_d  = to_device(U);
  auto VT_d = to_device(VT);
  nda::lapack::gesvd(A_d, s_d, U_d, VT_d);
  s  = s_d;
  U  = U_d;
  VT = VT_d;

  auto Sigma = matrix_t::zeros(A.shape());
  for (auto i : nda::range(std::min(m, n))) Sigma(i, i) = s(i);
  EXPECT_ARRAY_NEAR(A, U * Sigma * VT, 1e-14);
}

TEST(NDA, CULAPACKGesvd) {
  test_gesvd<double>();
  test_gesvd<std::complex<double>>();
}

// Test the CULAPACK getrs and getrf functions.
template <typename value_t>
void test_getrs_getrf() {
  using matrix_t = nda::matrix<value_t, nda::F_layout>;

  auto A = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto B = matrix_t{{1, 5}, {4, 5}, {3, 6}};

  // solve A * x = B using exact matrix inverse
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  auto X1   = matrix_t{Ainv * B};
  EXPECT_ARRAY_NEAR(matrix_t{A * X1}, B);

  // solve A * x = B using getrf and getrs
  auto A_d = to_device(A);
  auto B_d = to_device(B);
  nda::cuarray<int, 1> ipiv(3);
  nda::lapack::getrf(A_d, ipiv);
  nda::lapack::getrs(A_d, B_d, ipiv);

  auto X2 = to_host(B_d);
  EXPECT_ARRAY_NEAR(matrix_t{A * X2}, B);
  EXPECT_ARRAY_NEAR(X1, X2);
}

TEST(NDA, CULAPACKGetrsAndGetrf) {
  test_getrs_getrf<double>();
  test_getrs_getrf<std::complex<double>>();
}
