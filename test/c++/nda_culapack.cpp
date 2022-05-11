// Copyright (c) 2021 Simons Foundation
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

#include "test_common.hpp"
#include <nda/lapack.hpp>
#include <nda/lapack/gelss_worker.hpp>

using namespace nda;

// ==================================== gesvd ============================================

template <typename value_t>
void test_gesvd() { //NOLINT
  using matrix_t = matrix<value_t, F_layout>;

  auto A      = matrix_t{{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}}};
  auto [M, N] = A.shape();

  auto U  = matrix_t(M, M);
  auto VT = matrix_t(N, N);

  auto S = vector<double>(std::min(M, N));

  //using cumatrix_t = cumatrix<value_t, F_layout>;
  auto A_d         = to_device(A);
  auto S_d         = to_device(S);
  auto U_d         = to_device(U);
  auto VT_d        = to_device(VT);
  lapack::gesvd(A_d, S_d, U_d, VT_d);
  S  = S_d;
  U  = U_d;
  VT = VT_d;

  auto Sigma = matrix_t::zeros(A.shape());
  for (auto i : range(std::min(M, N))) Sigma(i, i) = S(i);
  EXPECT_ARRAY_NEAR(A, U * Sigma * VT, 1e-14);
}
TEST(culapack, gesvd) { test_gesvd<double>(); }    //NOLINT
TEST(culapack, zgesvd) { test_gesvd<dcomplex>(); } //NOLINT

// =================================== getrs =======================================

template <typename value_t>
void test_getrs() {
  using matrix_t = matrix<value_t, F_layout>;

  auto A = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto B = matrix_t{{1, 5}, {4, 5}, {3, 6}};

  // Solve A * x = B using exact Matrix inverse
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  auto X1   = matrix_t{Ainv * B};
  EXPECT_ARRAY_NEAR(matrix_t{A * X1}, B);

  // Solve A * x = B using getrf,getrs
  auto A_d = to_device(A);
  auto B_d = to_device(B);
  cuarray<int, 1> ipiv(3);
  lapack::getrf(A_d, ipiv);
  lapack::getrs(A_d, B_d, ipiv);

  auto X2 = to_host(B_d);
  EXPECT_ARRAY_NEAR(matrix_t{A * X2}, B);
  EXPECT_ARRAY_NEAR(X1, X2);
}
TEST(lapack, getrs) { test_getrs<double>(); }    //NOLINT
TEST(lapack, zgetrs) { test_getrs<dcomplex>(); } //NOLINT
