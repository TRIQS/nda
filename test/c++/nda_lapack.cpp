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
// Authors: Nils Wentzell

#include "test_common.hpp"
#include <nda/lapack.hpp>
#include <nda/lapack/gelss_worker.hpp>

using namespace nda;

// ======================================= gtsv =====================================

TEST(lapack, dgtsv) { //NOLINT

  array<double, 1> DL = {4, 3, 2, 1};    // sub-diagonal elements
  array<double, 1> D  = {1, 2, 3, 4, 5}; // diagonal elements
  array<double, 1> DU = {1, 2, 3, 4};    // super-diagonal elements

  array<double, 1> B1 = {6, 2, 7, 4, 5};  // RHS column 1
  array<double, 1> B2 = {1, 3, 8, 9, 10}; // RHS column 2
  matrix<double, F_layout> B(5, 2);
  B(_, 0) = B1;
  B(_, 1) = B2;

  // reference solutions
  array<double, 1> ref_sol_1 = {43.0 / 33.0, 155.0 / 33.0, -208.0 / 33.0, 130.0 / 33.0, 7.0 / 33.0};
  array<double, 1> ref_sol_2 = {-28.0 / 33.0, 61.0 / 33.0, 89.0 / 66.0, -35.0 / 66.0, 139.0 / 66.0};
  matrix<double, F_layout> ref_sol(5, 2);
  ref_sol(_, 0) = ref_sol_1;
  ref_sol(_, 1) = ref_sol_2;

  {
    auto dl(DL);
    auto d(D);
    auto du(DU);
    int info = lapack::gtsv(dl, d, du, B1);
    EXPECT_EQ(info, 0);
    EXPECT_ARRAY_NEAR(B1, ref_sol_1);
  }
  {
    auto dl(DL);
    auto d(D);
    auto du(DU);
    int info = lapack::gtsv(dl, d, du, B2);
    EXPECT_EQ(info, 0);
    EXPECT_ARRAY_NEAR(B2, ref_sol_2);
  }
  {
    auto dl(DL);
    auto d(D);
    auto du(DU);
    int info = lapack::gtsv(dl, d, du, B);
    EXPECT_EQ(info, 0);
    EXPECT_ARRAY_NEAR(B, ref_sol);
  }
}

//---------------------------------------------------------

TEST(lapack, cgtsv) { //NOLINT

  array<dcomplex, 1> DL = {-4i, -3i, -2i, -1i}; // sub-diagonal elements
  array<dcomplex, 1> D  = {1, 2, 3, 4, 5};      // diagonal elements
  array<dcomplex, 1> DU = {1i, 2i, 3i, 4i};     // super-diagonal elements

  array<dcomplex, 1> B1 = {6 + 0i, 2i, 7 + 0i, 4i, 5 + 0i}; // RHS column 1
  array<dcomplex, 1> B2 = {1i, 3 + 0i, 8i, 9 + 0i, 10i};    // RHS column 2
  matrix<dcomplex, F_layout> B(5, 2);
  B(_, 0) = B1;
  B(_, 1) = B2;

  // reference solutions
  array<dcomplex, 1> ref_sol_1 = {137.0 / 33.0 + 0i, -61i / 33.0, 368.0 / 33.0 + 0i, 230i / 33.0, -13.0 / 33.0 + 0i};
  array<dcomplex, 1> ref_sol_2 = {-35i / 33.0, 68.0 / 33.0 + 0i, -103i / 66.0, 415.0 / 66.0 + 0i, 215i / 66.0};
  matrix<dcomplex, F_layout> ref_sol(5, 2);
  ref_sol(_, 0) = ref_sol_1;
  ref_sol(_, 1) = ref_sol_2;

  {
    auto dl(DL);
    auto d(D);
    auto du(DU);
    int info = lapack::gtsv(dl, d, du, B1);
    EXPECT_EQ(info, 0);
    EXPECT_ARRAY_NEAR(B1, ref_sol_1);
  }
  {
    auto dl(DL);
    auto d(D);
    auto du(DU);
    int info = lapack::gtsv(dl, d, du, B2);
    EXPECT_EQ(info, 0);
    EXPECT_ARRAY_NEAR(B2, ref_sol_2);
  }
  {
    auto dl(DL);
    auto d(D);
    auto du(DU);
    int info = lapack::gtsv(dl, d, du, B);
    EXPECT_EQ(info, 0);
    EXPECT_ARRAY_NEAR(B, ref_sol);
  }
}

// ==================================== gesvd ============================================

TEST(lapack, gesvd) { //NOLINT

  auto A = matrix<dcomplex, F_layout>{{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}}};
  int M  = A.extent(0);
  int N  = A.extent(1);

  auto U  = matrix<dcomplex, F_layout>(M, M);
  auto VT = matrix<dcomplex, F_layout>(N, N);

  auto S = array<double, 1>(std::min(M, N));

  auto a_copy = A;
  lapack::gesvd(A, S, U, VT);

  auto S_Mat = A;
  S_Mat()    = 0.0;
  for (int i : range(std::min(M, N))) S_Mat(i, i) = S(i);

  EXPECT_ARRAY_NEAR(a_copy, U * S_Mat * VT, 1e-14);
}

// =================================== gelss =======================================

TEST(lapack, gelss) { //NOLINT

  static_assert(std::is_same_v<long, get_value_t<array_const_view<long, 2>>>, "Oops");
  static_assert(std::is_same_v<long, get_value_t<array_view<long const, 2>>>, "Oops");

  // Cf. http://www.netlib.org/lapack/explore-html/d3/d77/example___d_g_e_l_s__colmajor_8c_source.html
  auto A = matrix<dcomplex>{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}};
  auto B = matrix<dcomplex>{{-10, -3}, {12, 14}, {14, 12}, {16, 16}, {18, 16}};

  int M = A.extent(0);
  int N = A.extent(1);
  //int NRHS = B.extent(1);

  auto x_exact = matrix<dcomplex>{{2, 1}, {1, 1}, {1, 2}};
  auto S       = array<double, 1>(std::min(M, N));

  auto gelss_new    = lapack::gelss_worker<dcomplex>{A};
  auto [x_1, eps_1] = gelss_new(B);
  EXPECT_ARRAY_NEAR(x_exact, x_1, 1e-14);

  //int i;
  //lapack::gelss(A, B, S, 1e-18, i);
  //auto x_2 = B(range(N), range(NRHS));

  //EXPECT_ARRAY_NEAR(x_exact, x_2, 1e-14);
}

// =================================== getrs =======================================

TEST(lapack, getrs) { //NOLINT

  using matrix_t = matrix<double, F_layout>;

  auto A = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto B = matrix_t{{1, 5}, {4, 5}, {3, 6}};

  // Solve A * x = B using exact Matrix inverse
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  auto X1   = matrix_t{Ainv * B};
  EXPECT_ARRAY_NEAR(matrix_t{A * X1}, B);

  // Solve A * x = B using getrf,getrs
  auto Acopy = matrix_t{A};
  auto Bcopy = matrix_t{B};
  array<int, 1> ipiv(3);
  lapack::getrf(Acopy, ipiv);
  lapack::getrs(Acopy, Bcopy, ipiv);
  auto X2  = matrix_t{Bcopy};
  EXPECT_ARRAY_NEAR(matrix_t{A * X2}, B);

  EXPECT_ARRAY_NEAR(X1, X2);
}
