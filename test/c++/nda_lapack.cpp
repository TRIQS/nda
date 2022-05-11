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

// ======================================= gtsv =====================================

template <typename value_t>
void test_gtsv() {

  vector<value_t> DL = {4, 3, 2, 1};    // sub-diagonal elements
  vector<value_t> D  = {1, 2, 3, 4, 5}; // diagonal elements
  vector<value_t> DU = {1, 2, 3, 4};    // super-diagonal elements

  vector<value_t> B1 = {6, 2, 7, 4, 5};  // RHS column 1
  vector<value_t> B2 = {1, 3, 8, 9, 10}; // RHS column 2
  auto B             = matrix<value_t, F_layout>(5, 2);
  B(_, 0)            = B1;
  B(_, 1)            = B2;

  // reference solutions
  vector<double> ref_sol_1 = {43.0 / 33.0, 155.0 / 33.0, -208.0 / 33.0, 130.0 / 33.0, 7.0 / 33.0};
  vector<double> ref_sol_2 = {-28.0 / 33.0, 61.0 / 33.0, 89.0 / 66.0, -35.0 / 66.0, 139.0 / 66.0};
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
TEST(lapack, gtsv) { test_gtsv<double>(); }    // NOLINT
TEST(lapack, zgtsv) { test_gtsv<dcomplex>(); } // NOLINT

//---------------------------------------------------------

TEST(lapack, cgtsv) { //NOLINT

  vector<dcomplex> DL = {-4i, -3i, -2i, -1i}; // sub-diagonal elements
  vector<dcomplex> D  = {1, 2, 3, 4, 5};      // diagonal elements
  vector<dcomplex> DU = {1i, 2i, 3i, 4i};     // super-diagonal elements

  vector<dcomplex> B1 = {6 + 0i, 2i, 7 + 0i, 4i, 5 + 0i}; // RHS column 1
  vector<dcomplex> B2 = {1i, 3 + 0i, 8i, 9 + 0i, 10i};    // RHS column 2
  matrix<dcomplex, F_layout> B(5, 2);
  B(_, 0) = B1;
  B(_, 1) = B2;

  // reference solutions
  vector<dcomplex> ref_sol_1 = {137.0 / 33.0 + 0i, -61i / 33.0, 368.0 / 33.0 + 0i, 230i / 33.0, -13.0 / 33.0 + 0i};
  vector<dcomplex> ref_sol_2 = {-35i / 33.0, 68.0 / 33.0 + 0i, -103i / 66.0, 415.0 / 66.0 + 0i, 215i / 66.0};
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

template <typename value_t>
void test_gesvd() { //NOLINT
  using matrix_t = matrix<value_t, F_layout>;

  auto A      = matrix_t{{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}}};
  auto [M, N] = A.shape();

  auto U  = matrix_t(M, M);
  auto VT = matrix_t(N, N);

  auto S     = vector<double>(std::min(M, N));
  auto Acopy = matrix_t{A};
  lapack::gesvd(Acopy, S, U, VT);

  auto Sigma = matrix_t::zeros(A.shape());
  for (auto i : range(std::min(M, N))) Sigma(i, i) = S(i);
  EXPECT_ARRAY_NEAR(A, U * Sigma * VT, 1e-14);
}
TEST(lapack, gesvd) { test_gesvd<double>(); }    //NOLINT
TEST(lapack, zgesvd) { test_gesvd<dcomplex>(); } //NOLINT

// =================================== gelss =======================================

template <typename value_t>
void test_gelss() {
  // Cf. http://www.netlib.org/lapack/explore-html/d3/d77/example___d_g_e_l_s__colmajor_8c_source.html
  auto A = matrix<value_t>{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}};
  auto B = matrix<value_t>{{-10, -3}, {12, 14}, {14, 12}, {16, 16}, {18, 16}};

  auto [M, N]  = A.shape();
  auto x_exact = matrix<value_t>{{2, 1}, {1, 1}, {1, 2}};
  auto S       = vector<double>(std::min(M, N));

  auto gelss_new    = lapack::gelss_worker<value_t>{A};
  auto [x_1, eps_1] = gelss_new(B);
  EXPECT_ARRAY_NEAR(x_exact, x_1, 1e-14);

  int rank{};
  matrix<value_t, F_layout> AF{A}, BF{B};
  lapack::gelss(AF, BF, S, 1e-18, rank);
  EXPECT_ARRAY_NEAR(x_exact, BF(range(N), _), 1e-14);
}
TEST(lapack, gelss) { test_gelss<double>(); }    //NOLINT
TEST(lapack, zgelss) { test_gelss<dcomplex>(); } //NOLINT

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
  auto Acopy = matrix_t{A};
  auto Bcopy = matrix_t{B};
  array<int, 1> ipiv(3);
  lapack::getrf(Acopy, ipiv);
  lapack::getrs(Acopy, Bcopy, ipiv);
  auto X2 = matrix_t{Bcopy};
  EXPECT_ARRAY_NEAR(matrix_t{A * X2}, B);
  EXPECT_ARRAY_NEAR(X1, X2);
}
TEST(lapack, getrs) { test_getrs<double>(); }    //NOLINT
TEST(lapack, zgetrs) { test_getrs<dcomplex>(); } //NOLINT
