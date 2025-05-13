// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/lapack/gelss_worker.hpp>
#include <nda/nda.hpp>

#include <algorithm>
#include <complex>
#include <type_traits>

using namespace nda;

// Test LAPACK gtsv function.
void test_gtsv(auto dl, auto d, auto du, auto B, auto exp) {
  int info = lapack::gtsv(dl, d, du, B);
  EXPECT_EQ(info, 0);
  EXPECT_ARRAY_NEAR(B, exp);
}

TEST(NDA, LAPACKGtsvDouble) {
  auto check = []<typename T>() {
    // sub-diagonal, diagonal, and super-diagonal elements
    auto dl = vector<T>{4, 3, 2, 1};
    auto d  = vector<T>{1, 2, 3, 4, 5};
    auto du = vector<T>{1, 2, 3, 4};

    // right hand sides
    auto b1          = vector<T>{6, 2, 7, 4, 5};
    auto b2          = vector<T>{1, 3, 8, 9, 10};
    auto B           = matrix<T, F_layout>(5, 2);
    B(range::all, 0) = b1;
    B(range::all, 1) = b2;

    // expected solutions
    auto exp_x1          = vector<double>{43.0 / 33.0, 155.0 / 33.0, -208.0 / 33.0, 130.0 / 33.0, 7.0 / 33.0};
    auto exp_x2          = vector<double>{-28.0 / 33.0, 61.0 / 33.0, 89.0 / 66.0, -35.0 / 66.0, 139.0 / 66.0};
    auto exp_X           = matrix<double, F_layout>(5, 2);
    exp_X(range::all, 0) = exp_x1;
    exp_X(range::all, 1) = exp_x2;

    test_gtsv(dl, d, du, b1, exp_x1);
    test_gtsv(dl, d, du, b2, exp_x2);
    test_gtsv(dl, d, du, B, exp_X);
  };

  check.operator()<double>();
  check.operator()<std::complex<double>>();
}

TEST(NDA, LAPACKGtsvComplex) {
  // sub-diagonal, diagonal, and super-diagonal elements
  auto dl = vector<std::complex<double>>{-4i, -3i, -2i, -1i};
  auto d  = vector<std::complex<double>>{1, 2, 3, 4, 5};
  auto du = vector<std::complex<double>>{1i, 2i, 3i, 4i};

  // right hand sides
  auto b1          = vector<std::complex<double>>{6 + 0i, 2i, 7 + 0i, 4i, 5 + 0i};
  auto b2          = vector<std::complex<double>>{1i, 3 + 0i, 8i, 9 + 0i, 10i};
  auto B           = matrix<std::complex<double>, F_layout>(5, 2);
  B(range::all, 0) = b1;
  B(range::all, 1) = b2;

  // expected solutions
  auto exp_x1          = vector<std::complex<double>>{137.0 / 33.0 + 0i, -61i / 33.0, 368.0 / 33.0 + 0i, 230i / 33.0, -13.0 / 33.0 + 0i};
  auto exp_x2          = vector<std::complex<double>>{-35i / 33.0, 68.0 / 33.0 + 0i, -103i / 66.0, 415.0 / 66.0 + 0i, 215i / 66.0};
  auto exp_X           = matrix<std::complex<double>, F_layout>(5, 2);
  exp_X(range::all, 0) = exp_x1;
  exp_X(range::all, 1) = exp_x2;

  test_gtsv(dl, d, du, b1, exp_x1);
  test_gtsv(dl, d, du, b2, exp_x2);
  test_gtsv(dl, d, du, B, exp_X);
}

// Test LAPACK gesvd function.
template <typename T, typename Layout>
void test_gesvd() {
  using matrix_t = matrix<T, Layout>;

  auto A      = matrix_t{{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}}};
  auto [m, n] = A.shape();

  auto U  = matrix_t(m, m);
  auto VT = matrix_t(n, n);

  auto S     = vector<double>(std::min(m, n));
  auto Acopy = matrix_t{A};
  lapack::gesvd(Acopy, S, U, VT);

  auto Sigma = matrix_t::zeros(A.shape());
  for (auto i : range(std::min(m, n))) Sigma(i, i) = S(i);
  EXPECT_ARRAY_NEAR(A, U * Sigma * VT, 1e-14);
}

TEST(NDA, LAPACKGesvd) {
  test_gesvd<double, C_layout>();
  test_gesvd<double, F_layout>();
  test_gesvd<std::complex<double>, C_layout>();
  test_gesvd<std::complex<double>, F_layout>();
}

// Test LAPACK geqp3, orgqr and ungqr functions.
template <typename T, bool wide_matrix = false>
void test_geqp3_orgqr_ungqr() {
  using matrix_t = matrix<T, F_layout>;

  auto A = matrix_t{{{1, 1, 1}, {3, 2, 4}, {5, 3, 2}, {2, 4, 5}, {4, 5, 3}}};
  if constexpr (wide_matrix) A = matrix_t{transpose(A)};
  auto [m, n] = A.shape();

  // compute QR factorization with column pivoting, i.e. A * P = Q * R
  auto jpvt = nda::zeros<int>(n);
  auto tau  = nda::vector<T>(std::min(m, n));
  auto Q    = matrix_t{A};
  lapack::geqp3(Q, jpvt, tau);

  // compute A * P by permuting columns of A
  jpvt -= 1;
  auto AP = matrix_t{A};
  for (int j = 0; j < n; ++j) { AP(range::all, j) = A(range::all, jpvt(j)); }

  // extract upper triangular matrix R
  auto R = nda::matrix<T, F_layout>::zeros(std::min(m, n), n);
  for (int i = 0; i < std::min(m, n); ++i) {
    for (int j = i; j < n; ++j) { R(i, j) = Q(i, j); }
  }

  // extract matrix Q with orthonormal columns
  if constexpr (std::is_same_v<T, double>) {
    lapack::orgqr(Q(range::all, range(std::min(m, n))), tau);
  } else {
    lapack::ungqr(Q(range::all, range(std::min(m, n))), tau);
  }

  EXPECT_ARRAY_NEAR(AP, Q(range::all, range(std::min(m, n))) * R, 1e-14);
}

TEST(NDA, LAPACKGeqp3UngqrAndOrgqr) {
  // tall matrix, i.e. n_rows > n_cols
  test_geqp3_orgqr_ungqr<double>();
  test_geqp3_orgqr_ungqr<std::complex<double>>();

  // wide matrix, i.e. n_rows < n_cols
  test_geqp3_orgqr_ungqr<double, true>();
  test_geqp3_orgqr_ungqr<std::complex<double>, true>();
}

// Test LAPACK gelss function and the gelss_worker class.
template <typename value_t>
void test_gelss() {
  // Cf. https://www.netlib.org/lapack/lapack-3.9.0/LAPACKE/example/example_DGELS_colmajor.c
  auto A = matrix<value_t>{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}};
  auto B = matrix<value_t>{{-10, -3}, {12, 14}, {14, 12}, {16, 16}, {18, 16}};
  auto b = vector<value_t>{-10, 12, 14, 16, 18};

  auto [m, n]  = A.shape();
  auto x_exact = matrix<value_t>{{2, 1}, {1, 1}, {1, 2}};
  auto s       = vector<double>(std::min(m, n));

  // using the gelss_worker class
  auto worker       = lapack::gelss_worker<value_t>{A};
  auto [x_1, eps_1] = worker(B);
  EXPECT_ARRAY_NEAR(x_exact, x_1, 1e-14);

  auto [x_2, eps_2] = worker(b);
  EXPECT_ARRAY_NEAR(x_exact(range::all, 0), x_2, 1e-14);

  // call the gelss function directly
  int rank{};
  matrix<value_t, F_layout> A_f{A}, B_f{B};
  lapack::gelss(A_f, B_f, s, 1e-18, rank);
  EXPECT_ARRAY_NEAR(x_exact, B_f(range(n), range::all), 1e-14);

  A_f = A;
  lapack::gelss(A_f, b, s, 1e-18, rank);
  EXPECT_ARRAY_NEAR(x_exact(range::all, 0), b(range(n)), 1e-14);
}

TEST(NDA, LAPACKGelss) {
  test_gelss<double>();
  test_gelss<std::complex<double>>();
}

// Test LAPACK getrs, getrf and getri functions.
template <typename value_t>
void test_getrs_getrf_getri() {
  using matrix_t = matrix<value_t, F_layout>;

  auto A = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto B = matrix_t{{1, 5}, {4, 5}, {3, 6}};

  // solve A * x = B using the exact matrix inverse
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  auto X1   = matrix_t{Ainv * B};
  EXPECT_ARRAY_NEAR(matrix_t{A * X1}, B);

  // solve A * x = B using getrf and getrs
  auto Acopy = matrix_t{A};
  auto Bcopy = matrix_t{B};
  array<int, 1> ipiv(3);
  lapack::getrf(Acopy, ipiv);
  lapack::getrs(Acopy, Bcopy, ipiv);
  auto X2 = matrix_t{Bcopy};
  EXPECT_ARRAY_NEAR(matrix_t{A * X2}, B);
  EXPECT_ARRAY_NEAR(X1, X2);

  // compute the inverse of A using getrf and getri
  auto Ainv2 = Acopy;
  lapack::getri(Ainv2, ipiv);
  EXPECT_ARRAY_NEAR(Ainv, Ainv2);
}

TEST(NDA, LAPAKCGetrsGetrfAndGetri) {
  test_getrs_getrf_getri<double>();
  test_getrs_getrf_getri<std::complex<double>>();
}
