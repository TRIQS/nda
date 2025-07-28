// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"
#include "nda/traits.hpp"

#include <limits>
#include <nda/gtest_tools.hpp>
#include <nda/lapack/gelss_worker.hpp>
#include <nda/nda.hpp>

#include <algorithm>
#include <complex>
#include <tuple>
#include <type_traits>

using namespace nda;
using namespace std::complex_literals;

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
  using fp_type  = nda::get_fp_t<T>;
  // condition number is ~7, and magnitude 5 with absolute error checking
  constexpr double eps_close = 7 * 5 * std::numeric_limits<fp_type>::epsilon();

  auto A      = matrix_t{{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}}};
  auto [m, n] = A.shape();

  auto U  = matrix_t(m, m);
  auto VT = matrix_t(n, n);

  auto S     = vector<fp_type>(std::min(m, n));
  auto Acopy = matrix_t{A};
  lapack::gesvd(Acopy, S, U, VT);

  auto Sigma = matrix_t::zeros(A.shape());
  for (auto i : range(std::min(m, n))) Sigma(i, i) = S(i);
  EXPECT_ARRAY_NEAR(A, U * Sigma * VT, eps_close);
}

TEST(NDA, LAPACKGesvd) {
  test_gesvd<float, C_layout>();
  test_gesvd<float, F_layout>();
  test_gesvd<std::complex<float>, C_layout>();
  test_gesvd<std::complex<float>, F_layout>();
  test_gesvd<double, C_layout>();
  test_gesvd<double, F_layout>();
  test_gesvd<std::complex<double>, C_layout>();
  test_gesvd<std::complex<double>, F_layout>();
}

// Test LAPACK geqp3, orgqr and ungqr functions.
template <typename T, bool wide_matrix = false>
void test_geqp3_orgqr_ungqr() {
  using matrix_t = matrix<T, F_layout>;
  using fp_type  = nda::get_fp_t<T>;
  // condition number is ~7, and magnitude 5 with absolute error checking
  constexpr double eps_close = 7 * 5 * std::numeric_limits<fp_type>::epsilon();

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
  if constexpr (std::is_same_v<T, double> or std::is_same_v<T, float>) {
    lapack::orgqr(Q, tau);
  } else {
    lapack::ungqr(Q, tau);
  }

  EXPECT_ARRAY_NEAR(AP, Q(range::all, range(std::min(m, n))) * R, eps_close);
}

TEST(NDA, LAPACKGeqp3UngqrAndOrgqr) {
  // tall matrix, i.e. n_rows > n_cols
  test_geqp3_orgqr_ungqr<float>();
  test_geqp3_orgqr_ungqr<std::complex<float>>();
  test_geqp3_orgqr_ungqr<double>();
  test_geqp3_orgqr_ungqr<std::complex<double>>();

  // wide matrix, i.e. n_rows < n_cols
  test_geqp3_orgqr_ungqr<float, true>();
  test_geqp3_orgqr_ungqr<std::complex<float>, true>();
  test_geqp3_orgqr_ungqr<double, true>();
  test_geqp3_orgqr_ungqr<std::complex<double>, true>();
}

// Test LAPACK gelss function and the gelss_worker class.
template <typename value_t>
void test_gelss() {
  using fp_type = nda::get_fp_t<value_t>;
  // condition number of B is ~8, and magnitude ~10 with absolute error checking
  constexpr double eps_close = 8 * 10 * std::numeric_limits<fp_type>::epsilon();

  // Cf. https://www.netlib.org/lapack/lapack-3.9.0/LAPACKE/example/example_DGELS_colmajor.c
  auto A = matrix<value_t>{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}};
  auto B = matrix<value_t>{{-10, -3}, {12, 14}, {14, 12}, {16, 16}, {18, 16}};
  auto b = vector<value_t>{-10, 12, 14, 16, 18};

  auto [m, n]  = A.shape();
  auto x_exact = matrix<value_t>{{2, 1}, {1, 1}, {1, 2}};
  auto s       = vector<fp_type>(std::min(m, n));

  // using the gelss_worker class
  auto worker       = lapack::gelss_worker<value_t>{A};
  auto [x_1, eps_1] = worker(B);
  EXPECT_ARRAY_NEAR(x_exact, x_1, eps_close);

  auto [x_2, eps_2] = worker(b);
  EXPECT_ARRAY_NEAR(x_exact(range::all, 0), x_2, eps_close);

  // call the gelss function directly
  int rank{};
  matrix<value_t, F_layout> A_f{A}, B_f{B};
  lapack::gelss(A_f, B_f, s, 1e-18, rank);
  EXPECT_ARRAY_NEAR(x_exact, B_f(range(n), range::all), eps_close);

  A_f = A;
  lapack::gelss(A_f, b, s, 1e-18, rank);
  EXPECT_ARRAY_NEAR(x_exact(range::all, 0), b(range(n)), eps_close);
}

TEST(NDA, LAPACKGelss) {
  test_gelss<float>();
  test_gelss<std::complex<float>>();
  test_gelss<double>();
  test_gelss<std::complex<double>>();
}

// Test LAPACK getrs, getrf and getri functions.
template <typename T, typename Layout>
void test_getrs_getrf_getri() {
  using matrix_t   = matrix<T, Layout>;
  using f_matrix_t = matrix<T, F_layout>;
  T fac            = 1.0;
  if constexpr (nda::is_complex_v<T>) fac = 1.0i;

  auto A = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  A *= fac;
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  Ainv /= fac;
  auto B = matrix_t{{1, 5}, {4, 5}, {3, 6}};

  // solve A * X = B using getrf and getrs
  auto Acopy = matrix_t{A};
  auto Bcopy = f_matrix_t{B};
  array<int, 1> ipiv(3);
  lapack::getrf(Acopy, ipiv);
  lapack::getrs(Acopy, Bcopy, ipiv);
  auto X = matrix_t{Bcopy};
  EXPECT_ARRAY_NEAR(matrix_t{A * X}, B);
  EXPECT_ARRAY_NEAR(matrix_t{Ainv * B}, X);

  // solve A^T * X = B using getrf and getrs
  Acopy = A;
  Bcopy = B;
  lapack::getrf(Acopy, ipiv);
  lapack::getrs(nda::transpose(Acopy), Bcopy, ipiv);
  X = matrix_t{Bcopy};
  EXPECT_ARRAY_NEAR(matrix_t{nda::transpose(A) * X}, B);
  EXPECT_ARRAY_NEAR(matrix_t{nda::transpose(Ainv) * B}, X);

  // solve A^H * X = B using getrf and getrs
  if constexpr (blas::has_F_layout<matrix_t>) {
    Acopy = A;
    Bcopy = B;
    lapack::getrf(Acopy, ipiv);
    lapack::getrs(nda::conj(nda::transpose(Acopy)), Bcopy, ipiv);
    X = matrix_t{Bcopy};
    EXPECT_ARRAY_NEAR(matrix_t{nda::conj(nda::transpose(A)) * X}, B);
    EXPECT_ARRAY_NEAR(matrix_t{nda::conj(nda::transpose(Ainv)) * B}, X);
  }

  // solve A * x = b using getrf and getrs
  Acopy  = A;
  auto b = vector<T>{B(range::all, 0)};
  lapack::getrf(Acopy, ipiv);
  lapack::getrs(Acopy, b, ipiv);
  EXPECT_ARRAY_NEAR(A * b, B(range::all, 0));
  EXPECT_ARRAY_NEAR((Ainv * B)(range::all, 0), b);

  // compute the inverse of A using getrf and getri
  auto Ainv2 = Acopy;
  lapack::getri(Ainv2, ipiv);
  EXPECT_ARRAY_NEAR(Ainv, Ainv2);
}

TEST(NDA, LAPACKGetrsGetrfAndGetri) {
  test_getrs_getrf_getri<double, C_layout>();
  test_getrs_getrf_getri<double, F_layout>();
  test_getrs_getrf_getri<std::complex<double>, C_layout>();
  test_getrs_getrf_getri<std::complex<double>, F_layout>();
}

TEST(NDA, LAPACKGetrfWithRectangularMatrix) {
  auto A    = matrix<double, F_layout>{{1, 5}, {4, 5}, {3, 6}};
  auto AT   = matrix<double, F_layout>(nda::transpose(A));
  auto A_c  = matrix<double, C_layout>{A};
  auto AT_c = matrix<double, C_layout>{AT};
  auto ipiv = array<int, 1>(2);

  // get the matrices P, L, U from getrf output
  auto get_plu = [](auto const &M, auto const &ipiv, int m, int n) {
    using layout_t   = std::conditional_t<blas::has_C_layout<decltype(M)>, C_layout, F_layout>;
    auto P           = matrix<double, layout_t>::zeros(m, m);
    auto L           = matrix<double, layout_t>::zeros(m, m);
    auto U           = matrix<double, layout_t>::zeros(m, n);
    nda::diagonal(P) = 1;
    nda::diagonal(L) = 1;
    for (int i = 0; i < ipiv.size(); ++i) deep_swap(P(i, nda::range::all), P(ipiv(i) - 1, nda::range::all));
    for (int i = 0; i < m; ++i) {
      L(i, nda::range(i))    = (blas::has_C_layout<decltype(M)> ? M(nda::range(i), i) : M(i, nda::range(i)));
      U(i, nda::range(i, n)) = (blas::has_C_layout<decltype(M)> ? M(nda::range(i, n), i) : M(i, nda::range(i, n)));
    }
    return std::make_tuple(P, L, U);
  };

  // LU decomposition for 3x2 Fortran layout matrix
  auto LU_f_32 = A;
  lapack::getrf(LU_f_32, ipiv);
  auto [P_f_32, L_f_32, U_f_32] = get_plu(LU_f_32, ipiv, 3, 2);
  EXPECT_ARRAY_NEAR(P_f_32 * A, L_f_32 * U_f_32);

  // LU decomposition for 2x3 Fortran layout matrix
  auto LU_f_23 = AT;
  lapack::getrf(LU_f_23, ipiv);
  auto [P_f_23, L_f_23, U_f_23] = get_plu(LU_f_23, ipiv, 2, 3);
  EXPECT_ARRAY_NEAR(P_f_23 * AT, L_f_23 * U_f_23);

  // LU decomposition for 3x2 C layout matrix
  auto LU_c_32 = A_c;
  lapack::getrf(LU_c_32, ipiv);
  auto [P_c_32, L_c_32, U_c_32] = get_plu(LU_c_32, ipiv, 2, 3);
  EXPECT_ARRAY_NEAR(P_c_32 * nda::transpose(A_c), L_c_32 * U_c_32);

  // LU decomposition for 2x3 C layout matrix
  auto LU_c_23 = AT_c;
  lapack::getrf(LU_c_23, ipiv);
  auto [P_c_23, L_c_23, U_c_23] = get_plu(LU_c_23, ipiv, 3, 2);
  EXPECT_ARRAY_NEAR(P_c_23 * nda::transpose(AT_c), L_c_23 * U_c_23);
}

// Check that the eigenvectors/values are correct.
void check_eigen(auto const &A, auto const &V, auto const &l, double eps_close) {
  for (auto i : nda::range(0, A.extent(0))) { EXPECT_ARRAY_NEAR(A * V(nda::range::all, i), l(i) * V(nda::range::all, i), eps_close); }
}

void check_eigen(auto const &A, auto const &B, auto const &V, auto const &l, int itype, double eps_close) {
  for (auto i : nda::range(0, A.extent(0))) {
    if (itype == 1) {
      EXPECT_ARRAY_NEAR(A * V(nda::range::all, i), l(i) * B * V(nda::range::all, i), eps_close);
    } else if (itype == 2) {
      EXPECT_ARRAY_NEAR(A * B * V(nda::range::all, i), l(i) * V(nda::range::all, i), eps_close);
    } else {
      EXPECT_ARRAY_NEAR(B * A * V(nda::range::all, i), l(i) * V(nda::range::all, i), eps_close);
    }
  }
}

// Create a symmetric or hermitian matrix with restricted eigenvalues.
template <typename T>
auto syhe_matrix(int n, double a = 1e-6, double b = 1.0) {
  using matrix_t = nda::matrix<T, nda::F_layout>;

  // orthogonal/unitary matrix Q
  auto jpvt = nda::zeros<int>(n);
  auto tau  = nda::vector<T>(n);
  auto Q    = nda::matrix<T, nda::F_layout>::rand(n, n);
  nda::lapack::geqp3(Q, jpvt, tau);
  if constexpr (nda::is_complex_v<T>) {
    nda::lapack::ungqr(Q, tau);
  } else {
    nda::lapack::orgqr(Q, tau);
  }

  // diagonal matrix containing the eigenvalues
  auto D = nda::eye<double>(n) * a + nda::diag(nda::vector<double>::rand(n)) * (b - a);

  // return Q * D * Q^H (hermitian/symmetric)
  return matrix_t{Q * D * nda::dagger(Q)};
}

// Test LAPACK syev and heev functions.
template <typename T>
void test_syev_heev(auto xxev) {
  using fp_type = nda::get_fp_t<T>;
  // 100*epsilon is heuristic, since we're not actually applying the eigenvectors, we don't
  // need the condition number
  constexpr double eps_close = 100 * std::numeric_limits<fp_type>::epsilon();

  for (auto i : nda::range(1, 6)) {
    auto A = syhe_matrix<T>(i, -1, 1);

    // compute eigenvalues and eigenvectors
    auto A1 = A;
    auto w1 = nda::vector<fp_type>(i);
    xxev(A1, w1);
    check_eigen(A, A1, w1, eps_close);

    // compute eigenvalues only
    auto A2 = A;
    auto w2 = nda::vector<fp_type>{};
    xxev(A2, w2, 'N');
    EXPECT_ARRAY_NEAR(w2, w1, eps_close);

    // compute eigenvalues and eigenvectors of the transpose
    auto A3 = nda::matrix<T, nda::C_layout>{A};
    auto w3 = nda::vector<fp_type>(i);
    xxev(nda::transpose(A3), w3);
    EXPECT_ARRAY_NEAR(w3, w1, eps_close);
    if constexpr (nda::is_complex_v<T>) {
      check_eigen(nda::transpose(A), nda::transpose(A3), w3, eps_close);
    } else {
      check_eigen(A, nda::transpose(A3), w3, eps_close);
      EXPECT_ARRAY_NEAR(nda::transpose(A3), A1, eps_close);
    }

    // compute eigenvalues and eigenvectors of a view
    if (i > 3) {
      auto A4 = A;
      auto w4 = nda::vector<fp_type>{};
      xxev(A4(nda::range(3), nda::range(3)), w4);
      check_eigen(A(nda::range(3), nda::range(3)), A4(nda::range(3), nda::range(3)), w4, eps_close);
    }
  }
}

TEST(NDA, LAPACKSyevAndHeev) {
  constexpr auto syev = [](auto &&...ts) { return lapack::syev(ts...); };
  constexpr auto heev = [](auto &&...ts) { return lapack::heev(ts...); };
  test_syev_heev<float>(syev);
  test_syev_heev<std::complex<float>>(heev);
  test_syev_heev<double>(syev);
  test_syev_heev<std::complex<double>>(heev);
}

// Test LAPACK sygv and hegv functions.
template <typename T>
void test_sygv_hegv(int itype, auto xxgv) {
  using fp_type = nda::get_fp_t<T>;
  // 100*epsilon is heuristic, since we're not actually applying the eigenvectors, we don't
  // need the condition number
  constexpr double eps_close = 100 * std::numeric_limits<fp_type>::epsilon();

  for (auto i : nda::range(1, 6)) {
    auto A = syhe_matrix<T>(i, -1, 1);
    auto B = syhe_matrix<T>(i, 1e-6, 1);

    // compute eigenvalues and eigenvectors
    auto A1 = A;
    auto B1 = B;
    auto w1 = nda::vector<fp_type>(i);
    xxgv(A1, B1, w1, 'V', itype);
    check_eigen(A, B, A1, w1, itype, eps_close);

    // compute eigenvalues only
    auto A2 = A;
    auto B2 = B;
    auto w2 = nda::vector<fp_type>{};
    xxgv(A2, B2, w2, 'N', itype);
    EXPECT_ARRAY_NEAR(w2, w1, eps_close);

    // compute eigenvalues and eigenvectors of a view
    if (i > 3) {
      auto A3 = A;
      auto B3 = B;
      auto w3 = nda::vector<fp_type>{};
      auto rg = nda::range(3);
      xxgv(A3(rg, rg), B3(rg, rg), w3, 'V', itype);
      check_eigen(A(rg, rg), B(rg, rg), A3(rg, rg), w3, itype, eps_close);
    }
  }
}

TEST(NDA, LAPACKSyegvAndHegv) {
  auto sygv = [](auto &&...ts) { return lapack::sygv(ts...); };
  auto hegv = [](auto &&...ts) { return lapack::hegv(ts...); };
  test_sygv_hegv<float>(1, sygv);
  test_sygv_hegv<float>(2, sygv);
  test_sygv_hegv<float>(3, sygv);
  test_sygv_hegv<std::complex<float>>(1, hegv);
  test_sygv_hegv<std::complex<float>>(2, hegv);
  test_sygv_hegv<std::complex<float>>(3, hegv);
  test_sygv_hegv<double>(1, sygv);
  test_sygv_hegv<double>(2, sygv);
  test_sygv_hegv<double>(3, sygv);
  test_sygv_hegv<std::complex<double>>(1, hegv);
  test_sygv_hegv<std::complex<double>>(2, hegv);
  test_sygv_hegv<std::complex<double>>(3, hegv);
}
