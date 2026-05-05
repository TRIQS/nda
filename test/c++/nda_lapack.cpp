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
#include <concepts>
#include <limits>
#include <numbers>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace std::complex_literals;
using nda::C_layout, nda::F_layout;

// Test LAPACK gtsv function.
template <typename T>
void test_gtsv() {
  auto du    = nda::vector<T>{4, 3, 2, 1};
  auto d     = nda::vector<T>{1, 2, 3, 4, 5};
  auto dl    = nda::vector<T>{1, 2, 3, 4};
  auto B     = nda::matrix<T, F_layout>{{9, 34}, {14, 44}, {21, 56}, {30, 70}, {41, 86}};
  auto exp_X = nda::matrix<T>{{1, 6}, {2, 7}, {3, 8}, {4, 9}, {5, 10}};
  if constexpr (nda::is_complex_v<T>) {
    dl *= T{1i};
    d *= T{1i};
    du *= T{1i};
    B *= T{-1};
    exp_X *= T{1i};
  }

  // verify the result
  auto verify_gtsv = [](auto dl, auto d, auto du, auto b, auto exp) {
    nda::lapack::gtsv(dl, d, du, b);
    EXPECT_ARRAY_NEAR(b, exp, fp_tol<T>);
  };

  // solve A * X = B
  verify_gtsv(dl, d, du, B, exp_X);

  // solve A * x = b
  verify_gtsv(dl, d, du, make_regular(B(nda::range::all, 0)), make_regular(exp_X(nda::range::all, 0)));
  verify_gtsv(dl, d, du, make_regular(B(nda::range::all, 1)), make_regular(exp_X(nda::range::all, 1)));
}

TEST(NDA, LAPACKGtsv) {
  test_gtsv<float>();
  test_gtsv<std::complex<float>>();
  test_gtsv<double>();
  test_gtsv<std::complex<double>>();
}

// Test LAPACK gesvd function.
template <typename T, typename Layout>
void test_gesvd() {
  using matrix_t = nda::matrix<T, Layout>;
  using fp_t     = nda::get_fp_t<T>;

  auto A      = matrix_t{{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}}};
  auto [m, n] = A.shape();

  // expected condition number and spectral norm of A from numpy
  constexpr fp_t cond_A = 6.784414066333698;
  constexpr fp_t norm_A = 12.316822252443167;

  // compute SVD
  auto U     = matrix_t(m, m);
  auto VH    = matrix_t(n, n);
  auto s     = nda::vector<fp_t>(std::min(m, n));
  auto Acopy = matrix_t{A};
  nda::lapack::gesvd(Acopy, s, U, VH);

  // construct diagonal singular value matrix
  auto Sigma = matrix_t::zeros(A.shape());
  for (auto i : nda::range(std::min(m, n))) Sigma(i, i) = s(i);

  // check condition number and spectral norm
  EXPECT_NEAR(s(0) / s(s.size() - 1), cond_A, fp_tol<T>);
  EXPECT_NEAR(s(0), norm_A, fp_tol<T>);

  // check backward error
  EXPECT_ARRAY_NEAR(A, U * Sigma * VH, fp_tol<T>);
}

template <typename T>
void test_gesvd_layouts() {
  test_gesvd<T, C_layout>();
  test_gesvd<T, F_layout>();
}

TEST(NDA, LAPACKGesvd) {
  test_gesvd_layouts<float>();
  test_gesvd_layouts<std::complex<float>>();
  test_gesvd_layouts<double>();
  test_gesvd_layouts<std::complex<double>>();
}

// Test LAPACK geqp3/geqrf, orgqr and ungqr functions.
template <typename T, bool wide_matrix = false>
void test_geqxx_orgqr_ungqr(bool with_pivoting = true) {
  using matrix_t = nda::matrix<T, F_layout>;

  auto A = matrix_t{{{1, 1, 1}, {3, 2, 4}, {5, 3, 2}, {2, 4, 5}, {4, 5, 3}}};
  if constexpr (wide_matrix) A = matrix_t{transpose(A)};
  auto [m, n] = A.shape();

  // compute QR factorization (with column pivoting), i.e. A * P = Q * R
  auto jpvt = nda::zeros<int>(n);
  auto tau  = nda::vector<T>(std::min(m, n));
  auto Q    = matrix_t{A};
  if (with_pivoting) {
    nda::lapack::geqp3(Q, jpvt, tau);
  } else {
    nda::lapack::geqrf(Q, tau);
    jpvt = nda::arange<int>(1, n + 1);
  }

  // compute A * P by permuting columns of A
  jpvt -= 1;
  auto AP = matrix_t{A};
  for (int j = 0; j < n; ++j) { AP(nda::range::all, j) = A(nda::range::all, jpvt(j)); }

  // extract upper triangular matrix R
  auto R = nda::matrix<T, F_layout>::zeros(std::min(m, n), n);
  for (int i = 0; i < std::min(m, n); ++i) {
    for (int j = i; j < n; ++j) { R(i, j) = Q(i, j); }
  }

  // extract matrix Q with orthonormal columns
  if constexpr (std::floating_point<T>) {
    nda::lapack::orgqr(Q(nda::range::all, nda::range(std::min(m, n))), tau);
  } else {
    nda::lapack::ungqr(Q(nda::range::all, nda::range(std::min(m, n))), tau);
  }

  EXPECT_ARRAY_NEAR(AP, Q(nda::range::all, nda::range(std::min(m, n))) * R, fp_tol<T>);
}

TEST(NDA, LAPACKGeqp3UngqrAndOrgqr) {
  // tall matrix, i.e. n_rows > n_cols
  test_geqxx_orgqr_ungqr<float>();
  test_geqxx_orgqr_ungqr<std::complex<float>>();
  test_geqxx_orgqr_ungqr<double>();
  test_geqxx_orgqr_ungqr<std::complex<double>>();

  // wide matrix, i.e. n_rows < n_cols
  test_geqxx_orgqr_ungqr<float, true>();
  test_geqxx_orgqr_ungqr<std::complex<float>, true>();
  test_geqxx_orgqr_ungqr<double, true>();
  test_geqxx_orgqr_ungqr<std::complex<double>, true>();
}

TEST(NDA, LAPACKGeqrfUngqrAndOrgqr) {
  // tall matrix, i.e. n_rows > n_cols
  test_geqxx_orgqr_ungqr<float>(false);
  test_geqxx_orgqr_ungqr<std::complex<float>>(false);
  test_geqxx_orgqr_ungqr<double>(false);
  test_geqxx_orgqr_ungqr<std::complex<double>>(false);

  // wide matrix, i.e. n_rows < n_cols
  test_geqxx_orgqr_ungqr<float, true>(false);
  test_geqxx_orgqr_ungqr<std::complex<float>, true>(false);
  test_geqxx_orgqr_ungqr<double, true>(false);
  test_geqxx_orgqr_ungqr<std::complex<double>, true>(false);
}

// Test LAPACK geqrf_batch, orgqr/ungqr (per-slice) and the batched gqr dispatcher.
template <typename T, bool wide_matrix = false>
void test_geqrf_orgqr_ungqr_batch() {
  using matrix_t = nda::matrix<T, F_layout>;

  auto A = matrix_t{{{1, 1, 1}, {3, 2, 4}, {5, 3, 2}, {2, 4, 5}, {4, 5, 3}}};
  if constexpr (wide_matrix) A = matrix_t{transpose(A)};
  auto [m, n] = A.shape();

  // create batched arrays by stacking copies of A
  constexpr int batch_size = 3;
  auto A_batch             = nda::array<T, 3, F_layout>(m, n, batch_size);
  for (int i = 0; i < batch_size; ++i) A_batch(nda::range::all, nda::range::all, i) = A;

  // compute batched QR factorization
  auto tau = nda::matrix<T, F_layout>(std::min(m, n), batch_size);
  nda::lapack::geqrf_batch(A_batch, tau);

  // capture R from each slice before any Q reconstruction touches A_batch
  auto R = nda::array<T, 3, F_layout>(std::min(m, n), n, batch_size);
  R()    = T{0};
  for (int i = 0; i < batch_size; ++i) {
    for (int k = 0; k < std::min(m, n); ++k) {
      for (int l = k; l < n; ++l) R(k, l, i) = A_batch(k, l, i);
    }
  }

  // verify Q reconstruction via per-slice orgqr/ungqr (creates copies, leaves A_batch intact)
  for (int i = 0; i < batch_size; ++i) {
    auto Q_i   = matrix_t{A_batch(nda::range::all, nda::range::all, i)};
    auto tau_i = nda::vector<T>{tau(nda::range::all, i)};
    if constexpr (std::floating_point<T>) {
      nda::lapack::orgqr(Q_i(nda::range::all, nda::range(std::min(m, n))), tau_i);
    } else {
      nda::lapack::ungqr(Q_i(nda::range::all, nda::range(std::min(m, n))), tau_i);
    }
    auto R_i = matrix_t{R(nda::range::all, nda::range::all, i)};
    EXPECT_ARRAY_NEAR(A, Q_i(nda::range::all, nda::range(std::min(m, n))) * R_i, fp_tol<T>);
  }

  // verify Q reconstruction via the batched gqr() dispatcher (overwrites A_batch)
  int info = nda::lapack::gqr(A_batch, tau);
  EXPECT_EQ(info, 0);
  for (int i = 0; i < batch_size; ++i) {
    auto Q_i = matrix_t{A_batch(nda::range::all, nda::range(std::min(m, n)), i)};
    auto R_i = matrix_t{R(nda::range::all, nda::range::all, i)};
    EXPECT_ARRAY_NEAR(A, Q_i * R_i, fp_tol<T>);
  }
}

TEST(NDA, LAPACKGeqrfOrgqrUngqrBatch) {
  // tall matrix, i.e. n_rows > n_cols
  test_geqrf_orgqr_ungqr_batch<float>();
  test_geqrf_orgqr_ungqr_batch<std::complex<float>>();
  test_geqrf_orgqr_ungqr_batch<double>();
  test_geqrf_orgqr_ungqr_batch<std::complex<double>>();

  // wide matrix, i.e. n_rows < n_cols
  test_geqrf_orgqr_ungqr_batch<float, true>();
  test_geqrf_orgqr_ungqr_batch<std::complex<float>, true>();
  test_geqrf_orgqr_ungqr_batch<double, true>();
  test_geqrf_orgqr_ungqr_batch<std::complex<double>, true>();
}

// Test LAPACK gelss function and the gelss_worker class.
template <typename T>
void test_gelss() {
  using fp_t = nda::get_fp_t<T>;

  // Cf. https://www.netlib.org/lapack/lapack-3.9.0/LAPACKE/example/example_DGELS_colmajor.c
  auto A = nda::matrix<T>{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}};
  auto B = nda::matrix<T>{{-10, -3}, {12, 14}, {14, 12}, {16, 16}, {18, 16}};
  auto b = nda::vector<T>{-10, 12, 14, 16, 18};

  auto [m, n]  = A.shape();
  auto X_exact = nda::matrix<T>{{2, 1}, {1, 1}, {1, 2}};

  // using the gelss_worker class for matrix RHS
  auto worker       = nda::lapack::gelss_worker<T>{A};
  auto [x_1, eps_1] = worker(B);
  EXPECT_ARRAY_NEAR(X_exact, x_1, fp_tol<T>);

  // using the gelss_worker class for vector RHS
  auto [x_2, eps_2] = worker(b);
  EXPECT_ARRAY_NEAR(X_exact(nda::range::all, 0), x_2, fp_tol<T>);

  // call the gelss function directly for matrix RHS
  int rank{};
  nda::matrix<T, F_layout> A_f{A}, B_f{B};
  auto s = nda::vector<fp_t>(std::min(m, n));
  nda::lapack::gelss(A_f, B_f, s, 1e-18, rank);
  EXPECT_ARRAY_NEAR(X_exact, B_f(nda::range(n), nda::range::all), fp_tol<T>);

  // call the gelss function directly for vector RHS
  A_f = A;
  nda::lapack::gelss(A_f, b, s, 1e-18, rank);
  EXPECT_ARRAY_NEAR(X_exact(nda::range::all, 0), b(nda::range(n)), fp_tol<T>);
}

TEST(NDA, LAPACKGelss) {
  test_gelss<float>();
  test_gelss<std::complex<float>>();
  test_gelss<double>();
  test_gelss<std::complex<double>>();
}

// Test LAPACK gelss function for underdetermined systems (m < n).
template <typename T>
void test_gelss_underdetermined() {
  using fp_t = nda::get_fp_t<T>;

  // underdetermined system: A is 2x3
  // A = [[1, 0, 1], [0, 1, 1]], b = [1, 1]
  // minimum norm solution: X = [[1/3, 1], [1/3, 0], [2/3, 1]]
  auto A = nda::matrix<T>{{1, 0, 1}, {0, 1, 1}};
  auto B = nda::matrix<T>{{1, 2}, {1, 1}};

  auto [m, n]  = A.shape();
  auto X_exact = nda::matrix<T>{{1.0 / 3.0, 1}, {1.0 / 3.0, 0}, {2.0 / 3.0, 1}};

  // call the gelss function for matrix RHS
  int rank{};
  nda::matrix<T, F_layout> A_f{A}, B_f(std::max(m, n), 2);
  B_f(nda::range(m), nda::range::all) = B;
  auto s                              = nda::vector<fp_t>(std::min(m, n));
  nda::lapack::gelss(A_f, B_f, s, 1e-18, rank);
  EXPECT_ARRAY_NEAR(X_exact, B_f, fp_tol<T>);

  // call the gelss function for vector RHS
  A_f              = A;
  auto b           = nda::vector<T>(std::max(m, n));
  b(nda::range(m)) = B(nda::range::all, 0);
  nda::lapack::gelss(A_f, b, s, 1e-18, rank);
  EXPECT_ARRAY_NEAR(X_exact(nda::range::all, 0), b, fp_tol<T>);
}

TEST(NDA, LAPACKGelssUnderdetermined) {
  test_gelss_underdetermined<float>();
  test_gelss_underdetermined<std::complex<float>>();
  test_gelss_underdetermined<double>();
  test_gelss_underdetermined<std::complex<double>>();
}

// Test LAPACK getrs, getrf and getri functions.
template <typename T, typename Layout>
void test_getrs_getrf_getri() {
  using matrix_t   = nda::matrix<T, Layout>;
  using f_matrix_t = nda::matrix<T, F_layout>;
  using fp_t       = nda::get_fp_t<T>;

  auto A    = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  if constexpr (nda::is_complex_v<T>) {
    A *= T{1i};
    Ainv /= T{1i};
  }
  auto B = matrix_t{{1, 5}, {4, 5}, {3, 6}};

  // tolerance based on condition number: cond(A) ~ 332, ||Ainv||_max = 24
  // error ~ cond(A) * ||Ainv||_max * eps => use eps * 10000 as tolerance
  constexpr auto tol = std::numeric_limits<fp_t>::epsilon() * 10000;

  // solve A * X = B using getrf and getrs
  auto Acopy = matrix_t{A};
  auto Bcopy = f_matrix_t{B};
  nda::array<int, 1> ipiv(3);
  nda::lapack::getrf(Acopy, ipiv);
  nda::lapack::getrs(Acopy, Bcopy, ipiv);
  auto X = matrix_t{Bcopy};
  EXPECT_ARRAY_NEAR(A * X, B, tol);
  EXPECT_ARRAY_NEAR(Ainv * B, X, tol);

  // solve A^T * X = B using getrf and getrs
  Acopy = A;
  Bcopy = B;
  nda::lapack::getrf(Acopy, ipiv);
  nda::lapack::getrs(nda::transpose(Acopy), Bcopy, ipiv);
  X = matrix_t{Bcopy};
  EXPECT_ARRAY_NEAR(nda::transpose(A) * X, B, tol);
  EXPECT_ARRAY_NEAR(nda::transpose(Ainv) * B, X, tol);

  // solve A^H * X = B using getrf and getrs
  if constexpr (nda::blas_lapack::has_F_layout<matrix_t>) {
    Acopy = A;
    Bcopy = B;
    nda::lapack::getrf(Acopy, ipiv);
    nda::lapack::getrs(nda::dagger(Acopy), Bcopy, ipiv);
    X = matrix_t{Bcopy};
    EXPECT_ARRAY_NEAR(nda::dagger(A) * X, B, tol);
    EXPECT_ARRAY_NEAR(nda::dagger(Ainv) * B, X, tol);
  }

  // solve A * x = b using getrf and getrs
  Acopy  = A;
  auto b = nda::vector<T>{B(nda::range::all, 0)};
  nda::lapack::getrf(Acopy, ipiv);
  nda::lapack::getrs(Acopy, b, ipiv);
  EXPECT_ARRAY_NEAR(A * b, B(nda::range::all, 0), tol);
  EXPECT_ARRAY_NEAR((Ainv * B)(nda::range::all, 0), b, tol);

  // compute the inverse of A using getrf and getri
  auto Ainv2 = Acopy;
  nda::lapack::getri(Ainv2, ipiv);
  EXPECT_ARRAY_NEAR(Ainv, Ainv2, tol);
}

template <typename T>
void test_getrs_getrf_getri_layouts() {
  test_getrs_getrf_getri<T, C_layout>();
  test_getrs_getrf_getri<T, F_layout>();
}

TEST(NDA, LAPACKGetrsGetrfAndGetri) {
  test_getrs_getrf_getri_layouts<float>();
  test_getrs_getrf_getri_layouts<std::complex<float>>();
  test_getrs_getrf_getri_layouts<double>();
  test_getrs_getrf_getri_layouts<std::complex<double>>();
}

template <typename T>
void test_rectangular_getrf() {
  using namespace nda::blas_lapack;
  auto A    = nda::matrix<T, F_layout>{{1, 5}, {4, 5}, {3, 6}};
  auto AT   = nda::matrix<T, F_layout>(nda::transpose(A));
  auto A_c  = nda::matrix<T, C_layout>{A};
  auto AT_c = nda::matrix<T, C_layout>{AT};
  auto ipiv = nda::array<int, 1>(2);

  // get the matrices P, L, U from getrf output
  auto get_plu = [](auto const &M, auto const &ip, int m, int n) {
    using layout_t   = std::conditional_t<has_C_layout<decltype(M)>, C_layout, F_layout>;
    auto P           = nda::matrix<T, layout_t>::zeros(m, m);
    auto L           = nda::matrix<T, layout_t>::zeros(m, m);
    auto U           = nda::matrix<T, layout_t>::zeros(m, n);
    nda::diagonal(P) = 1;
    nda::diagonal(L) = 1;
    for (int i = 0; i < ip.size(); ++i) deep_swap(P(i, nda::range::all), P(ip(i) - 1, nda::range::all));
    for (int i = 0; i < m; ++i) {
      L(i, nda::range(i))    = (has_C_layout<decltype(M)> ? M(nda::range(i), i) : M(i, nda::range(i)));
      U(i, nda::range(i, n)) = (has_C_layout<decltype(M)> ? M(nda::range(i, n), i) : M(i, nda::range(i, n)));
    }
    return std::make_tuple(P, L, U);
  };

  // LU decomposition for 3x2 Fortran layout matrix
  auto LU_f_32 = A;
  nda::lapack::getrf(LU_f_32, ipiv);
  auto [P_f_32, L_f_32, U_f_32] = get_plu(LU_f_32, ipiv, 3, 2);
  EXPECT_ARRAY_NEAR(P_f_32 * A, L_f_32 * U_f_32, fp_tol<T>);

  // LU decomposition for 2x3 Fortran layout matrix
  auto LU_f_23 = AT;
  nda::lapack::getrf(LU_f_23, ipiv);
  auto [P_f_23, L_f_23, U_f_23] = get_plu(LU_f_23, ipiv, 2, 3);
  EXPECT_ARRAY_NEAR(P_f_23 * AT, L_f_23 * U_f_23, fp_tol<T>);

  // LU decomposition for 3x2 C layout matrix
  auto LU_c_32 = A_c;
  nda::lapack::getrf(LU_c_32, ipiv);
  auto [P_c_32, L_c_32, U_c_32] = get_plu(LU_c_32, ipiv, 2, 3);
  EXPECT_ARRAY_NEAR(P_c_32 * nda::transpose(A_c), L_c_32 * U_c_32, fp_tol<T>);

  // LU decomposition for 2x3 C layout matrix
  auto LU_c_23 = AT_c;
  nda::lapack::getrf(LU_c_23, ipiv);
  auto [P_c_23, L_c_23, U_c_23] = get_plu(LU_c_23, ipiv, 3, 2);
  EXPECT_ARRAY_NEAR(P_c_23 * nda::transpose(AT_c), L_c_23 * U_c_23, fp_tol<T>);
}

TEST(NDA, LAPACKGetrfWithRectangularMatrix) {
  test_rectangular_getrf<float>();
  test_rectangular_getrf<std::complex<float>>();
  test_rectangular_getrf<double>();
  test_rectangular_getrf<std::complex<double>>();
}

// Test LAPACK getrf_batch, getrs_batch and getri_batch functions.
template <typename T, typename Layout>
void test_getrs_getrf_getri_batch() {
  using matrix_t = nda::matrix<T, F_layout>;
  using fp_t     = nda::get_fp_t<T>;

  auto A    = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  if constexpr (nda::is_complex_v<T>) {
    A *= T{1i};
    Ainv /= T{1i};
  }
  auto B = matrix_t{{1, 5}, {4, 5}, {3, 6}};

  // tolerance based on condition number: cond(A) ~ 332, ||Ainv||_max = 24
  // error ~ cond(A) * ||Ainv||_max * eps => use eps * 10000 as tolerance
  constexpr auto tol = std::numeric_limits<fp_t>::epsilon() * 10000;

  // helper to get the i-th matrix view depending on layout
  auto get_mat = [](auto &arr, int i) {
    if constexpr (nda::blas_lapack::has_F_layout<decltype(arr)>) {
      return nda::matrix_view<T, F_layout>(arr(nda::range::all, nda::range::all, i));
    } else {
      return nda::matrix_view<T, C_layout>(arr(i, nda::range::all, nda::range::all));
    }
  };

  // create batched arrays by stacking copies of the matrices
  constexpr int batch_size = 3;
  auto const [m, n]        = A.shape();
  auto const [k, nrhs]     = B.shape();
  auto a_shape             = std::array<long, 3>{m, n, batch_size};
  auto ipiv_shape          = std::array<long, 2>{m, batch_size};
  if constexpr (std::is_same_v<Layout, C_layout>) {
    a_shape    = {batch_size, m, n};
    ipiv_shape = {batch_size, m};
  }
  auto A_batch = nda::array<T, 3, Layout>(a_shape);
  auto B_batch = nda::array<T, 3, F_layout>(m, nrhs, batch_size);
  for (int i = 0; i < batch_size; ++i) {
    get_mat(A_batch, i) = A;
    get_mat(B_batch, i) = B;
  }

  // solve A * X = B using getrf_batch and getrs_batch
  auto Acopy = A_batch;
  auto Bcopy = B_batch;
  auto ipiv  = nda::matrix<int, Layout>(ipiv_shape);
  nda::lapack::getrf_batch(Acopy, ipiv);
  nda::lapack::getrs_batch(Acopy, Bcopy, ipiv);
  for (int i = 0; i < batch_size; ++i) {
    auto X = get_mat(Bcopy, i);
    EXPECT_ARRAY_NEAR(A * X, B, tol);
    EXPECT_ARRAY_NEAR(Ainv * B, X, tol);
  }

  // solve A^T * X = B using getrf_batch and getrs_batch
  Acopy = A_batch;
  Bcopy = B_batch;
  nda::lapack::getrf_batch(Acopy, ipiv);
  nda::lapack::getrs_batch(nda::transpose(Acopy), Bcopy, ipiv);
  for (int i = 0; i < batch_size; ++i) {
    auto X = get_mat(Bcopy, i);
    EXPECT_ARRAY_NEAR(nda::transpose(A) * X, B, tol);
    EXPECT_ARRAY_NEAR(nda::transpose(Ainv) * B, X, tol);
  }

  // solve A^H * X = B using getrf_batch and getrs_batch (F-layout only)
  if constexpr (std::is_same_v<Layout, F_layout>) {
    Acopy = A_batch;
    Bcopy = B_batch;
    nda::lapack::getrf_batch(Acopy, ipiv);
    nda::lapack::getrs_batch(nda::conj(nda::transpose(Acopy)), Bcopy, ipiv);
    for (int i = 0; i < batch_size; ++i) {
      auto X = get_mat(Bcopy, i);
      EXPECT_ARRAY_NEAR(nda::conj(nda::transpose(A)) * X, B, tol);
      EXPECT_ARRAY_NEAR(nda::conj(nda::transpose(Ainv)) * B, X, tol);
    }
  }

  // compute the inverse of A using getrf_batch and getri_batch
  Acopy = A_batch;
  nda::lapack::getrf_batch(Acopy, ipiv);
  nda::lapack::getri_batch(Acopy, ipiv);
  for (int i = 0; i < batch_size; ++i) EXPECT_ARRAY_NEAR(Ainv, get_mat(Acopy, i), tol);
}

template <typename T>
void test_getrs_getrf_getri_batch_layouts() {
  test_getrs_getrf_getri_batch<T, C_layout>();
  test_getrs_getrf_getri_batch<T, F_layout>();
}

TEST(NDA, LAPACKGetrfGetrsGetriBatch) {
  test_getrs_getrf_getri_batch_layouts<float>();
  test_getrs_getrf_getri_batch_layouts<std::complex<float>>();
  test_getrs_getrf_getri_batch_layouts<double>();
  test_getrs_getrf_getri_batch_layouts<std::complex<double>>();
}

// Check that the eigenvectors/values are correct.
void check_eigen(auto const &A, auto const &V, auto const &l, bool is_left = false) {
  using fp_t = nda::get_fp_t<nda::get_value_t<decltype(A)>>;
  if (not is_left) {
    EXPECT_ARRAY_NEAR(A * V, V * nda::diag(l), fp_tol<fp_t>);
  } else {
    EXPECT_ARRAY_NEAR(nda::dagger(V) * A, nda::diag(l) * nda::dagger(V), fp_tol<fp_t>);
  }
}

void check_eigen(auto const &A, auto const &B, auto const &V, auto const &l, int itype = 1) {
  using fp_t = nda::get_fp_t<nda::get_value_t<decltype(A)>>;
  if (itype == 1) {
    EXPECT_ARRAY_NEAR(A * V, B * V * nda::diag(l), fp_tol<fp_t>);
  } else if (itype == 2) {
    EXPECT_ARRAY_NEAR(A * B * V, V * nda::diag(l), fp_tol<fp_t>);
  } else {
    EXPECT_ARRAY_NEAR(B * A * V, V * nda::diag(l), fp_tol<fp_t>);
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
  using fp_t = nda::get_fp_t<T>;
  for (auto i : nda::range(1, 6)) {
    auto A = syhe_matrix<T>(i, -1, 1);

    // compute eigenvalues and eigenvectors
    auto A1 = A;
    auto w1 = nda::vector<fp_t>(i);
    xxev(A1, w1);
    check_eigen(A, A1, w1);

    // compute eigenvalues only
    auto A2 = A;
    auto w2 = nda::vector<fp_t>{};
    xxev(A2, w2, 'N');
    EXPECT_ARRAY_NEAR(w2, w1, fp_tol<T>);

    // compute eigenvalues and eigenvectors of the transpose
    auto A3 = nda::matrix<T, nda::C_layout>{A};
    auto w3 = nda::vector<fp_t>(i);
    xxev(nda::transpose(A3), w3);
    EXPECT_ARRAY_NEAR(w3, w1, fp_tol<T>);
    if constexpr (nda::is_complex_v<T>) {
      check_eigen(nda::transpose(A), nda::transpose(A3), w3);
    } else {
      check_eigen(A, nda::transpose(A3), w3);
      EXPECT_ARRAY_NEAR(nda::transpose(A3), A1, fp_tol<T>);
    }

    // compute eigenvalues and eigenvectors of a view
    if (i > 3) {
      auto A4 = A;
      auto w4 = nda::vector<fp_t>{};
      auto rg = nda::range(3);
      xxev(A4(rg, rg), w4);
      check_eigen(A(rg, rg), A4(rg, rg), w4);
    }
  }
}

TEST(NDA, LAPACKSyevAndHeev) {
  auto syev = [](auto &&...ts) { return nda::lapack::syev(ts...); };
  auto heev = [](auto &&...ts) { return nda::lapack::heev(ts...); };
  test_syev_heev<float>(syev);
  test_syev_heev<std::complex<float>>(heev);
  test_syev_heev<double>(syev);
  test_syev_heev<std::complex<double>>(heev);
}

// Test LAPACK sygv and hegv functions.
template <typename T>
void test_sygv_hegv(int itype, auto xxgv) {
  using fp_t = nda::get_fp_t<T>;
  for (auto i : nda::range(1, 6)) {
    auto A = syhe_matrix<T>(i, -1, 1);
    auto B = syhe_matrix<T>(i, 1e-6, 1);

    // compute eigenvalues and eigenvectors
    auto A1 = A;
    auto B1 = B;
    auto w1 = nda::vector<fp_t>(i);
    xxgv(A1, B1, w1, 'V', itype);
    check_eigen(A, B, A1, w1, itype);

    // compute eigenvalues only
    auto A2 = A;
    auto B2 = B;
    auto w2 = nda::vector<fp_t>{};
    xxgv(A2, B2, w2, 'N', itype);
    EXPECT_ARRAY_NEAR(w2, w1, fp_tol<T>);

    // compute eigenvalues and eigenvectors of a view
    if (i > 3) {
      auto A3 = A;
      auto B3 = B;
      auto w3 = nda::vector<fp_t>{};
      auto rg = nda::range(3);
      xxgv(A3(rg, rg), B3(rg, rg), w3, 'V', itype);
      check_eigen(A(rg, rg), B(rg, rg), A3(rg, rg), w3, itype);
    }
  }
}

TEST(NDA, LAPACKSygvAndHegv) {
  auto sygv = [](auto &&...ts) { return nda::lapack::sygv(ts...); };
  auto hegv = [](auto &&...ts) { return nda::lapack::hegv(ts...); };
  for (int itype = 1; itype <= 3; ++itype) {
    test_sygv_hegv<float>(itype, sygv);
    test_sygv_hegv<double>(itype, sygv);
    test_sygv_hegv<std::complex<float>>(itype, hegv);
    test_sygv_hegv<std::complex<double>>(itype, hegv);
  }
}

// Test LAPACK geev function for real matrices.
template <typename T>
void test_geev_real() {
  for (auto n : nda::range(1, 6)) {
    auto A = nda::matrix<T, nda::F_layout>::rand(n, n);

    // compute eigenvalues and eigenvectors
    nda::matrix<T, nda::F_layout> A1{A}, vl(n, n), vr(n, n);
    nda::vector<T> wr(n), wi(n);
    int info = nda::lapack::geev(A1, wr, wi, vl, vr, 'V', 'V');
    EXPECT_EQ(info, 0);

    // convert to complex eigenvalues and eigenvectors and check eigenvector equation
    auto lambda = nda::linalg::get_geev_eigenvalues(wr, wi);
    auto V      = nda::linalg::unpack_eigenvectors(wi, vr);
    auto U      = nda::linalg::unpack_eigenvectors(wi, vl);
    auto Acpx   = nda::matrix<std::complex<T>, nda::F_layout>{A};
    check_eigen(Acpx, V, lambda);
    check_eigen(Acpx, U, lambda, true);

    // compute eigenvalues only
    nda::matrix<T, nda::F_layout> A2{A}, vl2{}, vr2{};
    nda::vector<T> wr2(n), wi2(n);
    info = nda::lapack::geev(A2, wr2, wi2, vl2, vr2, 'N', 'N');
    EXPECT_EQ(info, 0);
    EXPECT_ARRAY_NEAR(wr2, wr, fp_tol<T>);
    EXPECT_ARRAY_NEAR(wi2, wi, fp_tol<T>);

    // compute eigenvalues and eigenvectors of a view
    if (n > 3) {
      nda::matrix<T, nda::F_layout> A3{A}, vl3(3, 3), vr3(3, 3);
      nda::vector<T> wr3(3), wi3(3);
      auto rg = nda::range(3);
      info    = nda::lapack::geev(A3(rg, rg), wr3, wi3, vl3, vr3, 'N', 'V');
      EXPECT_EQ(info, 0);

      // convert to complex eigenvalues and eigenvectors and check eigenvector equation
      auto lambda3 = nda::linalg::get_geev_eigenvalues(wr3, wi3);
      auto V3      = nda::linalg::unpack_eigenvectors(wi3, vr3);
      auto Asub    = nda::matrix<std::complex<T>, nda::F_layout>{A(rg, rg)};
      check_eigen(Asub, V3, lambda3);
    }
  }
}

// Test LAPACK geev function for complex matrices.
template <typename T>
void test_geev_complex() {
  for (auto n : nda::range(1, 6)) {
    auto A = nda::matrix<T, nda::F_layout>::rand(n, n);

    // compute eigenvalues and eigenvectors
    nda::matrix<T, nda::F_layout> A1{A}, U(n, n), V(n, n);
    nda::vector<T> lambda(n);
    int info = nda::lapack::geev(A1, lambda, U, V, 'V', 'V');
    EXPECT_EQ(info, 0);

    // check eigenvector equation
    check_eigen(A, V, lambda);
    check_eigen(A, U, lambda, true);

    // compute eigenvalues only
    nda::matrix<T, nda::F_layout> A2{A}, vl2{}, vr2{};
    nda::vector<T> w2(n);
    info = nda::lapack::geev(A2, w2, vl2, vr2, 'N', 'N');
    EXPECT_EQ(info, 0);
    EXPECT_ARRAY_NEAR(w2, lambda, fp_tol<T>);

    // compute eigenvalues and eigenvectors of a view
    if (n > 3) {
      nda::matrix<T, nda::F_layout> A3{A}, U3(3, 3), V3(3, 3);
      nda::vector<T> lambda3(3);
      auto rg = nda::range(3);
      info    = nda::lapack::geev(A3(rg, rg), lambda3, U3, V3, 'N', 'V');
      EXPECT_EQ(info, 0);
      check_eigen(A(rg, rg), V3, lambda3);
    }
  }
}

TEST(NDA, LAPACKGeev) {
  test_geev_real<float>();
  test_geev_real<double>();
  test_geev_complex<std::complex<float>>();
  test_geev_complex<std::complex<double>>();
}

// Test LAPACK ggev function for real matrices.
template <typename T>
void test_ggev_real() {
  // use a slightly larger tolerance for generalized eigenvalue problems
  constexpr auto tol = fp_tol<T> * 10;

  for (auto n : nda::range(1, 6)) {
    auto A = nda::matrix<T, nda::F_layout>::rand(n, n);
    auto B = nda::matrix<T, nda::F_layout>::rand(n, n);

    // compute eigenvalues and eigenvectors
    nda::matrix<T, nda::F_layout> A1{A}, B1{B}, vl(n, n), vr(n, n);
    nda::vector<T> alphar(n), alphai(n), beta(n);
    int info = nda::lapack::ggev(A1, B1, alphar, alphai, beta, vl, vr, 'V', 'V');
    EXPECT_EQ(info, 0);

    // convert to complex eigenvalues and eigenvectors
    auto lambda = nda::linalg::get_ggev_eigenvalues(alphar, alphai, beta);
    auto V      = nda::linalg::unpack_eigenvectors(alphai, vr);
    auto U      = nda::linalg::unpack_eigenvectors(alphai, vl);

    // check A * V = B * V * diag(lambda)
    EXPECT_ARRAY_NEAR(A * V, B * V * nda::diag(lambda), tol);

    // check U^H * A = diag(lambda) * U^H * B
    EXPECT_ARRAY_NEAR(nda::dagger(U) * A, nda::diag(lambda) * nda::dagger(U) * B, tol);

    // compute eigenvalues only
    nda::matrix<T, nda::F_layout> A2{A}, B2{B}, vl2{}, vr2{};
    nda::vector<T> alphar2(n), alphai2(n), beta2(n);
    info = nda::lapack::ggev(A2, B2, alphar2, alphai2, beta2, vl2, vr2, 'N', 'N');
    EXPECT_EQ(info, 0);
    EXPECT_ARRAY_NEAR(nda::linalg::get_ggev_eigenvalues(alphar2, alphai2, beta2), lambda, tol);

    // compute eigenvalues and eigenvectors of a view
    if (n > 3) {
      nda::matrix<T, nda::F_layout> A3{A}, B3{B}, vl3(3, 3), vr3(3, 3);
      nda::vector<T> alphar3(3), alphai3(3), beta3(3);
      auto rg = nda::range(3);
      info    = nda::lapack::ggev(A3(rg, rg), B3(rg, rg), alphar3, alphai3, beta3, vl3, vr3, 'N', 'V');
      EXPECT_EQ(info, 0);

      // convert to complex eigenvalues and eigenvectors and check eigenvector equation
      auto lambda3 = nda::linalg::get_ggev_eigenvalues(alphar3, alphai3, beta3);
      auto V3      = nda::linalg::unpack_eigenvectors(alphai3, vr3);
      EXPECT_ARRAY_NEAR(A(rg, rg) * V3, B(rg, rg) * V3 * nda::diag(lambda3), tol);
    }
  }
}

// Test LAPACK ggev function for complex matrices.
template <typename T>
void test_ggev_complex() {
  // use a slightly larger tolerance for generalized eigenvalue problems
  constexpr auto tol = fp_tol<T> * 10;

  for (auto n : nda::range(1, 6)) {
    auto A = nda::matrix<T, nda::F_layout>::rand(n, n);
    auto B = nda::matrix<T, nda::F_layout>::rand(n, n);

    // compute eigenvalues and eigenvectors
    nda::matrix<T, nda::F_layout> A1{A}, B1{B}, U(n, n), V(n, n);
    nda::vector<T> alpha(n), beta(n);
    int info = nda::lapack::ggev(A1, B1, alpha, beta, U, V, 'V', 'V');
    EXPECT_EQ(info, 0);

    // convert to eigenvalues
    auto lambda = nda::linalg::get_ggev_eigenvalues(alpha, beta);

    // check A * V = B * V * diag(lambda)
    EXPECT_ARRAY_NEAR(A * V, B * V * nda::diag(lambda), tol);

    // check U^H * A = diag(lambda) * U^H * B
    EXPECT_ARRAY_NEAR(nda::dagger(U) * A, nda::diag(lambda) * nda::dagger(U) * B, tol);

    // compute eigenvalues only
    nda::matrix<T, nda::F_layout> A2{A}, B2{B}, vl2{}, vr2{};
    nda::vector<T> alpha2(n), beta2(n);
    info = nda::lapack::ggev(A2, B2, alpha2, beta2, vl2, vr2, 'N', 'N');
    EXPECT_EQ(info, 0);
    EXPECT_ARRAY_NEAR(nda::linalg::get_ggev_eigenvalues(alpha2, beta2), lambda, tol);

    // compute eigenvalues and eigenvectors of a view
    if (n > 3) {
      nda::matrix<T, nda::F_layout> A3{A}, B3{B}, U3(3, 3), V3(3, 3);
      nda::vector<T> alpha3(3), beta3(3);
      auto rg = nda::range(3);
      info    = nda::lapack::ggev(A3(rg, rg), B3(rg, rg), alpha3, beta3, U3, V3, 'N', 'V');
      EXPECT_EQ(info, 0);

      // convert to eigenvalues and check eigenvector equation
      auto lambda3 = nda::linalg::get_ggev_eigenvalues(alpha3, beta3);
      EXPECT_ARRAY_NEAR(A(rg, rg) * V3, B(rg, rg) * V3 * nda::diag(lambda3), tol);
    }
  }
}

TEST(NDA, LAPACKGgev) {
  test_ggev_real<float>();
  test_ggev_complex<std::complex<float>>();
  test_ggev_real<double>();
  test_ggev_complex<std::complex<double>>();
}

// Test the rank-3 overloads.
TEST(NDA, LAPACKRank3Overloads) {
  using nda::C_layout, nda::F_layout;
  using value_t = double;

  auto A            = nda::matrix<value_t, F_layout>{{{4, 3}, {6, 3}}};
  constexpr int n_b = 2;
  auto A3           = nda::array<value_t, 3, F_layout>(2, 2, n_b);
  for (int i = 0; i < n_b; ++i) A3(nda::range::all, nda::range::all, i) = A;

  // getrf via base name
  auto ipiv = nda::matrix<int, F_layout>(2, n_b);
  auto info = nda::lapack::getrf(A3, ipiv);
  for (int i = 0; i < n_b; ++i) EXPECT_EQ(info(i), 0);

  // getri via base name (host path: loops over batches)
  auto info2 = nda::lapack::getri(A3, ipiv);
  for (int i = 0; i < n_b; ++i) EXPECT_EQ(info2(i), 0);

  // each slice should now be A^{-1}: det(A) = 4*3 - 3*6 = -6, so A^{-1} = -1/6 * {{3, -3}, {-6, 4}}
  auto A_inv = nda::matrix<value_t, F_layout>{{{-0.5, 0.5}, {1.0, -2.0 / 3.0}}};
  for (int i = 0; i < n_b; ++i) {
    EXPECT_ARRAY_NEAR(nda::matrix<value_t, F_layout>{A3(nda::range::all, nda::range::all, i)}, A_inv, fp_tol<value_t>);
  }
}
