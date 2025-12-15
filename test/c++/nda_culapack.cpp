// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <algorithm>
#include <complex>

using namespace nda;

// Test the CULAPACK gesvd function.
template <typename value_t>
void test_gesvd(double tol = 1e-14) {
  using matrix_t = nda::matrix<value_t, nda::F_layout>;

  auto A      = matrix_t{{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}}};
  auto [m, n] = A.shape();

  auto U  = matrix_t(m, m);
  auto VT = matrix_t(n, n);

  auto s = nda::vector<nda::remove_complex_t<value_t>>(std::min(m, n));

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
  EXPECT_ARRAY_NEAR(A, U * Sigma * VT, tol);
}

TEST(NDA, CULAPACKGesvd) {
  test_gesvd<float>(1e-5);
  test_gesvd<double>();
  test_gesvd<std::complex<float>>(1e-5);
  test_gesvd<std::complex<double>>();
}

// Test the CULAPACK getrs and getrf functions.
template <typename value_t>
void test_getrs_getri_getrf(double tol = 1e-10) {
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
  EXPECT_ARRAY_NEAR(matrix_t{A * X2}, B, tol);
  EXPECT_ARRAY_NEAR(X1, X2, tol);

  nda::lapack::getri(A_d, ipiv);
  EXPECT_ARRAY_NEAR(Ainv, to_host(A_d), tol);
}

TEST(NDA, CULAPACKGetrsAndGetrf) {
  test_getrs_getri_getrf<float>(1e-4);
  test_getrs_getri_getrf<double>();
  test_getrs_getri_getrf<std::complex<float>>(1e-4);
  test_getrs_getri_getrf<std::complex<double>>();
}

// Test batched LAPACK getrf and getri functions.
template <typename value_t, nda::mem::AddressSpace AdrSp>
void test_batched_getrf_getrs_getri(double tol = 1e-10) {
  using namespace nda;
  using matrix_t = nda::matrix<value_t, nda::F_layout>;
  auto all       = nda::range::all;

  auto A0 = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto B0 = matrix_t{{1, 5}, {4, 5}, {3, 6}};

  // solve A0 * x = B0 using the exact matrix inverse
  auto A0inv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  auto X1    = matrix_t{A0inv * B0};
  EXPECT_ARRAY_NEAR(matrix_t{A0 * X1}, B0);

  array<value_t, 3, F_layout> A(3, 3, 5);
  for (int b = 0; b < A.extent(2); ++b) A(all, all, b) = A0;

  auto Aref(A);
  auto Ainv(A);
  array<int, 1> ipiv_ref(3);
  for (int i = 0; i < Aref.extent(2); ++i) {
    int info = lapack::getrf(Aref(range::all, range::all, i), ipiv_ref);
    EXPECT_TRUE(info == 0);
    Ainv(range::all, range::all, i) = Aref(range::all, range::all, i);
    info                            = lapack::getri(Ainv(range::all, range::all, i), ipiv_ref);
    EXPECT_TRUE(info == 0);
    EXPECT_ARRAY_NEAR(A0inv, Ainv(range::all, range::all, i), tol);
  }

  array<value_t, 3, F_layout, heap<AdrSp>> X(A);
  array<int, 2, F_layout, heap<AdrSp>> ipiv(3, 5);

  // getrf
  {
    auto info = lapack::getrf(X, ipiv);
    EXPECT_TRUE(std::all_of(info.begin(), info.end(), [](auto &&a) { return a == 0; }));
    EXPECT_ARRAY_NEAR(Aref, nda::to_host(X), tol);
  }

  // getrs
  {
    array<value_t, 3, F_layout, heap<AdrSp>> Y(3, 2, 5);
    for (int b = 0; b < Y.extent(2); ++b) Y(all, all, b) = B0;
    auto info = lapack::getrs(X, Y, ipiv);
    EXPECT_TRUE(std::all_of(info.begin(), info.end(), [](auto &&a) { return a == 0; }));
    auto Y_h = to_host(Y);
    for (int i = 0; i < A.extent(2); ++i) {
      auto X2 = Y_h(all, all, i);
      EXPECT_ARRAY_NEAR(X1, X2, tol);
    }
  }

  // getri
  {
    auto info = lapack::getri(X, ipiv);
    EXPECT_TRUE(std::all_of(info.begin(), info.end(), [](auto &&a) { return a == 0; }));
    EXPECT_ARRAY_NEAR(Ainv, nda::to_host(X), tol);
  }
}

TEST(NDA, CULAPACKBatchedGetrsGetrfAndGetri) {
  using nda::mem::Device;
  test_batched_getrf_getrs_getri<float, Device>(1e-4);
  test_batched_getrf_getrs_getri<double, Device>();
  test_batched_getrf_getrs_getri<std::complex<float>, Device>(1e-4);
  test_batched_getrf_getrs_getri<std::complex<double>, Device>();
}

// Test LAPACK geqrf, orgqr and ungqr functions.
template <typename value_t, bool wide_matrix = false>
void test_geqrf_gqr(double tol) {
  using matrix_t = matrix<value_t, F_layout>;
  auto A         = matrix_t{{{1, 1, 1}, {3, 2, 4}, {5, 3, 2}, {2, 4, 5}, {4, 5, 3}}};
  if (wide_matrix) A = matrix_t{transpose(A)};
  auto [m, n] = A.shape();

  // compute QR factorization, i.e. A = Q * R
  auto tau = nda::vector<value_t>(std::min(m, n));
  auto Q   = matrix_t{A};
  lapack::geqrf(Q, tau);

  // extract upper triangular matrix R
  auto R = nda::matrix<value_t, F_layout>::zeros(std::min(m, n), n);
  for (int i = 0; i < std::min(m, n); ++i) {
    for (int j = i; j < n; ++j) { R(i, j) = Q(i, j); }
  }

  // extract matrix Q with orthonormal columns
  lapack::gqr(Q, tau);
  EXPECT_ARRAY_NEAR(A, Q(range::all, range(std::min(m, n))) * R, tol);

  // now test Device
  auto Q_d   = to_device(A);
  auto tau_d = to_device(tau);

  lapack::geqrf(Q_d, tau_d);
  lapack::gqr(Q_d, tau_d);
  auto Q_h = to_host(Q_d);
  EXPECT_ARRAY_NEAR(A, Q_h(range::all, range(std::min(m, n))) * R, tol);
}
TEST(NDA, CULAPACKGeqrfGqr) {
  // tall matrix, i.e. n_rows > n_cols
  test_geqrf_gqr<float>(1e-5);
  test_geqrf_gqr<double>(1e-14);
  test_geqrf_gqr<std::complex<float>>(1e-5);
  test_geqrf_gqr<std::complex<double>>(1e-14);

  // wide matrix, i.e. n_rows < n_cols
  test_geqrf_gqr<float, true>(1e-5);
  test_geqrf_gqr<double, true>(1e-14);
  test_geqrf_gqr<std::complex<float>, true>(1e-5);
  test_geqrf_gqr<std::complex<double>, true>(1e-14);
}

template <typename value_t, bool wide_matrix = false>
void test_geqrf_gqr_batched(double tol = 1e-10) {
  auto all = nda::range::all;
  array<value_t, 3, F_layout> A(5, 3, 4);
  A() = rand<remove_complex_t<value_t>, int, 3>({5, 3, 4});
  if constexpr (is_complex_v<value_t>) A() += value_t(0.0, 1.0) * rand<remove_complex_t<value_t>, int, 3>({5, 3, 4});
  if (wide_matrix) {
    array<value_t, 3, F_layout> A_(3, 5, 4);
    for (int i = 0; i < A.extent(2); ++i) A_(all, all, i) = transpose(A(all, all, i));
    A = A_;
  }

  auto tau_ref = nda::array<value_t, 2, F_layout>(3, 4);
  auto Qref(A);
  for (int i = 0; i < Qref.extent(2); ++i) {
    lapack::geqrf(Qref(range::all, range::all, i), tau_ref(range::all, i));
    lapack::gqr(Qref(range::all, range::all, i), tau_ref(range::all, i));
  }

  array<value_t, 3, F_layout, heap<mem::Device>> Q(A);
  array<value_t, 2, F_layout, heap<mem::Device>> tau(3, 4);

  lapack::geqrf(Q, tau);
  lapack::gqr(Q, tau);
  auto Q_h = to_host(Q);
  EXPECT_ARRAY_NEAR(Qref, Q_h, tol);
}
TEST(NDA, CULAPACKGeqrfGqrBatched) {
  // tall matrix, i.e. n_rows > n_cols
  test_geqrf_gqr_batched<float>(1e-6);
  test_geqrf_gqr_batched<double>();
  test_geqrf_gqr_batched<std::complex<float>>(1e-6);
  test_geqrf_gqr_batched<std::complex<double>>();

  // wide matrix, i.e. n_rows < n_cols
  test_geqrf_gqr_batched<float, true>(1e-6);
  test_geqrf_gqr_batched<double, true>();
  test_geqrf_gqr_batched<std::complex<float>, true>(1e-6);
  test_geqrf_gqr_batched<std::complex<double>, true>();
}
