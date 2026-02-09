// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <algorithm>
#include <complex>
#include <concepts>
#include <limits>
#include <tuple>
#include <type_traits>

using namespace std::complex_literals;
using nda::C_layout, nda::F_layout;
using nda::mem::Host, nda::mem::Device, nda::mem::Unified;

// Test the CULAPACK gesvd function.
template <typename T, typename Layout, nda::mem::AddressSpace AS>
void test_gesvd() {
  using matrix_t = nda::matrix<T, Layout>;
  using fp_t     = nda::get_fp_t<T>;

  auto A = matrix_t{{{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {4, 2, 5}, {5, 4, 3}}};
  if constexpr (std::same_as<Layout, C_layout>) {
    // CUDA cannot handle when m < n
    A = matrix_t(nda::transpose(A));
  }
  auto [m, n] = A.shape();

  // expected condition number and spectral norm of A from numpy
  constexpr fp_t cond_A = 6.784414066333698;
  constexpr fp_t norm_A = 12.316822252443167;

  // compute SVD
  auto A_d  = to_addr_space<AS>(A);
  auto U_d  = to_addr_space<AS>(matrix_t(m, m));
  auto VH_d = to_addr_space<AS>(matrix_t(n, n));
  auto s_d  = to_addr_space<AS>(nda::vector<fp_t>(std::min(m, n)));
  nda::lapack::gesvd(A_d, s_d, U_d, VH_d);

  // construct diagonal singular value matrix
  auto s     = nda::to_host(s_d);
  auto Sigma = matrix_t::zeros(A.shape());
  for (auto i : nda::range(std::min(m, n))) Sigma(i, i) = s(i);

  // check condition number and spectral norm
  EXPECT_NEAR(s(0) / s(s.size() - 1), cond_A, fp_tol<T>);
  EXPECT_NEAR(s(0), norm_A, fp_tol<T>);

  // check backward error
  EXPECT_ARRAY_NEAR(A, nda::to_host(U_d) * Sigma * nda::to_host(VH_d), fp_tol<T>);
}

template <typename T, typename Layout>
void test_gesvd_address_spaces() {
  test_gesvd<T, Layout, Device>();
  test_gesvd<T, Layout, Unified>();
}

template <typename T>
void test_gesvd_layouts() {
  test_gesvd_address_spaces<T, C_layout>();
  test_gesvd_address_spaces<T, F_layout>();
}

TEST(NDA, CULAPACKGesvd) {
  test_gesvd_layouts<float>();
  test_gesvd_layouts<std::complex<float>>();
  test_gesvd_layouts<double>();
  test_gesvd_layouts<std::complex<double>>();
}

// Test CULAPACK geqrf, orgqr and ungqr functions.
template <typename T, nda::mem::AddressSpace AS, bool wide_matrix = false>
void test_geqrf_orgqr_ungqr() {
  using matrix_t = nda::matrix<T, F_layout>;

  auto A = matrix_t{{{1, 1, 1}, {3, 2, 4}, {5, 3, 2}, {2, 4, 5}, {4, 5, 3}}};
  if constexpr (wide_matrix) A = matrix_t{transpose(A)};
  auto [m, n] = A.shape();

  // compute QR factorization , i.e. A = Q * R
  auto Q_d   = to_addr_space<AS>(A);
  auto tau_d = to_addr_space<AS>(nda::vector<T>(std::min(m, n)));
  nda::lapack::geqrf(Q_d, tau_d);

  // extract upper triangular matrix R
  auto Q = nda::to_host(Q_d);
  auto R = nda::matrix<T, F_layout>::zeros(std::min(m, n), n);
  for (int i = 0; i < std::min(m, n); ++i) {
    for (int j = i; j < n; ++j) { R(i, j) = Q(i, j); }
  }

  // extract matrix Q with orthonormal columns
  if constexpr (std::floating_point<T>) {
    nda::lapack::orgqr(Q_d(nda::range::all, nda::range(std::min(m, n))), tau_d);
  } else {
    nda::lapack::ungqr(Q_d(nda::range::all, nda::range(std::min(m, n))), tau_d);
  }
  Q = nda::to_host(Q_d);

  EXPECT_ARRAY_NEAR(A, Q(nda::range::all, nda::range(std::min(m, n))) * R, fp_tol<T>);
}

template <typename T, bool wide_matrix = false>
void test_geqrf_orgqr_ungqr_address_spaces() {
  test_geqrf_orgqr_ungqr<T, Device, wide_matrix>();
  test_geqrf_orgqr_ungqr<T, Unified, wide_matrix>();
}

TEST(NDA, CULAPACKGeqrfUngqrAndOrgqr) {
  // tall matrix, i.e. n_rows > n_cols
  test_geqrf_orgqr_ungqr_address_spaces<float>();
  test_geqrf_orgqr_ungqr_address_spaces<std::complex<float>>();
  test_geqrf_orgqr_ungqr_address_spaces<double>();
  test_geqrf_orgqr_ungqr_address_spaces<std::complex<double>>();

  // wide matrix, i.e. n_rows < n_cols
  test_geqrf_orgqr_ungqr_address_spaces<float, true>();
  test_geqrf_orgqr_ungqr_address_spaces<std::complex<float>, true>();
  test_geqrf_orgqr_ungqr_address_spaces<double, true>();
  test_geqrf_orgqr_ungqr_address_spaces<std::complex<double>, true>();
}

// Test the CULAPACK getrs and getrf functions.
template <typename T, typename Layout, nda::mem::AddressSpace AS1, nda::mem::AddressSpace AS2>
void test_getrs_getrf() {
  using matrix_t   = nda::matrix<T, Layout>;
  using f_matrix_t = nda::matrix<T, F_layout>;
  using fp_t       = nda::get_fp_t<T>;

  auto A    = matrix_t{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
  auto Ainv = matrix_t{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}};
  if constexpr (nda::is_complex_v<T>) {
    A *= T{1i};
    Ainv /= T{1i};
  }
  auto B = f_matrix_t{{1, 5}, {4, 5}, {3, 6}};

  // tolerance based on condition number: cond(A) ~ 332, ||Ainv||_max = 24
  // error ~ cond(A) * ||Ainv||_max * eps => use eps * 10000 as tolerance
  constexpr auto tol = std::numeric_limits<fp_t>::epsilon() * 10000;

  // solve A * X = B using getrf and getrs
  auto A_d    = to_addr_space<AS1>(A);
  auto B_d    = to_addr_space<AS2>(B);
  auto ipiv_d = to_addr_space<AS1>(nda::array<int, 1>(3));
  nda::lapack::getrf(A_d, ipiv_d);
  nda::lapack::getrs(A_d, B_d, ipiv_d);
  auto X = matrix_t{nda::to_host(B_d)};
  EXPECT_ARRAY_NEAR(A * X, B, tol);
  EXPECT_ARRAY_NEAR(Ainv * B, X, tol);

  // solve A^T * X = B using getrf and getrs
  A_d = A;
  B_d = B;
  nda::lapack::getrf(A_d, ipiv_d);
  nda::lapack::getrs(nda::transpose(A_d), B_d, ipiv_d);
  X = matrix_t{nda::to_host(B_d)};
  EXPECT_ARRAY_NEAR(nda::transpose(A) * X, B, tol);
  EXPECT_ARRAY_NEAR(nda::transpose(Ainv) * B, X, tol);

  // solve A^H * X = B using getrf and getrs
  if constexpr (nda::blas_lapack::has_F_layout<matrix_t>) {
    A_d = A;
    B_d = B;
    nda::lapack::getrf(A_d, ipiv_d);
    nda::lapack::getrs(nda::dagger(A_d), B_d, ipiv_d);
    X = matrix_t{nda::to_host(B_d)};
    EXPECT_ARRAY_NEAR(nda::dagger(A) * X, B, tol);
    EXPECT_ARRAY_NEAR(nda::dagger(Ainv) * B, X, tol);
  }

  // solve A * x = b using getrf and getrs
  A_d      = A;
  auto b   = B(nda::range::all, 0);
  auto b_d = to_addr_space<AS2>(nda::vector<T>{b});
  nda::lapack::getrf(A_d, ipiv_d);
  nda::lapack::getrs(A_d, b_d, ipiv_d);
  auto x = nda::to_host(b_d);
  EXPECT_ARRAY_NEAR(A * x, b, tol);
  EXPECT_ARRAY_NEAR(Ainv * b, x, tol);
}

template <typename T, typename Layout>
void test_getrs_getrf_address_spaces() {
  test_getrs_getrf<T, Layout, Device, Device>();
  test_getrs_getrf<T, Layout, Device, Unified>();
  test_getrs_getrf<T, Layout, Unified, Device>();
  test_getrs_getrf<T, Layout, Unified, Unified>();
  test_getrs_getrf<T, Layout, Unified, Host>();
  test_getrs_getrf<T, Layout, Host, Unified>();
}

template <typename T>
void test_getrs_getrf_layouts() {
  test_getrs_getrf_address_spaces<T, C_layout>();
  test_getrs_getrf_address_spaces<T, F_layout>();
}

TEST(NDA, CULAPACKGetrsAndGetrf) {
  test_getrs_getrf_layouts<float>();
  test_getrs_getrf_layouts<std::complex<float>>();
  test_getrs_getrf_layouts<double>();
  test_getrs_getrf_layouts<std::complex<double>>();
}

template <typename T, nda::mem::AddressSpace AS>
void test_rectangular_getrf() {
  using namespace nda::blas_lapack;
  auto A      = nda::matrix<T, F_layout>{{1, 5}, {4, 5}, {3, 6}};
  auto AT     = nda::matrix<T, F_layout>(nda::transpose(A));
  auto A_c    = nda::matrix<T, C_layout>{A};
  auto AT_c   = nda::matrix<T, C_layout>{AT};
  auto ipiv_d = nda::array<int, 1, C_layout, nda::heap<AS>>(2);

  // get the matrices P, L, U from getrf output
  auto get_plu = [](auto const &M, auto const &ipiv, int m, int n) {
    using layout_t   = std::conditional_t<has_C_layout<decltype(M)>, C_layout, F_layout>;
    auto P           = nda::matrix<T, layout_t>::zeros(m, m);
    auto L           = nda::matrix<T, layout_t>::zeros(m, m);
    auto U           = nda::matrix<T, layout_t>::zeros(m, n);
    nda::diagonal(P) = 1;
    nda::diagonal(L) = 1;
    for (int i = 0; i < ipiv.size(); ++i) deep_swap(P(i, nda::range::all), P(ipiv(i) - 1, nda::range::all));
    for (int i = 0; i < m; ++i) {
      L(i, nda::range(i))    = (has_C_layout<decltype(M)> ? M(nda::range(i), i) : M(i, nda::range(i)));
      U(i, nda::range(i, n)) = (has_C_layout<decltype(M)> ? M(nda::range(i, n), i) : M(i, nda::range(i, n)));
    }
    return std::make_tuple(P, L, U);
  };

  // LU decomposition for 3x2 Fortran layout matrix
  auto LU_f_32 = to_addr_space<AS>(A);
  nda::lapack::getrf(LU_f_32, ipiv_d);
  auto [P_f_32, L_f_32, U_f_32] = get_plu(nda::to_host(LU_f_32), nda::to_host(ipiv_d), 3, 2);
  EXPECT_ARRAY_NEAR(P_f_32 * A, L_f_32 * U_f_32, fp_tol<T>);

  // LU decomposition for 2x3 Fortran layout matrix
  auto LU_f_23 = to_addr_space<AS>(AT);
  nda::lapack::getrf(LU_f_23, ipiv_d);
  auto [P_f_23, L_f_23, U_f_23] = get_plu(nda::to_host(LU_f_23), nda::to_host(ipiv_d), 2, 3);
  EXPECT_ARRAY_NEAR(P_f_23 * AT, L_f_23 * U_f_23, fp_tol<T>);

  // LU decomposition for 3x2 C layout matrix
  auto LU_c_32 = to_addr_space<AS>(A_c);
  nda::lapack::getrf(LU_c_32, ipiv_d);
  auto [P_c_32, L_c_32, U_c_32] = get_plu(nda::to_host(LU_c_32), nda::to_host(ipiv_d), 2, 3);
  EXPECT_ARRAY_NEAR(P_c_32 * nda::transpose(A_c), L_c_32 * U_c_32, fp_tol<T>);

  // LU decomposition for 2x3 C layout matrix
  auto LU_c_23 = to_addr_space<AS>(AT_c);
  nda::lapack::getrf(LU_c_23, ipiv_d);
  auto [P_c_23, L_c_23, U_c_23] = get_plu(nda::to_host(LU_c_23), nda::to_host(ipiv_d), 3, 2);
  EXPECT_ARRAY_NEAR(P_c_23 * nda::transpose(AT_c), L_c_23 * U_c_23, fp_tol<T>);
}

template <typename T>
void test_rectangular_getrf_address_spaces() {
  test_rectangular_getrf<T, Device>();
  test_rectangular_getrf<T, Unified>();
}

TEST(NDA, CULAPACKGetrfWithRectangularMatrix) {
  test_rectangular_getrf_address_spaces<float>();
  test_rectangular_getrf_address_spaces<std::complex<float>>();
  test_rectangular_getrf_address_spaces<double>();
  test_rectangular_getrf_address_spaces<std::complex<double>>();
}
